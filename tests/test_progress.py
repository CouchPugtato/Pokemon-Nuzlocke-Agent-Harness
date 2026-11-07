import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from pokemon_harness.agent.prompts import messages
from pokemon_harness.cli import main
from pokemon_harness.config import load_config
from pokemon_harness.control.loop_detection import cycle_signals, image_features, input_signature
from pokemon_harness.demo import DemoAgent, create_images, demo_outcome_reviews
from pokemon_harness.emulator.interface import ReplayEmulator
from pokemon_harness.memory.database import Memory
from pokemon_harness.memory.progress import progress_settings
from pokemon_harness.runner import run
from pokemon_harness.telemetry.backup import create_backup
from pokemon_harness.telemetry.recorder import write_json
from pokemon_harness.telemetry.restore import restore_backup


def strategy(kind="interact", target="visible object"):
    return {"kind": kind, "target": target, "intended_effect": "Test visible response"}


class SignalTests(unittest.TestCase):
    def test_visual_cycles_cover_periods_one_through_eight(self):
        for period in range(1, 9):
            frames = [{"image_hash": str(index % period), "observation_id": index + 1,
                       "signature": (("press", ("A",)),)} for index in range(period * 3)]
            with self.subTest(period=period):
                signals = cycle_signals(frames)
                signal = next(signal for signal in signals if signal["kind"] == "exact_visual")
                self.assertEqual(signal["period"], period)
                self.assertEqual(len(signal["evidence_observation_ids"]), period * 3)

    def test_short_windows_and_missing_results_do_not_create_visual_cycles(self):
        frame = {"image_hash": None, "observation_id": None, "signature": (("wait", ()),)}
        self.assertEqual(cycle_signals([frame, frame]), [])
        self.assertEqual([signal["kind"] for signal in cycle_signals([frame] * 3)], ["input_cycle"])

    def test_coarse_similarity_checks_scene_and_dialogue_separately(self):
        first = {"scene": [20] * 64, "dialogue": [80] * 64}
        second = {"scene": [21] * 64, "dialogue": [79] * 64}
        frames = [{"image_hash": str(index), "features": features, "observation_id": index + 1,
                   "signature": (("press", ("A",)),)} for index, features in enumerate([first, second, first])]
        self.assertIn("similar_visual", [signal["kind"] for signal in cycle_signals(frames)])
        frames[1]["features"] = {"scene": [21] * 64, "dialogue": [120] * 64}
        self.assertNotIn("similar_visual", [signal["kind"] for signal in cycle_signals(frames)])

    def test_features_and_signatures_use_pixels_and_buttons_without_timing(self):
        png = io.BytesIO()
        Image.new("RGB", (12, 12), (30, 30, 30)).save(png, format="PNG")
        self.assertEqual(image_features(png.getvalue()), {"scene": [30] * 64, "dialogue": [30] * 64})
        left = {"kind": "press", "buttons": ["B", "A"], "hold_frames": 6, "settle_frames": 0}
        right = dict(left, buttons=["A", "B"], hold_frames=30, settle_frames=20)
        self.assertEqual(input_signature([left]), input_signature([right]))


class ProgressTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "memory.sqlite"
        self.memory = Memory(self.path)
        self.addCleanup(self.memory.close)
        self.config = {"objective": "Beat first gym", "rules": [], "progress": {
            "window": 8, "max_cycle_period": 1, "no_progress_attempts": 3}}
        self.memory.initialize(self.config)

    def value(self, button="A"):
        observation = self.memory.observe("before.png", "same", "test")
        context = {"observation_id": observation, "outcome_review": self.memory.outcomes.context()}
        value = DemoAgent().decide(messages(context, b"png"))[0]
        value["action"].update(buttons=[button])
        value["strategy"] = strategy()
        progress = self.memory.monitor_progress()
        if progress["requires_recovery"]:
            value["recovery_plan"] = {"episode_id": progress["episode_id"],
                                      "explanation": "Repeated screen without advancement so test another input"}
        return observation, value

    def attempt(self, button="A", digest="same", prior_assessment="uncertain"):
        observation, value = self.value(button)
        for review in value["outcome_reviews"]:
            for step in review["steps"]:
                step["assessment"] = prior_assessment
        turn = self.memory.plan(observation, value)
        self.memory.status(turn, "executing")
        self.memory.step_status(turn, 1, "executing")
        after = self.memory.observe("after.png", digest, "test")
        self.memory.step_status(turn, 1, "completed", after)
        self.memory.status(turn, "completed")
        return self.memory.monitor_progress()

    def stalled(self):
        for _ in range(3):
            context = self.attempt()
        self.assertEqual(context["mode"], "reassess")
        return context

    def finish_review(self, assessment="uncertain"):
        observation = self.memory.observe("review.png", "review", "test")
        reviews = demo_outcome_reviews({"observation_id": observation, "outcome_review": self.memory.outcomes.context()})
        for review in reviews:
            for step in review["steps"]:
                step["assessment"] = assessment
        self.memory.review_outcomes(observation, reviews)
        return self.memory.monitor_progress()

    def test_stagnation_and_episode_survive_reopen_and_subgoal_changes(self):
        context = self.stalled()
        observation, value = self.value("B")
        value["working_state_update"] = {"expected_revision": 0, "location": None, "uncertainties": [],
            "next_check": "Inspect", "subgoal_change": {"operation": "start", "id": "renamed",
            "definition": {"description": "Another label", "objective_relation": "Gym",
                           "completion_condition": "Visible progress", "abandonment_condition": "Stuck"},
            "reason": "Reassess", "evidence_observation_ids": []}}
        turn = self.memory.plan(observation, value)
        self.memory.status(turn, "stopped")
        with contextlib.closing(Memory(self.path)) as reopened:
            reopened.initialize(self.config)
            current = reopened.monitor_progress()
            self.assertEqual(current["episode_id"], context["episode_id"])
            self.assertEqual(current["recovery_attempts_used"], 1)
            self.assertEqual(current["no_progress_attempts"], 3)
            self.assertEqual(reopened.working.current()["subgoal"]["id"], "renamed")

    def test_retiming_and_renaming_do_not_create_an_alternative(self):
        self.stalled()
        observation, value = self.value()
        value["action"]["hold_frames"] = 30
        value["strategy"] = strategy("inspect_menu", "a new target label")
        with self.assertRaisesRegex(ValueError, "rename or retime"):
            self.memory.plan(observation, value)
        self.assertEqual(self.memory.monitor_progress()["recovery_attempts_used"], 0)
        self.assertEqual(len(self.memory.recent_turns()), 3)

    def test_recovery_requires_current_episode_single_move_and_safe_strategy(self):
        self.stalled()
        observation, value = self.value("B")
        for kind in ("missing", "stale", "sequence", "battle", "capture", "flee", "unknown_scene"):
            invalid = copy.deepcopy(value)
            if kind == "missing":
                invalid["recovery_plan"] = None
            elif kind == "stale":
                invalid["recovery_plan"]["episode_id"] += 1
            elif kind == "sequence":
                invalid["additional_actions"] = [dict(invalid["action"])]
            elif kind == "unknown_scene":
                invalid["scene"] = "unknown"
            else:
                invalid["strategy"]["kind"] = kind
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.memory.plan(observation, invalid)
        self.assertEqual(self.memory.monitor_progress()["recovery_attempts_used"], 0)

    def test_exhaustion_waits_for_final_review_then_opens_one_durable_case(self):
        self.stalled()
        self.attempt("B")
        context = self.attempt("START")
        self.assertEqual(context["recovery_attempts_remaining"], 0)
        self.assertEqual(context["mode"], "test_alternative")
        self.assertEqual(self.memory.escalations.cases(), [])
        stopped = self.finish_review()
        self.assertEqual(stopped["mode"], "stopped")
        case = self.memory.escalations.cases()[0]
        self.assertTrue(case["blocking"])
        self.assertEqual(case["details"]["kind"], "stalled_progress")
        self.assertEqual(self.memory.monitor_progress()["case_id"], case["id"])
        self.assertEqual(len(self.memory.escalations.cases()), 1)

    def test_interruption_consumes_reserved_test_and_cannot_reuse_it(self):
        self.stalled()
        observation, value = self.value("B")
        self.memory.plan(observation, value)
        self.memory.recover()
        self.assertEqual(self.memory.monitor_progress()["recovery_attempts_used"], 1)
        observation, value = self.value("B")
        with self.assertRaisesRegex(ValueError, "different input"):
            self.memory.plan(observation, value)
        observation, value = self.value("START")
        self.memory.plan(observation, value)
        self.memory.recover()
        self.assertEqual(self.memory.monitor_progress()["mode"], "stopped")

    def test_changed_pixels_without_model_progress_do_not_reset_counter(self):
        self.attempt("A", "one")
        self.attempt("B", "two")
        context = self.attempt("START", "three")
        self.assertEqual(context["no_progress_attempts"], 3)
        self.assertTrue(context["requires_recovery"])

    def test_same_pixels_with_claimed_progress_do_not_reset_counter(self):
        self.attempt(prior_assessment="progress_observed")
        self.attempt(prior_assessment="progress_observed")
        context = self.attempt(prior_assessment="progress_observed")
        self.assertEqual(context["no_progress_attempts"], 3)
        self.assertTrue(context["requires_recovery"])

    def test_relevant_changed_evidence_can_recover_and_preserves_episode_history(self):
        initial = self.stalled()
        self.attempt("B", "different")
        context = self.finish_review("progress_observed")
        self.assertEqual(context["mode"], "recovered")
        self.assertEqual(context["episode_id"], initial["episode_id"])
        self.assertEqual(context["no_progress_attempts"], 0)
        self.assertEqual(context["recovery_attempts_used"], 1)
        self.assertEqual(self.memory.progress.history()[0]["kind"], "recovered")

    def test_last_recovery_test_can_succeed_before_a_case_is_opened(self):
        self.stalled()
        self.attempt("B")
        self.attempt("START", "different")
        context = self.finish_review("progress_observed")
        self.assertEqual(context["mode"], "recovered")
        self.assertEqual(self.memory.escalations.cases(), [])

    def test_visual_cycle_still_triggers_when_model_claims_progress(self):
        memory = Memory(self.root / "cycle.sqlite")
        self.addCleanup(memory.close)
        memory.initialize({"objective": "Test", "rules": [], "progress": {
            "window": 8, "max_cycle_period": 2, "no_progress_attempts": 8, "strategy_attempts": 8}})
        for index in range(6):
            observation = memory.observe("before.png", "different", "test")
            context = {"observation_id": observation, "outcome_review": memory.outcomes.context()}
            value = DemoAgent().decide(messages(context, b"png"))[0]
            for review in value["outcome_reviews"]:
                for step in review["steps"]:
                    step["assessment"] = "progress_observed"
            turn = memory.plan(observation, value)
            after = memory.observe("after.png", str(index % 2), "test")
            memory.step_status(turn, 1, "completed", after)
            memory.status(turn, "completed")
            state = memory.monitor_progress()
        self.assertEqual(state["no_progress_attempts"], 1)
        self.assertTrue(state["requires_recovery"])
        self.assertIn("exact_visual", [signal["kind"] for signal in state["signals"]])

    def test_revising_older_review_does_not_create_fresh_recovery_evidence(self):
        self.stalled()
        self.attempt("B")
        older = self.memory.outcomes.history()[-1]
        review = {key: older[key] for key in ("turn_id", "steps")}
        review["expected_revision"] = older["revision"]
        review["steps"][0]["assessment"] = "progress_observed"
        observation = self.memory.observe("review.png", "fresh", "test")
        self.memory.review_outcomes(observation, [review])
        self.assertEqual(self.memory.monitor_progress()["mode"], "test_alternative")

    def test_similarity_and_repeated_inputs_override_claimed_progress(self):
        memory = Memory(self.root / "animated.sqlite")
        self.addCleanup(memory.close)
        memory.initialize({"objective": "Test", "rules": [], "progress": {
            "window": 8, "max_cycle_period": 1, "no_progress_attempts": 8, "strategy_attempts": 8}})
        png = io.BytesIO()
        Image.new("RGB", (12, 12), (30, 30, 30)).save(png, format="PNG")
        for index in range(3):
            observation = memory.observe("before.png", "before", "test")
            context = {"observation_id": observation, "outcome_review": memory.outcomes.context()}
            value = DemoAgent().decide(messages(context, b"png"))[0]
            for review in value["outcome_reviews"]:
                review["steps"][0]["assessment"] = "progress_observed"
            turn = memory.plan(observation, value)
            after = memory.observe("after.png", str(index), "test", png.getvalue())
            memory.step_status(turn, 1, "completed", after)
            memory.status(turn, "completed")
            state = memory.monitor_progress()
        self.assertEqual(state["no_progress_attempts"], 1)
        self.assertTrue(state["requires_recovery"])
        self.assertNotIn("exact_visual", [signal["kind"] for signal in state["signals"]])
        self.assertIn("similar_visual", [signal["kind"] for signal in state["signals"]])

    def test_user_override_is_explicit_and_creates_a_new_bounded_episode(self):
        self.stalled()
        self.attempt("B")
        self.attempt("START")
        stopped = self.finish_review()
        self.memory.escalations.resolve(stopped["case_id"], "user-assertion", "User inspected the emulator",
                                        allow_continue=True)
        current = self.memory.monitor_progress()
        self.assertGreater(current["episode_id"], stopped["episode_id"])
        self.assertEqual(current["recovery_attempts_remaining"], 2)
        self.assertTrue(current["requires_recovery"])
        self.assertEqual(self.memory.escalations.validity(), "unverifiable")

    def test_policy_is_validated_and_cannot_reset_an_existing_run(self):
        for supplied in ({"max_recovery_attempts": 0}, {"window": 8}, {"cycle_repetitions": True}, {"disable": True}):
            with self.subTest(supplied=supplied), self.assertRaises(ValueError):
                progress_settings({"progress": supplied})
        changed = copy.deepcopy(self.config)
        changed["progress"]["no_progress_attempts"] = 4
        with self.assertRaisesRegex(ValueError, "saved run"):
            self.memory.initialize(changed)


class ProgressIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        config_path = self.root / "config.json"
        write_json(config_path, {"run_dir": "run", "objective": "Test", "model": {"name": "test"},
            "emulator": {"backend": "replay"}, "max_sequence_length": 10, "sequence_delay_ms": 0,
            "progress": {"window": 8, "max_cycle_period": 1, "no_progress_attempts": 3}})
        self.config = load_config(config_path)
        self.directory = Path(self.config["run_dir"])
        self.images = create_images(self.root / "fixtures")

    def test_repeated_sequence_is_cancelled_and_recovery_context_uses_attempted_steps(self):
        requests = []

        class Agent:
            def decide(self, request):
                requests.append(request)
                value, trace = DemoAgent().decide(request)
                if len(requests) == 1:
                    value["additional_actions"] = [dict(value["action"])] * 8
                else:
                    value["action"].update(kind="stop", buttons=[])
                return value, trace

        emulator = ReplayEmulator(self.images)
        result = run(self.config, emulator, Agent(), turns=2)
        self.assertEqual(result["completed_actions"], 4)
        self.assertEqual(len(emulator.actions), 4)
        context = json.loads(requests[1][1]["content"][0]["text"])
        self.assertTrue(context["progress"]["requires_recovery"])
        self.assertEqual([step["input_status"] for step in context["outcome_review"]["next_plan"]["steps"]],
                         ["completed"] * 4 + ["cancelled"] * 5)

    def test_exhausted_case_survives_backup_restore_and_blocks_model_startup(self):
        requests = []

        class Agent:
            def decide(self, request):
                context = json.loads(request[1]["content"][0]["text"])
                requests.append(context)
                value, trace = DemoAgent().decide(request)
                value["strategy"] = strategy()
                value["action"].update(buttons=["A"])
                progress = context["progress"]
                if progress["requires_recovery"]:
                    used = progress["recovery_attempts_used"]
                    if progress["recovery_attempts_remaining"]:
                        value["action"]["buttons"] = [["B"], ["START"]][used]
                        value["recovery_plan"] = {"episode_id": progress["episode_id"],
                                                  "explanation": "Inspect a different control"}
                    else:
                        value["action"].update(kind="stop", buttons=[])
                return value, trace

        run(self.config, ReplayEmulator(self.images), Agent(), turns=10)
        with contextlib.closing(Memory(self.directory / "memory.sqlite")) as memory:
            state = memory.monitor_progress()
            self.assertEqual(state["mode"], "stopped")
            history = memory.progress.history()
        backup = create_backup(self.directory, self.root / "backups")
        target = self.root / "restored"
        restore_backup(backup, target)
        with contextlib.closing(Memory(target / "memory.sqlite")) as restored:
            self.assertEqual(restored.monitor_progress()["mode"], "stopped")
            self.assertEqual(restored.progress.history(), history)
        with patch("pokemon_harness.cli.LMStudio") as model, patch("pokemon_harness.cli.emulator_for") as emulator:
            self.assertEqual(main(["resume", str(self.directory), "--turns", "1"]), 1)
        model.assert_not_called()
        emulator.assert_not_called()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["progress", str(self.directory), "--history", "--limit", "1"]), 0)
        self.assertEqual(json.loads(output.getvalue())["history"], history[:1])
