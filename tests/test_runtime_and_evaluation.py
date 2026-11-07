import contextlib
import hashlib
import io
import json
import sqlite3
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from pokemon_harness.agent.prompts import messages
from pokemon_harness.cli import main
from pokemon_harness.config import load_config
from pokemon_harness.control.timing import RuntimeBudget, TimingLimit, bounded_decision, timing_settings
from pokemon_harness.demo import DemoAgent, create_images, demo_outcome_reviews
from pokemon_harness.emulator.interface import ReplayEmulator
from pokemon_harness.evaluation import evaluate, load_labels, validate_context_gates
from pokemon_harness.memory.database import Memory
from pokemon_harness.runner import run
from pokemon_harness.telemetry.recorder import write_json


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.memory = Memory(self.root / "memory.sqlite")
        self.addCleanup(self.memory.close)
        self.memory.initialize({"objective": "Test", "rules": []})

    def test_runtime_usage_survives_sessions_and_cannot_reset_by_changing_policy(self):
        clock = Mock(return_value=0)
        config = {"timing_budgets": {"run_seconds": 5, "session_seconds": 4}}
        budget = RuntimeBudget(self.memory, config, clock)
        budget.start()
        clock.return_value = 3
        budget.finish()
        budget = RuntimeBudget(self.memory, config, clock)
        budget.start()
        clock.return_value = 5
        with self.assertRaises(TimingLimit):
            budget.check()
        budget.finish()
        with self.assertRaises(TimingLimit):
            RuntimeBudget(self.memory, config, clock).start()
        with self.assertRaisesRegex(ValueError, "policy"):
            RuntimeBudget(self.memory, {"timing_budgets": {"run_seconds": 100}}).start()

    def test_crashed_sessions_are_conservatively_charged_before_resume(self):
        budget = RuntimeBudget(self.memory, {"timing_budgets": {"session_seconds": 2, "run_seconds": 2}})
        budget.start()
        with self.memory.connection:
            self.memory.connection.execute("UPDATE runtime_sessions SET started_at=?", ((datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat(),))
        with self.assertRaises(TimingLimit):
            RuntimeBudget(self.memory, {"timing_budgets": {"session_seconds": 2, "run_seconds": 2}}).start()
        row = self.memory.connection.execute("SELECT elapsed_seconds,closed FROM runtime_sessions").fetchone()
        self.assertEqual(tuple(row), (2, 1))

    def test_scene_freshness_and_unknown_age_block_input(self):
        budget = RuntimeBudget(self.memory, {"timing_budgets": {"scene_age_seconds": {"battle": 2}}})
        budget.start()
        self.addCleanup(budget.finish)
        budget.fresh(3, "menu")
        for age in (3, None):
            with self.assertRaises(TimingLimit):
                budget.fresh(age, "battle")

    def test_generation_timeout_abandons_the_worker_without_waiting(self):
        release, finished = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        class Agent:
            def decide(self, request):
                release.wait(2)
                finished.set()
                return {}, {}

        with self.assertRaises(TimingLimit):
            bounded_decision(Agent(), [], 0.02)
        self.assertFalse(finished.is_set())
        release.set()
        self.assertTrue(finished.wait(1))

    def test_invalid_timing_configuration_is_rejected(self):
        for settings in ({"run_seconds": 0}, {"session_seconds": True}, {"generation_seconds": 999999},
                         {"scene_age_seconds": {"battle": 999}}, {"unknown": 1}):
            with self.assertRaises(ValueError):
                timing_settings({"timing_budgets": settings})

    def test_explicit_extension_adds_time_without_resetting_usage(self):
        clock = Mock(return_value=0)
        config = {"timing_budgets": {"run_seconds": 2}}
        budget = RuntimeBudget(self.memory, config, clock)
        budget.start()
        clock.return_value = 2
        budget.finish()
        budget.extend(3, "Continue supervised run")
        budget = RuntimeBudget(self.memory, config, clock)
        budget.start()
        self.assertEqual(budget.used, 2)
        self.assertEqual(budget.run_limit, 5)
        budget.check()
        budget.finish()
        self.assertEqual(self.memory.connection.execute("SELECT reason FROM runtime_extensions").fetchone()[0], "Continue supervised run")


class RepetitionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "memory.sqlite"
        self.memory = Memory(self.path)
        self.addCleanup(self.memory.close)
        self.memory.initialize({"objective": "Test", "rules": [], "progress": {
            "window": 8, "max_cycle_period": 1, "cycle_repetitions": 2,
            "strategy_attempts": 2, "no_progress_attempts": 3}})
        observation = self.memory.observe("frame.png", "start", "test")
        value = DemoAgent().decide(messages({"observation_id": observation}, b"png"))[0]
        value["action"].update(kind="stop", buttons=[])
        value["working_state_update"] = {"expected_revision": 0, "location": None, "uncertainties": [], "next_check": "Walk",
            "subgoal_change": {"operation": "start", "id": "walk", "evidence_observation_ids": [observation], "reason": "Walk to exit",
                "definition": {"description": "Walk to exit", "objective_relation": "Reach gym", "completion_condition": "Exit visible", "abandonment_condition": "Blocked"}}}
        turn = self.memory.plan(observation, value)
        self.memory.status(turn, "stopped")

    def grant(self, moves=5):
        self.memory.repetition.grant(self.memory.working_context(), "move", ["UP"], moves, 300, "Visible corridor", "Doorway reached")

    def move(self, digest, button="UP", scene="overworld", fail=False):
        observation = self.memory.observe("before.png", "before", "test")
        context = {"observation_id": observation, "outcome_review": self.memory.outcomes.context()}
        value = DemoAgent().decide(messages(context, b"png"))[0]
        value["scene"] = scene
        value["action"]["buttons"] = [button]
        value["strategy"] = {"kind": "move", "target": "exit", "intended_effect": "Reach exit"}
        value["memory_updates"] = []
        turn = self.memory.plan(observation, value)
        if fail:
            self.memory.status(turn, "unknown")
            return turn
        result = self.memory.observe("after.png", digest, "test")
        self.memory.step_status(turn, 1, "completed", result)
        self.memory.status(turn, "completed")
        return turn

    def test_scoped_allowance_defers_input_repetition_but_expires_at_reserved_limit(self):
        self.grant(moves=3)
        self.move("one")
        self.move("two")
        progress = self.memory.monitor_progress()
        self.assertFalse(progress["requires_recovery"])
        self.assertTrue(progress["suppressed_signals"])
        self.move("three")
        self.assertTrue(self.memory.monitor_progress()["requires_recovery"])
        self.assertFalse(self.memory.repetition.context(self.memory.working_context())["active"])
        with contextlib.closing(Memory(self.path)) as reopened:
            self.assertEqual(reopened.repetition.context(reopened.working_context())["moves_remaining"], 0)

    def test_exact_visual_cycles_still_require_recovery(self):
        self.grant()
        self.move("same")
        self.move("same")
        self.assertTrue(self.memory.monitor_progress()["requires_recovery"])

    def test_other_buttons_and_battle_moves_do_not_get_exemptions(self):
        self.grant()
        self.move("one", button="DOWN")
        self.move("two", button="DOWN")
        self.assertTrue(self.memory.monitor_progress()["requires_recovery"])
        self.assertEqual(self.memory.connection.execute("SELECT COUNT(*) FROM repetition_uses").fetchone()[0], 0)

    def test_battle_moves_do_not_consume_or_use_the_allowance(self):
        self.grant()
        self.move("one", scene="battle")
        self.move("two", scene="battle")
        self.assertTrue(self.memory.monitor_progress()["requires_recovery"])
        self.assertEqual(self.memory.repetition.context(self.memory.working_context())["moves_remaining"], 5)

    def test_no_progress_limit_survives_an_active_allowance(self):
        self.grant()
        self.move("one")
        self.move("two")
        self.move("three")
        progress = self.memory.monitor_progress()
        self.assertTrue(progress["repetition_allowance"]["active"])
        self.assertTrue(progress["requires_recovery"])

    def test_reservation_is_not_refunded_after_interruption_or_renewable(self):
        self.grant()
        self.move("one", fail=True)
        self.assertEqual(self.memory.repetition.context(self.memory.working_context())["moves_remaining"], 4)
        with self.assertRaises(sqlite3.IntegrityError):
            self.grant()

    def test_expiry_and_state_revision_end_the_allowance(self):
        self.grant()
        state = self.memory.working_context()
        self.assertFalse(self.memory.repetition.context(dict(state, revision=state["revision"] + 1))["active"])
        with self.memory.connection:
            self.memory.connection.execute("UPDATE repetition_allowances SET expires_at=?", ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),))
        self.assertFalse(self.memory.repetition.context(state)["active"])


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        config_path = self.root / "config.json"
        write_json(config_path, {"run_dir": "run", "objective": "Test", "rules": [],
                               "emulator": {"backend": "replay"}, "model": {"name": "fixture"}})
        self.config = load_config(config_path)
        self.source = Path(self.config["run_dir"])
        self.images = create_images(self.root / "fixtures")
        run(self.config, ReplayEmulator(self.images), DemoAgent(), turns=1)

    def fingerprint(self):
        return {str(path.relative_to(self.source)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.source.rglob("*") if path.is_file()}

    def test_replay_rehydrates_exact_requests_without_loading_a_model_or_mutating_source(self):
        before = self.fingerprint()
        agent = Mock()
        report = evaluate(self.source, self.root / "replay", agent=agent)
        self.assertEqual(report["summary"]["valid"], 1)
        self.assertEqual(report["controller_inputs_sent"], 0)
        self.assertFalse(report["semantic_success_verified"])
        agent.decide.assert_not_called()
        self.assertEqual(self.fingerprint(), before)

    def test_evaluation_includes_uncheckpointed_wal_records_without_touching_source(self):
        with contextlib.closing(Memory(self.source / "memory.sqlite")) as memory:
            row = memory.previous_observation()
            png = (self.source / row["image_path"]).read_bytes()
            observation = memory.observe(row["image_path"], row["image_hash"], "fixture")
            memory.begin_request(messages({"observation_id": observation}, png), observation, {}, "fixture")
            self.assertTrue((self.source / "memory.sqlite-wal").is_file())
            before = self.fingerprint()
            report = evaluate(self.source, self.root / "wal")
            self.assertEqual(report["summary"]["valid"], 1)
            self.assertEqual(report["summary"]["unscored"], 1)
            self.assertEqual(self.fingerprint(), before)

    def test_shadow_repeats_compare_proposals_without_applying_them(self):
        before = self.fingerprint()

        class Agent:
            def __init__(self):
                self.calls = 0

            def decide(self, request):
                self.calls += 1
                value, trace = DemoAgent().decide(request)
                value["action"]["buttons"] = ["A" if self.calls % 2 else "B"]
                return value, trace

        report = evaluate(self.source, self.root / "shadow", "shadow", Agent(), repeat=2)
        self.assertEqual(report["summary"]["valid"], 2)
        self.assertEqual(report["proposal_variants"]["1"], 2)
        self.assertFalse(report["results"][0]["same_as_archived_input_plan"])
        self.assertEqual(self.fingerprint(), before)

    def test_missing_or_tampered_evidence_blocks_shadow_generation(self):
        image = next((self.source / "screenshots").glob("*.png"))
        for image in (self.source / "screenshots").glob("*.png"):
            image.write_bytes(b"tampered")
        agent = Mock()
        report = evaluate(self.source, self.root / "bad", "shadow", agent)
        self.assertEqual(report["summary"]["errors"], 1)
        agent.decide.assert_not_called()

    def test_human_criteria_are_reported_as_literal_checks_without_claiming_semantic_success(self):
        labels = self.root / "labels.json"
        write_json(labels, {"cases": [{"request_id": 1, "allowed_action_kinds": ["stop"],
                                      "forbidden_claim_fragments": ["doorway"], "note": "Synthetic regression"}]})
        report = evaluate(self.source, self.root / "labeled", labels=labels)
        self.assertEqual(report["summary"]["label_failures"], 1)
        self.assertFalse(report["semantic_success_verified"])
        self.assertEqual(report["results"][0]["labels"]["forbidden_fragment_hits"], ["doorway"])

    def test_invalid_labels_and_overlapping_output_are_rejected(self):
        labels = self.root / "labels.json"
        for value in ({"cases": [{"request_id": True}]}, {"cases": [{"request_id": 1}]}, {"cases": "bad"}):
            write_json(labels, value)
            with self.assertRaises(ValueError):
                load_labels(labels)
        with self.assertRaises(ValueError):
            evaluate(self.source, self.source / "evaluation")
        with self.assertRaises(ValueError):
            evaluate(self.source, self.root / "replay", repeat=2)

    def test_cli_replay_never_constructs_an_emulator(self):
        with patch("pokemon_harness.cli.emulator_for", side_effect=AssertionError("no emulator")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["evaluate", str(self.source), "--output-dir", str(self.root / "cli-evaluation")]), 0)

    def test_archived_outcome_and_recovery_gates_reject_bypasses(self):
        decision = DemoAgent().decide(messages({"observation_id": 2}, b"png"))[0]
        context = {"observation_id": 2, "outcome_review": {"pending_count": 1, "next_plan": {"turn_id": 1}}}
        with self.assertRaisesRegex(ValueError, "pending"):
            validate_context_gates(decision, context)
        context = {"progress": {"requires_recovery": True, "episode_id": 1,
                               "recovery_attempts_remaining": 1, "recovery_signatures": [], "last_input_signature": None}}
        with self.assertRaisesRegex(ValueError, "recovery"):
            validate_context_gates(decision, context)
        decision["recovery_plan"] = {"episode_id": 1, "explanation": "Try another input"}
        validate_context_gates(decision, context)
        context["progress"]["last_input_signature"] = [["press", ["UP"]]]
        with self.assertRaisesRegex(ValueError, "forbidden"):
            validate_context_gates(decision, context)

    def test_shadow_generation_expiry_preserves_partial_report_and_stops_repeats(self):
        output = self.root / "timeout"
        before = self.fingerprint()
        with patch("pokemon_harness.evaluation.bounded_decision", side_effect=TimingLimit("expired")) as generate:
            with self.assertRaises(TimingLimit):
                evaluate(self.source, output, "shadow", DemoAgent(), repeat=3)
        self.assertEqual(generate.call_count, 1)
        report = json.loads((output / "evaluation-report.json").read_text())
        self.assertFalse(report["completed"])
        self.assertEqual(report["termination"], "generation_time_budget_exhausted")
        self.assertEqual(len(report["results"]), 1)
        self.assertEqual(self.fingerprint(), before)


class TimingIntegrationTests(unittest.TestCase):
    def test_expiry_before_queued_input_cancels_it_without_marking_it_attempted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "config.json"
            write_json(path, {"run_dir": "run", "objective": "Test", "rules": [],
                "sequence_delay_ms": 0, "emulator": {"backend": "replay"}, "model": {"name": "fixture"}})
            config = load_config(path)
            emulator = ReplayEmulator(create_images(root / "fixtures"))

            class Agent(DemoAgent):
                def decide(self, request):
                    decision, trace = super().decide(request)
                    decision["additional_actions"] = [dict(decision["action"])]
                    return decision, trace

            def fresh(age, scene):
                if emulator.actions:
                    raise TimingLimit("Queued decision expired")

            with patch("pokemon_harness.control.timing.RuntimeBudget.fresh", side_effect=fresh):
                with self.assertRaises(TimingLimit):
                    run(config, emulator, Agent(), turns=1)
            self.assertEqual(len(emulator.actions), 1)
            with contextlib.closing(Memory(Path(config["run_dir"]) / "memory.sqlite")) as memory:
                self.assertEqual([step["status"] for step in memory.recent_turns()[0]["steps"]], ["completed", "cancelled"])

    def test_session_expiry_during_generation_discards_late_response(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "config.json"
            write_json(path, {"run_dir": "run", "objective": "Test", "rules": [],
                "emulator": {"backend": "replay"}, "model": {"name": "fixture"},
                "timing_budgets": {"session_seconds": 1}})
            config = load_config(path)
            started, release, finished = threading.Event(), threading.Event(), threading.Event()
            self.addCleanup(release.set)

            class Agent:
                def decide(self, request):
                    started.set()
                    release.wait(2)
                    result = DemoAgent().decide(request)
                    finished.set()
                    return result

            emulator = ReplayEmulator(create_images(root / "fixtures"))
            clock = lambda: 2 if started.is_set() else 0
            with patch("pokemon_harness.runner.RuntimeBudget", side_effect=lambda memory, settings: RuntimeBudget(memory, settings, clock)):
                with self.assertRaises(TimingLimit):
                    run(config, emulator, Agent(), turns=1)
            database = Path(config["run_dir"]) / "memory.sqlite"
            with contextlib.closing(Memory(database)) as memory:
                before = memory.conversation.history()
                self.assertEqual(memory.conversation.request(1)["status"], "failed")
            release.set()
            self.assertTrue(finished.wait(1))
            with contextlib.closing(Memory(database)) as memory:
                self.assertEqual(memory.conversation.history(), before)
                self.assertEqual(memory.recent_turns(), [])
            self.assertEqual(emulator.actions, [])

    def test_stale_decision_cannot_commit_or_send_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "config.json"
            write_json(path, {"run_dir": "run", "objective": "Test", "rules": [],
                              "emulator": {"backend": "replay"}, "model": {"name": "fixture"}})
            config = load_config(path)
            emulator = ReplayEmulator(create_images(root / "fixtures"))
            with patch("pokemon_harness.runner.capture_age", return_value=10000):
                with self.assertRaises(TimingLimit):
                    run(config, emulator, DemoAgent(), turns=1)
            self.assertEqual(emulator.actions, [])
            with contextlib.closing(Memory(Path(config["run_dir"]) / "memory.sqlite")) as memory:
                self.assertEqual(memory.recent_turns(), [])
                self.assertEqual(memory.conversation.request(1)["status"], "failed")
                self.assertEqual(memory.connection.execute("SELECT closed FROM runtime_sessions").fetchone()[0], 1)
