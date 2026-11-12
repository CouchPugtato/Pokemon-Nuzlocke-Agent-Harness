import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jsonschema import Draft202012Validator, ValidationError

from pokemon_harness.agent.prompts import messages
from pokemon_harness.agent.reassessment import PERCEPTION_FIELDS, schema, settings
from pokemon_harness.agent.schemas import request_schema, schema_for_context
from pokemon_harness.config import load_config
from pokemon_harness.demo import DemoAgent, create_images
from pokemon_harness.emulator.interface import ReplayEmulator
from pokemon_harness.memory.database import Memory
from pokemon_harness.memory.conversation import compaction_settings
from pokemon_harness.memory.recall import automatic_recall, recall_settings
from pokemon_harness.memory.retrieval import retrieve
from pokemon_harness.memory.scenes import goal_token
from pokemon_harness.runner import run
from pokemon_harness.telemetry.recorder import write_json


def report(context):
    value = {"observation_id": context["observation_id"], "visible_objects": ["Player among storage boxes"],
        "visible_text": [], "possible_exits": [{"description": "Bright opening on the right", "confidence": "low"}],
        "scene": "overworld", "subgoal_assessment": "contradicted" if context.get("subgoal") else "none",
        "subgoal_explanation": "No professor or selection menu is visible", "stale_scene_fact_ids": [],
        "next_check": "Inspect the open floor toward the bright boundary"}
    value.update(context.get("perception", {}))
    return value


def perception(context):
    value = report(context)
    return {key: value[key] for key in PERCEPTION_FIELDS}


class SceneReassessmentTests(unittest.TestCase):
    def test_observer_has_no_old_goal_and_reconciler_cannot_rewrite_visual_evidence(self):
        context = {"phase": "scene_observation", "observation_id": 1}
        observation = perception(context)
        request = messages(context, b"png")
        self.assertNotIn("subgoal", json.loads(request[1]["content"][0]["text"]))
        validator = Draft202012Validator(schema(context))
        validator.validate(observation)
        invalid = dict(observation, subgoal_assessment="supported")
        self.assertFalse(validator.is_valid(invalid))
        context = self.context()
        context["perception"] = observation
        value = report(context)
        validator = Draft202012Validator(schema(context))
        validator.validate(value)
        for key, replacement in (("visible_objects", ["Professor in a lab"]), ("visible_text", ["Choose a starter"]),
                                 ("possible_exits", []), ("scene", "dialogue")):
            invalid = dict(value, **{key: replacement})
            self.assertFalse(validator.is_valid(invalid))
        context["perception"]["observation_id"] = 99
        with self.assertRaisesRegex(ValueError, "stale observation"):
            schema(context)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "memory.sqlite"
        self.memory = Memory(self.path)
        self.addCleanup(self.memory.close)
        self.memory.initialize({"objective": "Test", "rules": []})
        self.observation = self.memory.observe("before.png", "before", "test")
        value = DemoAgent().decide(messages({"observation_id": self.observation}, b"png"))[0]
        value["memory_updates"] = [{"category": "working", "key": "scene:current", "claim": "Professor is still speaking",
            "location": "Lab", "basis": "observed", "confidence": "high", "status": "active"}]
        value["working_state_update"] = {"expected_revision": 0, "location": None, "uncertainties": [],
            "next_check": "Wait for starter selection", "subgoal_change": {"operation": "start", "id": "starter",
            "definition": {"description": "Choose a starter", "objective_relation": "Prepare for gym",
                "completion_condition": "Starter in party", "abandonment_condition": "Run ends"},
            "reason": "Expected dialogue trigger", "evidence_observation_ids": [self.observation]}}
        self.memory.status(self.memory.plan(self.observation, value), "stopped")

    def context(self):
        state = self.memory.working.current()
        return {"phase": "scene_reassessment", "observation_id": self.observation,
            "subgoal": state["subgoal"], "working_revision": state["revision"], "goal_token": goal_token(state),
            "scene_fact_candidates": self.memory.scenes.candidates(self.memory.facts())}

    def save(self):
        context = self.context()
        value = report(context)
        value["stale_scene_fact_ids"] = [1]
        with self.memory.connection:
            self.memory.scenes.save(context, value, self.memory.working.current(), self.memory.monitor_progress())
        return value

    def test_report_schema_binds_observation_and_cannot_contain_controller_input(self):
        context = self.context()
        self.assertEqual(request_schema(messages(context, b"png")), schema(context))
        validator = Draft202012Validator(schema(context))
        value = report(context)
        validator.validate(value)
        value["action"] = {"kind": "press", "buttons": ["A"]}
        self.assertFalse(validator.is_valid(value))
        del value["action"]
        value["observation_id"] += 1
        self.assertFalse(validator.is_valid(value))

    def test_grounded_probe_keeps_recovery_limits_and_forbids_history_search_or_writes(self):
        context = {"observation_id": 1, "grounded_scene_action": True,
            "outcome_review": {"pending_count": 0}, "progress": {"requires_recovery": True,
                "episode_id": 3, "recovery_attempts_remaining": 1, "recovery_signatures": [],
                "last_input_signature": [["wait", []]]}}
        value = DemoAgent().decide(messages(context, b"png"))[0]
        value.update(memory_updates=[], recovery_plan={"episode_id": 3, "explanation": "Inspect a different visible path"})
        validator = Draft202012Validator(schema_for_context(context))
        validator.validate(value)
        from pokemon_harness.evaluation import validate_context_gates

        validate_context_gates(value, context)
        invalid = copy.deepcopy(value)
        invalid["recovery_plan"]["episode_id"] = 2
        self.assertFalse(validator.is_valid(invalid))
        invalid = copy.deepcopy(value)
        invalid["action"]["kind"] = "search_memory"
        self.assertFalse(validator.is_valid(invalid))
        invalid = copy.deepcopy(value)
        invalid["action"]["buttons"] = ["A"]
        self.assertFalse(validator.is_valid(invalid))
        value["additional_actions"] = [dict(value["action"])]
        self.assertFalse(validator.is_valid(value))
        value["additional_actions"] = []
        value["memory_updates"] = DemoAgent().decide(messages(context, b"png"))[0]["memory_updates"]
        self.assertFalse(validator.is_valid(value))
        with self.assertRaises(ValidationError):
            validate_context_gates(value, context)

    def test_stale_memory_and_goal_are_withheld_without_changing_source_records(self):
        original = self.memory.working.current()
        progress = self.memory.monitor_progress()
        self.save()
        self.assertEqual(self.memory.working.current(), original)
        self.assertIsNone(self.memory.working_context()["subgoal"]["definition"])
        self.assertEqual(self.memory.facts()[0]["scene_validity"], "stale")
        self.assertEqual(retrieve(self.memory.facts(), "Professor", self.observation), [])
        recalled, _ = automatic_recall(self.memory.facts(), [], self.memory.working_context(), [], [],
            "Test", self.observation, 24, recall_settings({}))
        self.assertEqual(recalled, [])
        self.assertEqual(self.memory.monitor_progress(), progress)
        self.assertEqual(self.memory.events(), [])
        self.memory.save_checkpoint(compaction=compaction_settings({}))
        with contextlib.closing(Memory(self.path)) as reopened:
            self.assertEqual(reopened.scenes.latest()["payload"]["scene"], "overworld")
            self.assertEqual(reopened.facts(True)[0]["claim"], "Professor is still speaking")
            self.assertIsNotNone(reopened.conversation.context()["checkpoint"])

    def test_controller_recall_keeps_perception_without_repeating_the_rejected_story(self):
        self.save()
        value = self.memory.scenes.controller_context(self.observation)
        self.assertEqual(value["visible_objects"], ["Player among storage boxes"])
        self.assertNotIn("subgoal_explanation", value)
        self.assertNotIn("next_check", value)
        self.assertIn("subgoal_explanation", self.memory.scenes.latest()["payload"])

    def test_new_working_revision_does_not_restore_unchanged_stale_goal(self):
        self.save()
        self.observation = self.memory.observe("next.png", "next", "test")
        value = DemoAgent().decide(messages({"observation_id": self.observation}, b"png"))[0]
        value["working_state_update"] = {"expected_revision": 1, "location": None,
            "uncertainties": [], "next_check": "Another check", "subgoal_change": None}
        self.memory.status(self.memory.plan(self.observation, value), "stopped")
        self.assertIsNone(self.memory.working_context()["subgoal"]["definition"])

    def test_later_uncertain_report_does_not_restore_a_contradicted_goal(self):
        self.save()
        self.observation = self.memory.observe("later.png", "later", "test")
        context = self.context()
        value = report(context)
        value["subgoal_assessment"] = "uncertain"
        with self.memory.connection:
            self.memory.scenes.save(context, value, self.memory.working.current(), {})
        self.assertIsNone(self.memory.working_context()["subgoal"]["definition"])

    def test_uncertain_current_scene_withholds_old_scene_claims_without_declaring_them_false(self):
        self.observation = self.memory.observe("later.png", "later", "test")
        context = self.context()
        value = report(context)
        value["subgoal_assessment"] = "uncertain"
        with self.memory.connection:
            self.memory.scenes.save(context, value, self.memory.working.current(), {})
        self.assertEqual(self.memory.facts()[0]["scene_validity"], "needs_refresh")
        self.assertEqual(self.memory.working_context()["subgoal"]["scene_validity"], "needs_refresh")
        self.assertEqual(self.memory.scenes.latest()["payload"]["stale_scene_fact_ids"], [])

    def test_report_cannot_mark_unlisted_fact_stale_or_use_changed_goal(self):
        context = self.context()
        value = report(context)
        value["stale_scene_fact_ids"] = [99]
        with self.assertRaises(ValidationError), self.memory.connection:
            self.memory.scenes.save(context, value, self.memory.working.current(), {})
        value["stale_scene_fact_ids"] = []
        state = copy.deepcopy(self.memory.working.current())
        state["subgoal"]["definition"]["description"] = "Different goal"
        with self.assertRaisesRegex(ValueError, "goal changed"), self.memory.connection:
            self.memory.scenes.save(context, value, state, {})
        self.assertIsNone(self.memory.scenes.latest())

    def test_policy_and_trigger_intervals_are_bounded(self):
        policy = settings({"scene_reassessment": {"enabled": True}})
        self.assertEqual(self.memory.scenes.reason(self.observation, {}, policy), "initial_scene_check")
        self.save()
        self.assertIsNone(self.memory.scenes.reason(self.observation, {"no_progress_attempts": 8}, policy))
        self.assertEqual(self.memory.scenes.reason(self.observation,
            {"requires_recovery": True, "episode_id": 1}, policy), "new_recovery_episode")
        self.memory.note("session_start", {})
        self.assertEqual(self.memory.scenes.reason(self.observation, {}, policy), "resumed_scene_check")
        for value in ({"enabled": 1}, {"min_turn_interval": 20, "max_turn_interval": 3}, {"unknown": 1}):
            with self.assertRaises(ValueError):
                settings({"scene_reassessment": value})

    def test_explicit_subgoal_revision_replaces_the_stale_definition(self):
        self.save()
        self.observation = self.memory.observe("next.png", "next", "test")
        value = DemoAgent().decide(messages({"observation_id": self.observation}, b"png"))[0]
        definition = dict(self.memory.working.current()["subgoal"]["definition"], description="Inspect the visible room")
        value["working_state_update"] = {"expected_revision": 1, "location": None, "uncertainties": [],
            "next_check": "Inspect the floor", "subgoal_change": {"operation": "revise", "id": "starter",
            "definition": definition, "reason": "Current pixels contradict the old scene",
            "evidence_observation_ids": [self.observation]}}
        self.memory.status(self.memory.plan(self.observation, value), "stopped")
        self.assertEqual(self.memory.working_context()["subgoal"]["definition"]["description"], "Inspect the visible room")

    def test_runner_reassesses_before_action_and_can_stop_without_input(self):
        path = self.root / "config.json"
        write_json(path, {"run_dir": "replay", "objective": "Test", "rules": [], "model": {"name": "fixture"},
            "emulator": {"backend": "replay"}, "scene_reassessment": {"enabled": True}})
        config = load_config(path)
        phases = []
        observer_contexts = []

        class Agent:
            def decide(self, request):
                context = json.loads(request[1]["content"][0]["text"])
                phases.append(context.get("phase", "action"))
                if context.get("phase") == "scene_observation":
                    observer_contexts.append(context)
                    return perception(context), {"backend": "fixture"}
                if context.get("phase") == "scene_reassessment":
                    return report(context), {"backend": "fixture"}
                value, trace = DemoAgent().decide(request)
                value.update(memory_updates=[], strategy=None)
                value["action"].update(kind="stop", buttons=[], hold_frames=1, settle_frames=0)
                return value, trace

        emulator = ReplayEmulator(create_images(self.root / "images"))
        with contextlib.redirect_stdout(io.StringIO()):
            result = run(config, emulator, Agent(), turns=1)
        self.assertEqual(phases, ["scene_observation", "scene_reassessment", "action"])
        for key in ("subgoal", "scene_fact_candidates", "objective", "rules", "goal_token"):
            self.assertNotIn(key, observer_contexts[0])
        self.assertEqual(emulator.actions, [])
        self.assertEqual(result["reason"], "agent_stop")

    def test_failed_reassessment_stops_before_any_controller_input(self):
        path = self.root / "config.json"
        write_json(path, {"run_dir": "failed-replay", "objective": "Test", "rules": [], "model": {"name": "fixture"},
            "emulator": {"backend": "replay"}, "scene_reassessment": {"enabled": True}})

        class Agent:
            def decide(self, request):
                raise ValueError("Perception failed")

        emulator = ReplayEmulator(create_images(self.root / "images"))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Perception failed"):
            run(load_config(path), emulator, Agent(), turns=1)
        self.assertEqual(emulator.actions, [])

    def test_failed_goal_check_does_not_commit_the_successful_visual_pass(self):
        path = self.root / "config.json"
        write_json(path, {"run_dir": "failed-goal-check", "objective": "Test", "rules": [], "model": {"name": "fixture"},
            "emulator": {"backend": "replay"}, "scene_reassessment": {"enabled": True}})
        config = load_config(path)
        phases = []

        class Agent:
            def decide(self, request):
                context = json.loads(request[1]["content"][0]["text"])
                phases.append(context["phase"])
                if context["phase"] == "scene_observation":
                    return perception(context), {"backend": "fixture"}
                raise ValueError("Goal check failed")

        emulator = ReplayEmulator(create_images(self.root / "images"))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Goal check failed"):
            run(config, emulator, Agent(), turns=1)
        self.assertEqual(phases, ["scene_observation", "scene_reassessment"])
        self.assertEqual(emulator.actions, [])
        with contextlib.closing(Memory(Path(config["run_dir"]) / "memory.sqlite")) as memory:
            self.assertIsNone(memory.scenes.latest())
            self.assertEqual(memory.connection.execute("SELECT COUNT(*) FROM stale_scene_facts").fetchone()[0], 0)

    def test_scene_change_and_periodic_checks_respect_turn_intervals(self):
        from test_capture_settling import png

        self.observation = self.memory.observe("scene.png", "black", "test", png("black"))
        self.save()
        current = self.memory.observe("changed.png", "white", "test", png("white"))
        policy = settings({})
        with patch.object(self.memory.scenes, "watermark", return_value=3):
            self.assertEqual(self.memory.scenes.reason(current, {}, policy), "changed_scene")
            self.assertEqual(self.memory.scenes.reason(current, {"no_progress_attempts": 2}, policy), "ineffective_inputs")
        with patch.object(self.memory.scenes, "watermark", return_value=12):
            self.assertEqual(self.memory.scenes.reason(self.observation, {}, policy), "periodic_scene_check")

    def test_reassessment_preserves_pending_outcome_gate(self):
        path = self.root / "config.json"
        write_json(path, {"run_dir": "review-replay", "objective": "Test", "rules": [], "model": {"name": "fixture"},
            "emulator": {"backend": "replay"}, "scene_reassessment": {"enabled": True}})
        phases = []

        class Agent:
            def decide(self, request):
                context = json.loads(request[1]["content"][0]["text"])
                if context.get("phase") == "scene_observation":
                    phases.append("perception")
                    return perception(context), {"backend": "fixture"}
                if context.get("phase") == "scene_reassessment":
                    phases.append("reassessment")
                    return report(context), {"backend": "fixture"}
                value, trace = DemoAgent().decide(request)
                if context.get("grounded_scene_action"):
                    value["memory_updates"] = []
                if context["outcome_review"]["pending_count"]:
                    phases.append("review")
                    value.update(memory_updates=[], events=[], strategy=None)
                    value["action"].update(kind="review_outcomes", buttons=[], hold_frames=1, settle_frames=0)
                else:
                    phases.append("action")
                    if phases.count("action") == 2:
                        value["action"].update(kind="stop", buttons=[])
                return value, trace

        emulator = ReplayEmulator(create_images(self.root / "images"))
        with contextlib.redirect_stdout(io.StringIO()):
            run(load_config(path), emulator, Agent(), turns=2)
        self.assertEqual(phases, ["perception", "reassessment", "action", "review", "action"])
        self.assertEqual(len(emulator.actions), 1)
