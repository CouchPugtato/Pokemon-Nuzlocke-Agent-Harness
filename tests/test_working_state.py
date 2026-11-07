import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import ValidationError

from pokemon_harness.agent.prompts import messages
from pokemon_harness.agent.schemas import validate_decision
from pokemon_harness.cli import main
from pokemon_harness.config import load_config
from pokemon_harness.demo import DemoAgent, create_images
from pokemon_harness.emulator.interface import ReplayEmulator
from pokemon_harness.memory.database import Memory
from pokemon_harness.runner import run
from pokemon_harness.telemetry.backup import create_backup, verify_backup
from pokemon_harness.telemetry.recorder import write_json
from pokemon_harness.telemetry.restore import restore_backup


def decision(observation_id):
    return DemoAgent().decide(messages({"observation_id": observation_id}, b"png"))[0]


def change(operation="start", identity="inspect:party", evidence=None):
    return {"operation": operation, "id": identity,
            "definition": {"description": "Inspect party", "objective_relation": "Check gym readiness",
                           "completion_condition": "Party menu visible", "abandonment_condition": "Unsafe battle"}
            if operation in {"start", "revise"} else None,
            "reason": "Inspect available party before continuing" if operation != "complete" else "Party menu visible",
            "evidence_observation_ids": evidence or []}


def update(revision=0, subgoal=None):
    return {"expected_revision": revision, "location": None, "uncertainties": [],
            "next_check": "Inspect party menu", "subgoal_change": subgoal}


class WorkingStateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.memory = Memory(self.root / "memory.sqlite")
        self.addCleanup(self.memory.close)
        self.memory.initialize({"objective": "Beat first gym", "rules": ["No items"]})

    def plan(self, patch):
        observation = self.memory.observe("unused.png", "hash", "test")
        value = decision(observation)
        value["working_state_update"] = patch
        return self.memory.plan(observation, value)

    def test_additive_schema_starts_empty_and_noop_does_not_append(self):
        self.assertEqual(self.memory.working.current()["revision"], 0)
        self.plan(None)
        self.assertEqual(self.memory.working.history(), [])
        self.assertEqual(self.memory.export()["working_state"]["subgoal"], None)

    def test_lifecycle_is_persistent_and_history_is_not_mutated(self):
        self.plan(update(subgoal=change()))
        first = self.memory.working.current()
        revised = change("revise")
        revised["definition"]["description"] = "Inspect starter summary"
        self.plan(update(1, revised))
        self.plan(update(2, change("complete", evidence=[3])))
        self.assertEqual(self.memory.working.current()["subgoal"]["status"], "completed")
        self.assertEqual(self.memory.working.history()[-1], first)
        self.plan(update(3, change(identity="travel:rustboro")))
        self.plan(update(4, change("abandon", "travel:rustboro")))
        self.assertEqual(self.memory.working.current()["subgoal"]["status"], "abandoned")
        self.assertEqual(len(self.memory.working.history()), 5)
        self.assertEqual(self.memory.working.history(2, 2)[0]["revision"], 3)
        with contextlib.closing(Memory(self.root / "memory.sqlite")) as reopened:
            self.assertEqual(reopened.working.current(), self.memory.working.current())

    def test_stale_revision_rolls_back_facts_events_turn_and_steps(self):
        self.plan(update(subgoal=change()))
        observation = self.memory.observe("unused.png", "hash", "test")
        value = decision(observation)
        value["working_state_update"] = update(0)
        value["events"] = [{"key": "test", "type": "progress", "subject": "test", "area": "test",
                            "details": "test", "confidence": "high", "corrects_event_key": ""}]
        with self.assertRaisesRegex(ValueError, "stale revision"):
            self.memory.plan(observation, value)
        self.assertEqual(len(self.memory.facts(True)), 1)
        self.assertEqual(self.memory.events(), [])
        self.assertEqual(len(self.memory.recent_turns()), 1)
        self.assertEqual(self.memory.connection.execute("SELECT COUNT(*) FROM action_steps").fetchone()[0], 1)
        self.assertEqual(self.memory.working.current()["revision"], 1)

    def test_active_subgoal_cannot_be_silently_replaced_or_renamed(self):
        self.plan(update(subgoal=change()))
        for operation in ("start", "revise", "complete", "abandon"):
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                self.plan(update(1, change(operation, "another")))
        self.assertEqual(self.memory.working.current()["revision"], 1)

    def test_retired_subgoal_cannot_be_reused_or_reactivated(self):
        self.plan(update(subgoal=change()))
        self.plan(update(1, change("abandon")))
        for operation in ("start", "revise", "complete", "abandon"):
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                self.plan(update(2, change(operation)))

    def test_completion_requires_current_observation_not_predictions(self):
        self.plan(update(subgoal=change()))
        for evidence in ([], [1]):
            with self.subTest(evidence=evidence), self.assertRaisesRegex(ValueError, "current visual evidence"):
                self.plan(update(1, change("complete", evidence=evidence)))
        self.assertEqual(self.memory.working.current()["subgoal"]["status"], "active")

    def test_evidence_rejects_missing_future_and_stale_observed_claims(self):
        self.plan(None)
        for basis, ids in (("inferred", []), ("observed", [1]), ("inferred", [999])):
            patch = update()
            patch["location"] = {"claim": "Route 101", "basis": basis, "evidence_observation_ids": ids}
            with self.subTest(basis=basis, ids=ids), self.assertRaises(ValueError):
                self.plan(patch)
        patch = update(subgoal=change(evidence=[999]))
        with self.assertRaisesRegex(ValueError, "unavailable observation"):
            self.plan(patch)

    def test_location_and_uncertainty_keep_original_provenance(self):
        self.plan(None)
        patch = update()
        patch["location"] = {"claim": "Route 101", "basis": "inferred", "evidence_observation_ids": [1]}
        patch["uncertainties"] = [{"claim": "Party unknown", "basis": "unknown", "evidence_observation_ids": []}]
        self.plan(patch)
        current = self.memory.working.current()
        self.assertEqual(current["location"]["evidence_observation_ids"], [1])
        self.assertEqual(current["evidence_observation_id"], 2)
        self.plan(update(1))
        self.assertIsNone(self.memory.working.current()["location"])
        self.assertEqual(self.memory.working.current()["uncertainties"], [])

    def test_schema_is_bounded_and_cannot_edit_fixed_objective(self):
        for key, value in (("objective", "Win league"), ("uncertainties", [None] * 9)):
            patch = update()
            patch[key] = value
            with self.subTest(key=key), self.assertRaises(ValidationError):
                self.plan(patch)
        malformed = change("complete")
        malformed["definition"] = change()["definition"]
        with self.assertRaisesRegex(ValueError, "Only start and revise"):
            self.plan(update(subgoal=malformed))

    def test_search_is_readonly_in_contract_and_journal(self):
        value = decision(1)
        value.update(memory_updates=[], events=[], working_state_update=update(),
                     memory_search={"query": "", "scope": "all", "category": "", "location": "",
                                    "include_history": False, "offset": 0, "limit": 1})
        value["action"] = {"kind": "search_memory", "buttons": [], "hold_frames": 1, "settle_frames": 0}
        with self.assertRaisesRegex(ValueError, "read-only"):
            validate_decision(value, 1)
        self.memory.observe("unused.png", "hash", "test")
        with self.assertRaisesRegex(ValueError, "cannot change"):
            self.memory.plan(1, value)
        self.assertEqual(self.memory.recent_turns(), [])

    def test_recovery_preserves_state_without_claiming_input_succeeded(self):
        turn = self.plan(update(subgoal=change()))
        before = self.memory.working.current()
        self.memory.status(turn, "executing")
        self.memory.recover()
        self.assertEqual(self.memory.working.current(), before)
        self.assertEqual(self.memory.recent_turns()[0]["status"], "unknown")
        self.assertEqual(before["subgoal"]["status"], "active")

    def test_runner_resume_backup_restore_and_inspection_keep_working_state(self):
        path = self.root / "config.json"
        write_json(path, {"run_dir": "run", "objective": "Test", "model": {"name": "test"},
                          "emulator": {"backend": "replay"}})
        config = load_config(path)
        images = create_images(self.root / "fixtures")
        contexts = []

        class Agent:
            def decide(self, request):
                context = json.loads(request[1]["content"][0]["text"])
                contexts.append(context)
                value = DemoAgent().decide(request)[0]
                if context["working_state"]["revision"] == 0:
                    value["working_state_update"] = update(subgoal=change())
                return value, {"decision": value}

        run(config, ReplayEmulator(images), Agent(), turns=1)
        run(config, ReplayEmulator(images), Agent(), turns=1)
        self.assertEqual(contexts[1]["working_state"]["revision"], 1)
        self.assertEqual(contexts[1]["working_state"]["subgoal"]["id"], "inspect:party")
        source = Path(config["run_dir"])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["state", str(source), "--history", "--limit", "1"]), 0)
        inspected = json.loads(output.getvalue())
        self.assertEqual(inspected["history"][0], contexts[1]["working_state"])
        backup = create_backup(source, self.root / "backups")
        self.assertTrue(verify_backup(backup)["verified"])
        target = self.root / "restored"
        restore_backup(backup, target)
        with contextlib.closing(Memory(target / "memory.sqlite")) as restored:
            self.assertEqual(restored.working.current(), contexts[1]["working_state"])
            self.assertTrue(restored.escalations.review(restored.events())["blocked_cases"])
