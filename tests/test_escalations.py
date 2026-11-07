import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pokemon_harness.agent.prompts import messages
from pokemon_harness.cli import main
from pokemon_harness.config import load_config
from pokemon_harness.demo import DemoAgent, create_images
from pokemon_harness.emulator.interface import ReplayEmulator
from pokemon_harness.memory.database import Memory
from pokemon_harness.runner import run
from pokemon_harness.telemetry.recorder import RunLock, write_json


def uncertain_event():
    return {"key": "encounter:route101", "type": "encounter", "subject": "Unknown Pokemon",
            "area": "Route 101", "details": "Eligibility uncertain", "confidence": "low",
            "corrects_event_key": ""}


class EscalationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        path = self.root / "config.json"
        write_json(path, {"run_dir": "run", "objective": "Test", "rules": ["Test rule"],
                          "model": {"name": "test"}, "emulator": {"backend": "replay"}})
        self.config = load_config(path)
        self.directory = Path(self.config["run_dir"])
        self.directory.mkdir()
        write_json(self.directory / "config.json", self.config)
        self.images = create_images(self.root / "fixtures")
        (self.directory / "evidence.png").write_bytes(self.images[0].read_bytes())
        self.memory = Memory(self.directory / "memory.sqlite")
        self.addCleanup(self.memory.close)
        self.memory.initialize(self.config)
        self.observation_id = self.memory.observe("evidence.png", "hash", "replay")

    def seed_uncertainty(self):
        value, trace = DemoAgent().decide(messages({"observation_id": self.observation_id}, b"png"))
        value["events"] = [uncertain_event()]
        turn = self.memory.plan(self.observation_id, value)
        self.memory.status(turn, "blocked")
        return self.memory.escalations.review(self.memory.events())["blocked_cases"][0]["id"]

    def record_correction(self):
        observation = self.memory.observe("evidence.png", "hash", "recorded review")
        value, trace = DemoAgent().decide(messages({"observation_id": observation}, b"png"))
        value["action"] = dict(value["action"], kind="stop", buttons=[])
        value["events"] = [dict(uncertain_event(), key="correction:encounter", type="correction",
                                confidence="high", corrects_event_key="encounter:route101")]
        turn = self.memory.plan(observation, value)
        self.memory.status(turn, "stopped")
        return observation

    def test_case_is_stable_and_persists_after_reopen(self):
        case_id = self.seed_uncertainty()
        self.memory.escalations.review(self.memory.events())
        with contextlib.closing(Memory(self.directory / "memory.sqlite")) as reopened:
            cases = reopened.escalations.cases()
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0]["id"], case_id)
            self.assertEqual(cases[0]["evidence_ids"], [self.observation_id])
            self.assertEqual(cases[0]["evidence"][0]["image_path"], "evidence.png")
            self.assertTrue(cases[0]["blocking"])

    def test_restart_cannot_capture_or_generate_around_open_case(self):
        self.seed_uncertainty()
        emulator, agent = Mock(), Mock()
        result = run(self.config, emulator, agent, turns=1)
        self.assertEqual(result["reason"], "review_required")
        emulator.capture.assert_not_called()
        emulator.execute.assert_not_called()
        agent.decide.assert_not_called()
        emulator.close.assert_called_once()

    def test_cli_resume_blocks_before_loading_model_or_emulator(self):
        self.seed_uncertainty()
        with patch("pokemon_harness.cli.LMStudio") as model, patch("pokemon_harness.cli.emulator_for") as emulator:
            self.assertEqual(main(["resume", str(self.directory)]), 1)
        model.assert_not_called()
        emulator.assert_not_called()

    def test_user_assertion_needs_explicit_continuation_and_never_rewrites_ledger(self):
        case_id = self.seed_uncertainty()
        before = self.memory.events()
        self.memory.escalations.resolve(case_id, "user-assertion", "User believes this encounter is eligible")
        self.assertTrue(self.memory.escalations.review(self.memory.events())["blocked_cases"])
        self.memory.escalations.resolve(case_id, "user-assertion", "Continue despite missing historical evidence",
                                        allow_continue=True)
        review = self.memory.escalations.review(self.memory.events())
        self.assertEqual(review["blocked_cases"], [])
        self.assertEqual(review["validity"], "unverifiable")
        self.assertEqual(len(self.memory.escalations.cases()[0]["resolutions"]), 2)
        self.assertEqual(self.memory.events(), before)
        result = run(self.config, ReplayEmulator(self.images), DemoAgent(), turns=1)
        self.assertEqual(result["completed_actions"], 1)

    def test_observation_id_alone_cannot_resolve_semantic_uncertainty(self):
        case_id = self.seed_uncertainty()
        with self.assertRaisesRegex(ValueError, "ledger correction"):
            self.memory.escalations.resolve(case_id, "evidence", "This screenshot looks right", [self.observation_id])
        with self.assertRaisesRegex(ValueError, "does not belong"):
            self.memory.escalations.resolve(case_id, "evidence", "Missing evidence", [99999])
        self.assertEqual(self.memory.escalations.cases()[0]["resolutions"], [])

    def test_active_ledger_correction_resolves_case_without_deleting_history(self):
        self.seed_uncertainty()
        self.record_correction()
        review = self.memory.escalations.review(self.memory.events())
        self.assertEqual(review["blocked_cases"], [])
        self.assertEqual(review["validity"], "supported")
        self.assertEqual(self.memory.escalations.cases()[0]["resolutions"][0]["outcome"], "evidence")
        self.assertEqual(len(self.memory.events()), 2)

    def test_correction_requires_the_correct_source_observation(self):
        case_id = self.seed_uncertainty()
        self.record_correction()
        with self.assertRaisesRegex(ValueError, "ledger correction"):
            self.memory.escalations.resolve(case_id, "evidence", "Wrong source", [self.observation_id],
                                            "correction:encounter")

    def test_invalid_status_cannot_be_reset_by_later_resolution(self):
        case_id = self.seed_uncertainty()
        self.memory.escalations.resolve(case_id, "invalid", "Rule violation confirmed")
        with self.assertRaisesRegex(ValueError, "cannot be authorized"):
            self.memory.escalations.resolve(case_id, "invalid", "Keep going", allow_continue=True)
        self.memory.escalations.resolve(case_id, "user-assertion", "Change of opinion", allow_continue=True)
        self.assertEqual(self.memory.escalations.validity(), "invalid")
        result = run(self.config, Mock(), Mock(), turns=1)
        self.assertEqual(result["reason"], "review_required")
        self.assertEqual(result["validity"], "invalid")

    def test_later_evidence_does_not_erase_prior_unverifiable_continuation(self):
        case_id = self.seed_uncertainty()
        self.memory.escalations.resolve(case_id, "unverifiable", "Historical evidence missing", allow_continue=True)
        observation = self.record_correction()
        self.memory.escalations.resolve(case_id, "evidence", "New recorded correction", [observation],
                                        "correction:encounter")
        self.assertEqual(self.memory.escalations.validity(), "unverifiable")

    def test_manual_case_and_resolution_commands_leave_an_audit_trail(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["case-open", str(self.directory), "--reason", "Save history uncertain",
                                   "--observation-id", str(self.observation_id)]), 0)
        case_id = json.loads(output.getvalue())["cases"][0]["id"]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["resolve-case", str(case_id), str(self.directory), "--outcome", "unverifiable",
                                   "--reason", "User accepts missing history", "--allow-continue"]), 0)
        self.assertEqual(json.loads(output.getvalue())["validity"], "unverifiable")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["cases", str(self.directory), "--case", str(case_id)]), 0)
        case = json.loads(output.getvalue())["cases"][0]
        self.assertEqual(case["resolutions"][0]["explanation"], "User accepts missing history")
        self.assertFalse(case["blocking"])
        export = json.loads((self.directory / "memory-export.json").read_text())
        self.assertEqual(export["validity"], "unverifiable")

    def test_case_changes_are_refused_while_run_is_owned(self):
        with RunLock(self.directory / ".lock"):
            self.assertEqual(main(["case-open", str(self.directory), "--reason", "Concurrent edit"]), 1)
        self.assertEqual(self.memory.escalations.cases(), [])
