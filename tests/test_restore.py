import contextlib
import errno
import io
import json
import tempfile
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pokemon_harness.agent.prompts import messages
from pokemon_harness.cli import main, saved_run
from pokemon_harness.config import load_config
from pokemon_harness.demo import DemoAgent, create_images
from pokemon_harness.emulator.interface import ReplayEmulator
from pokemon_harness.memory.database import Memory
from pokemon_harness.runner import run
from pokemon_harness.telemetry.backup import _fingerprint, create_backup, verify_backup
from pokemon_harness.telemetry.recorder import write_json
from pokemon_harness.telemetry.restore import restore_backup
from pokemon_harness.telemetry.storage import StorageLimitError


class RestoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        path = self.root / "config.json"
        write_json(path, {"run_dir": "run", "objective": "Test", "rules": ["Test rule"],
                          "model": {"name": "test"}, "emulator": {"backend": "replay"},
                          "storage": {"max_run_mb": 32, "max_backup_mb": 32, "min_free_mb": 0,
                                      "action_headroom_mb": 1, "emergency_reserve_mb": 1}})
        self.config = load_config(path)
        self.source = Path(self.config["run_dir"])
        self.images = create_images(self.root / "fixtures")
        run(self.config, ReplayEmulator(self.images), DemoAgent(), turns=2)
        self.backup = create_backup(self.source, self.root / "backups")
        self.target = self.root / "restored"

    def test_restoration_preserves_history_and_backup_but_adds_a_blocking_review(self):
        before = {path.relative_to(self.backup): _fingerprint(path) for path in self.backup.rglob("*") if path.is_file()}
        result = restore_backup(self.backup, self.target)
        self.assertTrue(result["review_required"])
        self.assertFalse(result["game_state_verified"])
        config = load_config(self.target / "config.json")
        self.assertEqual(config["run_dir"], str(self.target))
        with contextlib.closing(Memory(self.target / "memory.sqlite")) as memory:
            self.assertEqual(len(memory.recent_turns()), 2)
            self.assertEqual(memory.previous_observation()["id"], 3)
            cases = memory.escalations.review(memory.events())["blocked_cases"]
            self.assertEqual(cases[-1]["id"], result["case_id"])
            self.assertEqual(cases[-1]["details"]["kind"], "restore_reconciliation")
        marker = json.loads((self.target / ".handoff.json").read_text())
        self.assertEqual(marker["reason"], "backup_restored")
        self.assertFalse((self.target / "manifest.json").exists())
        self.assertTrue((self.target / "restores" / result["restore_id"] / "manifest.json").is_file())
        after = {path.relative_to(self.backup): _fingerprint(path) for path in self.backup.rglob("*") if path.is_file()}
        self.assertEqual(after, before)
        self.assertTrue(verify_backup(self.backup)["verified"])

    def test_restored_run_blocks_before_model_or_emulator_startup(self):
        restore_backup(self.backup, self.target)
        with patch("pokemon_harness.cli.LMStudio") as model, patch("pokemon_harness.cli.emulator_for") as emulator:
            self.assertEqual(main(["resume", str(self.target)]), 1)
        model.assert_not_called()
        emulator.assert_not_called()
        with contextlib.closing(Memory(self.target / "memory.sqlite")) as memory:
            config = load_config(self.target / "config.json")
            fake = Mock()
            self.assertEqual(run(config, fake, Mock(), turns=1)["reason"], "review_required")
            fake.capture.assert_not_called()
            fake.execute.assert_not_called()

    def test_backup_cannot_be_resumed_or_opened_by_mutating_commands(self):
        before = _fingerprint(self.backup / "memory.sqlite")
        with patch("pokemon_harness.cli.LMStudio") as model, patch("pokemon_harness.cli.emulator_for") as emulator:
            for arguments in (["resume", str(self.backup)], ["cases", str(self.backup)],
                              ["memory", "--run-dir", str(self.backup)], ["backup", str(self.backup)],
                              ["input-test", "--config", str(self.backup / "config.json")],
                              ["demo", "--run-dir", str(self.backup)]):
                with self.subTest(arguments=arguments):
                    self.assertEqual(main(arguments), 1)
        model.assert_not_called()
        emulator.assert_not_called()
        config = load_config(self.backup / "config.json")
        with self.assertRaisesRegex(ValueError, "snapshots cannot"):
            run(config, Mock(), Mock(), turns=1)
        self.assertEqual(_fingerprint(self.backup / "memory.sqlite"), before)
        self.assertFalse((self.backup / ".lock").exists())
        self.assertTrue(verify_backup(self.backup)["verified"])

    def test_incomplete_staging_is_not_discovered_or_resumable(self):
        stage = self.root / ".restore-incomplete"
        stage.mkdir()
        shutil.copyfile(self.source / "config.json", stage / "config.json")
        shutil.copyfile(self.source / "memory.sqlite", stage / "memory.sqlite")
        self.assertEqual(saved_run(None, self.root), self.source)
        with self.assertRaisesRegex(ValueError, "Incomplete snapshot staging"):
            saved_run(stage, self.root)
        with patch("pokemon_harness.cli.emulator_for") as emulator:
            self.assertEqual(main(["resume", str(stage)]), 1)
        emulator.assert_not_called()

    def test_existing_destination_is_never_overwritten(self):
        self.target.mkdir()
        with self.assertRaisesRegex(ValueError, "never overwrites"):
            restore_backup(self.backup, self.target)
        sentinel = self.target / "keep"
        sentinel.write_bytes(b"original")
        with self.assertRaises(ValueError):
            restore_backup(self.backup, self.target)
        self.assertEqual(sentinel.read_bytes(), b"original")

    def test_corrupted_backup_is_rejected_before_destination_creation(self):
        next((self.backup / "screenshots").glob("*.png")).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "corrupted"):
            restore_backup(self.backup, self.target)
        self.assertFalse(self.target.exists())
        self.assertEqual(list(self.root.glob(".restore-*/")), [])

    def test_copy_failure_removes_staging_and_keeps_source_snapshot(self):
        original = OSError(errno.ENOSPC, "Simulated full disk")
        with patch("pokemon_harness.telemetry.restore.shutil.copyfileobj", side_effect=original):
            with self.assertRaises(OSError) as error:
                restore_backup(self.backup, self.target)
        self.assertIs(error.exception, original)
        self.assertFalse(self.target.exists())
        self.assertEqual([path for path in self.root.glob(".restore-*") if path.is_dir()], [])
        self.assertTrue(verify_backup(self.backup)["verified"])

    def test_low_space_prevents_restore_publication(self):
        with patch("pokemon_harness.telemetry.storage.shutil.disk_usage", return_value=SimpleNamespace(free=0)):
            with self.assertRaises(StorageLimitError):
                restore_backup(self.backup, self.target)
        self.assertFalse(self.target.exists())
        self.assertTrue(verify_backup(self.backup)["verified"])

    def test_copy_corruption_is_detected_before_publication(self):
        def corrupt_copy(reader, writer, **kwargs):
            writer.write(b"wrong file")

        with patch("pokemon_harness.telemetry.restore.shutil.copyfileobj", side_effect=corrupt_copy):
            with self.assertRaisesRegex(ValueError, "changed while restoring"):
                restore_backup(self.backup, self.target)
        self.assertFalse(self.target.exists())

    def test_pending_actions_are_recovered_as_unknown_without_execution(self):
        with contextlib.closing(Memory(self.source / "memory.sqlite")) as memory:
            value, trace = DemoAgent().decide(messages(
                {"observation_id": 3, "outcome_review": memory.outcomes.context()}, b"png"))
            turn = memory.plan(3, value)
            memory.status(turn, "executing")
            memory.step_status(turn, 1, "executing")
        backup = create_backup(self.source, self.root / "backups")
        restore_backup(backup, self.target)
        with contextlib.closing(Memory(self.target / "memory.sqlite")) as memory:
            self.assertEqual(memory.recent_turns()[-1]["status"], "unknown")
            self.assertEqual(memory.recent_turns()[-1]["steps"][0]["status"], "unknown")

    def test_explicit_override_remains_unverifiable_and_preserves_prior_invalid_status(self):
        result = restore_backup(self.backup, self.target)
        with contextlib.closing(Memory(self.target / "memory.sqlite")) as memory:
            memory.escalations.resolve(result["case_id"], "user-assertion", "User checked the emulator manually",
                                        allow_continue=True)
            self.assertEqual(memory.escalations.validity(), "unverifiable")
            self.assertEqual(memory.escalations.review(memory.events())["blocked_cases"], [])
        config = load_config(self.target / "config.json")
        outcome = run(config, ReplayEmulator(self.images), DemoAgent(), turns=1)
        self.assertEqual(outcome["completed_actions"], 1)
        with contextlib.closing(Memory(self.source / "memory.sqlite")) as memory:
            case = memory.escalations.open("invalid", "Confirmed rule violation")
            memory.escalations.resolve(case, "invalid", "Run invalid")
        invalid_backup = create_backup(self.source, self.root / "backups")
        invalid_target = self.root / "invalid-restored"
        invalid_result = restore_backup(invalid_backup, invalid_target)
        with contextlib.closing(Memory(invalid_target / "memory.sqlite")) as memory:
            memory.escalations.resolve(invalid_result["case_id"], "user-assertion", "User checked emulator",
                                        allow_continue=True)
            self.assertEqual(memory.escalations.validity(), "invalid")

    def test_cli_restore_chooses_new_directory_without_emulator_or_model_calls(self):
        output = io.StringIO()
        with patch("pokemon_harness.cli.emulator_for") as emulator, patch("pokemon_harness.cli.LMStudio") as model, \
                contextlib.redirect_stdout(output):
            self.assertEqual(main(["restore", str(self.backup), "--runs-dir", str(self.root / "runs")]), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(Path(result["run_dir"]).parent, self.root / "runs")
        self.assertTrue(result["review_required"])
        emulator.assert_not_called()
        model.assert_not_called()

    def test_restored_run_can_be_backed_up_again_with_provenance_and_cases(self):
        first = restore_backup(self.backup, self.target)
        backup = create_backup(self.target, self.root / "backups")
        second = restore_backup(backup, self.root / "restored-again")
        self.assertNotEqual(first["restore_id"], second["restore_id"])
        with contextlib.closing(Memory(Path(second["run_dir"]) / "memory.sqlite")) as memory:
            self.assertEqual(len(memory.escalations.review(memory.events())["blocked_cases"]), 2)
