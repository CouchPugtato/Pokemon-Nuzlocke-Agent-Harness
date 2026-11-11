import contextlib
import errno
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pokemon_harness.cli import main
from pokemon_harness.config import load_config
from pokemon_harness.control.diagnostics import input_test
from pokemon_harness.control.registry import active_sessions
from pokemon_harness.control.session import RunControl
from pokemon_harness.demo import DemoAgent, create_images
from pokemon_harness.emulator.interface import Capture, ReplayEmulator
from pokemon_harness.memory.database import Memory
from pokemon_harness.runner import run
from pokemon_harness.telemetry.recorder import RunLock, save_capture, write_json
from pokemon_harness.telemetry.storage import MIB, StorageGuard, StorageLimitError, storage_settings


class StorageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = patch.dict("os.environ", {"POKEMON_HARNESS_SESSION_DIR": str(self.root / "registry")})
        environment.start()
        self.addCleanup(environment.stop)
        self.config_path = self.root / "config.json"
        write_json(self.config_path, {"run_dir": "run", "objective": "Test", "model": {"name": "test"},
                                     "emulator": {"backend": "replay", "screenshots_dir": "fixtures"},
                                     "sequence_delay_ms": 0,
                                     "storage": {"max_run_mb": 32, "min_free_mb": 0,
                                                 "action_headroom_mb": 1, "emergency_reserve_mb": 1}})
        self.config = load_config(self.config_path)
        self.directory = Path(self.config["run_dir"])
        self.images = create_images(self.root / "fixtures")

    def test_free_space_check_does_not_create_a_run_directory(self):
        guard = StorageGuard(self.config)
        with patch("pokemon_harness.telemetry.storage.shutil.disk_usage", return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(StorageLimitError, "Insufficient free disk space"):
                guard.check()
        self.assertFalse(self.directory.exists())

    def test_low_space_prevents_model_loading_and_emulator_creation(self):
        errors = io.StringIO()
        with patch("pokemon_harness.telemetry.storage.shutil.disk_usage", return_value=SimpleNamespace(free=0)), \
                patch("pokemon_harness.cli.LMStudio") as model, patch("pokemon_harness.cli.emulator_for") as emulator, \
                contextlib.redirect_stderr(errors):
            self.assertEqual(main(["run", "--config", str(self.config_path),
                                   "--runs-dir", str(self.root / "runs")]), 1)
        model.assert_not_called()
        emulator.assert_not_called()
        self.assertIn("Insufficient free disk space", errors.getvalue())

    def test_storage_failure_before_database_creation_still_closes_emulator(self):
        emulator, agent = Mock(), Mock()
        with patch("pokemon_harness.telemetry.storage.shutil.disk_usage", return_value=SimpleNamespace(free=0)):
            with self.assertRaises(StorageLimitError):
                run(self.config, emulator, agent, turns=1)
        emulator.close.assert_called_once()
        emulator.capture.assert_not_called()
        agent.decide.assert_not_called()
        self.assertFalse((self.directory / "memory.sqlite").exists())

    def test_capacity_counts_existing_files_and_required_headroom(self):
        self.directory.mkdir()
        with (self.directory / "archive.bin").open("wb") as stream:
            stream.truncate(32 * MIB)
        with self.assertRaisesRegex(StorageLimitError, "Run storage limit reached"):
            StorageGuard(self.config).check()
        self.assertEqual((self.directory / "archive.bin").stat().st_size, 32 * MIB)

    def test_emergency_reserve_is_allocated_and_released_without_touching_evidence(self):
        self.directory.mkdir()
        evidence = self.directory / "evidence.png"
        evidence.write_bytes(b"keep")
        guard = StorageGuard(self.config)
        guard.start()
        self.assertEqual(guard.reserve.stat().st_size, MIB)
        recovered = StorageGuard(self.config)
        recovered.start()
        recovered.release_reserve()
        self.assertFalse(guard.reserve.exists())
        self.assertEqual(evidence.read_bytes(), b"keep")

    def test_failed_reserve_allocation_preserves_error_and_releases_lock(self):
        original = OSError(errno.ENOSPC, "Original disk failure")
        emulator = Mock()
        emulator.close.side_effect = RuntimeError("Secondary close failure")
        with patch("pokemon_harness.telemetry.storage.os.fsync", side_effect=original), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(OSError) as failure:
                run(self.config, emulator, Mock(), turns=1)
        self.assertIs(failure.exception, original)
        self.assertFalse((self.directory / ".emergency-reserve").exists())
        with RunLock(self.directory / ".lock"):
            pass

    def test_folder_growth_during_inference_prevents_controller_input(self):
        directory = self.directory
        self.config["storage"]["max_run_mb"] = 4

        class GrowingAgent(DemoAgent):
            def decide(self, messages):
                with (directory / "external-growth.bin").open("wb") as stream:
                    stream.truncate(4 * MIB)
                return super().decide(messages)

        emulator = ReplayEmulator(self.images)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(StorageLimitError):
            run(self.config, emulator, GrowingAgent(), turns=1,
                control=RunControl(directory, hotkey=False))
        self.assertEqual(emulator.actions, [])
        self.assertEqual(active_sessions(), [])
        self.assertFalse((directory / ".emergency-reserve").exists())
        self.assertTrue((directory / "external-growth.bin").exists())

    def test_space_loss_after_input_preserves_unknown_outcome_and_cancels_remaining_moves(self):
        available = [100 * MIB]

        class FillingEmulator(ReplayEmulator):
            def execute(self, action):
                super().execute(action)
                available[0] = 0

        class SequenceAgent(DemoAgent):
            def decide(self, messages):
                value, trace = super().decide(messages)
                value["additional_actions"] = [dict(value["action"])]
                return value, trace

        emulator = FillingEmulator(self.images)
        with patch("pokemon_harness.telemetry.storage.shutil.disk_usage",
                   side_effect=lambda _: SimpleNamespace(free=available[0])), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(StorageLimitError):
            run(self.config, emulator, SequenceAgent(), turns=1)
        self.assertEqual(len(emulator.actions), 1)
        with contextlib.closing(Memory(self.directory / "memory.sqlite")) as memory:
            steps = memory.recent_turns()[0]["steps"]
            self.assertEqual([step["status"] for step in steps], ["unknown", "cancelled"])

    def test_failed_status_and_export_writes_do_not_mask_primary_model_failure(self):
        original = ValueError("Original model failure")
        agent = Mock()
        agent.decide.side_effect = original
        emulator = ReplayEmulator(self.images)

        def failing_export(path, value):
            if Path(path).name == "memory-export.json":
                raise OSError("Secondary export failure")
            write_json(path, value)

        errors = io.StringIO()
        original_note = Memory.note

        def fail_cleanup_notes(memory, kind, value):
            if kind == "session_start":
                return original_note(memory, kind, value)
            raise OSError("Secondary note failure")

        with patch.object(Memory, "note", autospec=True, side_effect=fail_cleanup_notes), \
                patch("pokemon_harness.runner.write_json", side_effect=failing_export), \
                contextlib.redirect_stderr(errors), self.assertRaises(ValueError) as failure:
            run(self.config, emulator, agent, turns=1,
                control=RunControl(self.directory, hotkey=False))
        self.assertIs(failure.exception, original)
        self.assertIn("Secondary export failure", errors.getvalue())
        self.assertIn("Secondary note failure", errors.getvalue())
        self.assertEqual(active_sessions(), [])
        self.assertFalse((self.directory / ".emergency-reserve").exists())

    def test_failed_json_write_keeps_prior_file_and_removes_partial_temporary(self):
        path = self.root / "record.json"
        write_json(path, {"prior": True})
        original = OSError(errno.ENOSPC, "No space")

        def partial_write(value, stream, **kwargs):
            stream.write("{")
            raise original

        with patch("pokemon_harness.telemetry.recorder.json.dump", side_effect=partial_write):
            with self.assertRaises(OSError) as failure:
                write_json(path, {"next": True})
        self.assertIs(failure.exception, original)
        self.assertEqual(path.read_text(), '{\n  "prior": true\n}')
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_failed_screenshot_write_removes_only_the_partial_new_file(self):
        original = OSError(errno.ENOSPC, "No space")
        png = self.images[0].read_bytes()
        with patch("pokemon_harness.telemetry.recorder.os.fsync", side_effect=original):
            with self.assertRaises(OSError):
                save_capture(self.root, Capture(png, "replay"))
        self.assertEqual(list((self.root / "screenshots").glob("*.png")), [])
        self.assertTrue(self.images[0].is_file())

    def test_diagnostic_input_is_blocked_when_disk_is_full(self):
        emulator = Mock()
        with patch("pokemon_harness.telemetry.storage.shutil.disk_usage", return_value=SimpleNamespace(free=0)):
            with self.assertRaises(StorageLimitError):
                input_test(self.config, emulator, "A")
        emulator.execute.assert_not_called()
        emulator.close.assert_called_once()

    def test_invalid_storage_settings_are_rejected(self):
        for settings in (False, {"max_run_mb": True}, {"min_free_mb": -1}, {"emergency_reserve_mb": 0},
                         {"max_run_mb": 2, "action_headroom_mb": 1, "emergency_reserve_mb": 1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                storage_settings({"storage": settings})
