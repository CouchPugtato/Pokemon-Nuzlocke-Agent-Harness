import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from pokemon_harness.cli import main, saved_run
from pokemon_harness.config import load_config
from pokemon_harness.control.session import RunControl, request_stop
from pokemon_harness.control.registry import active_sessions
from pokemon_harness.demo import DemoAgent, create_images
from pokemon_harness.emulator.interface import ReplayEmulator
from pokemon_harness.memory.database import Memory
from pokemon_harness.runner import run
from pokemon_harness.telemetry.recorder import RunLock, write_json


class SessionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = patch.dict("os.environ", {"POKEMON_HARNESS_SESSION_DIR": str(self.root / "registry")})
        environment.start()
        self.addCleanup(environment.stop)
        self.config_path = self.root / "config.json"
        write_json(self.config_path, {
            "run_dir": "old-run", "objective": "Test goal", "rules": ["Test rule"],
            "sequence_delay_ms": 0, "model": {"name": "test"},
            "emulator": {"backend": "replay", "screenshots_dir": "fixtures"},
        })
        self.config = load_config(self.config_path)
        self.images = create_images(self.root / "fixtures")

    def test_stop_while_model_is_pending_never_executes_its_response(self):
        directory = Path(self.config["run_dir"])
        control = RunControl(directory, hotkey=False)
        release = threading.Event()
        self.addCleanup(release.set)

        class PendingAgent:
            def decide(self, messages):
                request_stop(directory)
                release.wait(5)
                return DemoAgent().decide(messages)

        emulator = ReplayEmulator(self.images)
        result = run(self.config, emulator, PendingAgent(), turns=2, control=control)
        self.assertEqual(result["reason"], "user_stop")
        self.assertEqual(emulator.actions, [])
        self.assertTrue((directory / "memory-export.json").is_file())
        with RunLock(directory / ".lock"):
            pass

    def test_stop_after_first_move_cancels_sequence_and_resume_keeps_memory(self):
        directory = Path(self.config["run_dir"])
        control = RunControl(directory, hotkey=False)

        class SequenceAgent(DemoAgent):
            def decide(self, messages):
                decision, trace = super().decide(messages)
                decision["additional_actions"] = [dict(decision["action"])]
                return decision, trace

        class StoppingEmulator(ReplayEmulator):
            def execute(self, action):
                super().execute(action)
                request_stop(directory)

        emulator = StoppingEmulator(self.images)
        result = run(self.config, emulator, SequenceAgent(), turns=2, control=control)
        self.assertEqual(result["reason"], "user_stop")
        self.assertEqual(len(emulator.actions), 1)
        memory = Memory(directory / "memory.sqlite")
        try:
            steps = memory.recent_turns()[0]["steps"]
            self.assertEqual([step["status"] for step in steps], ["unknown", "cancelled"])
            self.assertIsNone(steps[0]["observation_id"])
            before = memory.previous_observation()["id"]
        finally:
            memory.close()
        run(self.config, ReplayEmulator(self.images), DemoAgent(), turns=1,
            control=RunControl(directory, hotkey=False))
        memory = Memory(directory / "memory.sqlite")
        try:
            self.assertGreater(memory.previous_observation()["id"], before)
            self.assertEqual(len(memory.recent_turns()), 2)
        finally:
            memory.close()

    def test_stop_command_rejects_inactive_run(self):
        directory = Path(self.config["run_dir"])
        directory.mkdir()
        with RunLock(directory / ".lock"):
            pass
        with self.assertRaisesRegex(ValueError, "not running"):
            request_stop(directory)

    def test_global_hotkey_requests_stop_without_emulator_focus(self):
        directory = self.root / "hotkey"
        directory.mkdir()
        control = RunControl(directory, hotkey=False)
        pressed = threading.Event()
        control.key_down = pressed.is_set
        control.start()
        try:
            pressed.set()
            self.assertTrue(control.stopped.wait(2))
        finally:
            control.close()
        self.assertFalse(control.watcher.is_alive())

    def test_cli_new_run_and_latest_resume_use_saved_settings(self):
        runs_dir = self.root / "runs"
        calls = []

        def simulated_run(config, emulator, agent, turns, control):
            calls.append(config)
            directory = Path(config["run_dir"])
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory / "config.json", config)
            memory = Memory(directory / "memory.sqlite")
            memory.initialize(config)
            memory.close()
            return {"reason": "turn_limit"}

        with patch("pokemon_harness.cli.run", side_effect=simulated_run), \
                patch("pokemon_harness.cli.emulator_for"), patch("pokemon_harness.cli.LMStudio"):
            self.assertEqual(main(["run", "--config", str(self.config_path),
                                   "--runs-dir", str(runs_dir), "--goal", "first-gym"]), 0)
            directory = Path(calls[0]["run_dir"])
            self.assertEqual(directory.parent, runs_dir)
            self.assertIn("Stone Badge", calls[0]["objective"])
            self.config_path.unlink()
            self.assertEqual(main(["resume", "--runs-dir", str(runs_dir)]), 0)
            self.assertEqual(calls[1], calls[0])
            self.assertEqual(saved_run(None, runs_dir), directory)
            self.assertEqual(main(["resume", "--runs-dir", str(runs_dir),
                                   "--max-tokens", "4096", "--thinking", "off"]), 0)
            self.assertEqual(calls[2]["model"]["max_tokens"], 4096)
            self.assertFalse(calls[2]["model"]["thinking"])
            self.assertEqual(calls[2]["rules"], calls[0]["rules"])
            self.assertEqual(main(["resume", "--runs-dir", str(runs_dir)]), 0)
            self.assertEqual(calls[3]["model"], calls[2]["model"])

    def test_new_run_refuses_existing_memory(self):
        directory = self.root / "occupied"
        directory.mkdir()
        (directory / "memory.sqlite").write_bytes(b"keep me")
        self.assertEqual(main(["run", "--config", str(self.config_path),
                               "--run-dir", str(directory)]), 1)
        self.assertEqual((directory / "memory.sqlite").read_bytes(), b"keep me")

    def test_failed_export_still_closes_emulator_and_unregisters_session(self):
        directory = Path(self.config["run_dir"])
        control = RunControl(directory, hotkey=False)
        emulator = ReplayEmulator(self.images)

        def failing_export(path, value):
            if Path(path).name == "memory-export.json":
                raise OSError("Export failed")
            write_json(path, value)

        with patch("pokemon_harness.runner.write_json", side_effect=failing_export), \
                patch.object(emulator, "close") as close:
            with self.assertRaisesRegex(OSError, "Export failed"):
                run(self.config, emulator, DemoAgent(), turns=1, control=control)
        close.assert_called_once()
        self.assertEqual(active_sessions(), [])
        with RunLock(directory / ".lock"):
            pass
