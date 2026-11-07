import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pokemon_harness.cli import main
from pokemon_harness.config import load_config
from pokemon_harness.control.handoff import ControlLost
from pokemon_harness.control.session import RunControl, request_stop
from pokemon_harness.demo import DemoAgent, create_images
from pokemon_harness.emulator.desktop import WindowsDesktopEmulator
from pokemon_harness.emulator.interface import ReplayEmulator
from pokemon_harness.memory.database import Memory
from pokemon_harness.runner import run
from pokemon_harness.telemetry.recorder import write_json


class HandoffTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = patch.dict("os.environ", {"POKEMON_HARNESS_SESSION_DIR": str(self.root / "registry")})
        environment.start()
        self.addCleanup(environment.stop)
        path = self.root / "config.json"
        write_json(path, {"run_dir": "run", "objective": "Test", "rules": ["Test rule"],
                          "sequence_delay_ms": 0, "model": {"name": "test"},
                          "emulator": {"backend": "replay"}})
        self.config = load_config(path)
        self.directory = Path(self.config["run_dir"])
        self.images = create_images(self.root / "fixtures")

    def desktop(self):
        emulator = WindowsDesktopEmulator.__new__(WindowsDesktopEmulator)
        emulator.ownership = Mock()
        emulator.target = Mock()
        emulator._control_started = False
        emulator._control_lost = None
        emulator.display_name = "Test emulator"
        emulator.hwnd = 7
        emulator.config = {}
        emulator.user32 = Mock()
        emulator.user32.IsWindow.return_value = True
        emulator.user32.IsIconic.return_value = False
        emulator.user32.IsWindowEnabled.return_value = True
        emulator.user32.GetForegroundWindow.return_value = 7
        return emulator

    def test_initial_focus_is_allowed_once_but_loss_cannot_be_automatically_reversed(self):
        emulator = self.desktop()
        emulator.user32.GetForegroundWindow.return_value = 99

        def focus(hwnd):
            emulator.user32.GetForegroundWindow.return_value = hwnd

        emulator.user32.SetForegroundWindow.side_effect = focus
        with patch("pokemon_harness.emulator.desktop.time.sleep"):
            emulator._focus()
        emulator.user32.SetForegroundWindow.assert_called_once_with(7)
        emulator._focus()
        emulator.user32.GetForegroundWindow.return_value = 99
        with self.assertRaisesRegex(ControlLost, "lost focus"):
            emulator._focus()
        emulator.user32.GetForegroundWindow.return_value = 7
        with self.assertRaisesRegex(ControlLost, "lost focus"):
            emulator._focus()
        emulator.user32.SetForegroundWindow.assert_called_once()

    def test_read_only_watchdog_latches_loss_even_after_focus_returns(self):
        emulator = self.desktop()
        emulator._control_started = True
        emulator.user32.GetForegroundWindow.return_value = 99
        with self.assertRaises(ControlLost):
            emulator.check_control()
        emulator.user32.GetForegroundWindow.return_value = 7
        with self.assertRaises(ControlLost):
            emulator.check_control()
        emulator.user32.SetForegroundWindow.assert_not_called()
        emulator.user32.SendInput.assert_not_called()

    def test_focus_loss_during_inference_discards_pending_response_and_persists_handoff(self):
        trigger = threading.Event()
        observed = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)

        class FocusFixture(ReplayEmulator):
            def check_control(self):
                if trigger.is_set() and not observed.is_set():
                    observed.set()
                    raise ControlLost("Focus changed during generation")

        class PendingAgent:
            def decide(self, messages):
                trigger.set()
                release.wait(5)
                return DemoAgent().decide(messages)

        emulator = FocusFixture(self.images)
        result = run(self.config, emulator, PendingAgent(), turns=2,
                     control=RunControl(self.directory, hotkey=False))
        self.assertEqual(result["reason"], "control_lost")
        self.assertTrue(observed.is_set())
        self.assertEqual(emulator.actions, [])
        marker = json.loads((self.directory / ".handoff.json").read_text())
        self.assertEqual(marker["last_observation_id"], 1)
        memory = Memory(self.directory / "memory.sqlite")
        try:
            self.assertEqual(memory.recent_turns(), [])
        finally:
            memory.close()

    def test_manual_handoff_command_and_resume_warn_with_fresh_evidence(self):
        root = self.directory
        results = []
        release = threading.Event()
        self.addCleanup(release.set)

        class PendingAgent:
            def decide(self, messages):
                results.append(main(["handoff"]))
                release.wait(5)
                return DemoAgent().decide(messages)

        result = run(self.config, ReplayEmulator(self.images), PendingAgent(), turns=1,
                     control=RunControl(root, hotkey=False))
        self.assertEqual(results, [0])
        self.assertEqual(result["reason"], "manual_handoff")
        self.assertTrue((root / ".handoff.json").is_file())
        run(self.config, ReplayEmulator(self.images), DemoAgent(), turns=1,
            control=RunControl(root, hotkey=False))
        context = json.loads((root / "turns" / "observation-000002.context.json").read_text())
        self.assertEqual(context["intervention"]["reason"], "manual_handoff")
        self.assertEqual(context["intervention"]["last_observation_id"], 1)
        self.assertEqual(context["observation_id"], 2)
        self.assertFalse((root / ".handoff.json").exists())

    def test_focus_loss_cancels_remaining_sequence_moves(self):
        class FocusFixture(ReplayEmulator):
            def check_control(self):
                if self.actions:
                    raise ControlLost("Focus changed after first move")

        class SequenceAgent(DemoAgent):
            def decide(self, messages):
                value, trace = super().decide(messages)
                value["additional_actions"] = [dict(value["action"])]
                return value, trace

        emulator = FocusFixture(self.images)
        result = run(self.config, emulator, SequenceAgent(), turns=1,
                     control=RunControl(self.directory, hotkey=False))
        self.assertEqual(result["reason"], "control_lost")
        self.assertEqual(len(emulator.actions), 1)
        memory = Memory(self.directory / "memory.sqlite")
        try:
            self.assertEqual([step["status"] for step in memory.recent_turns()[0]["steps"]],
                             ["completed", "cancelled"])
        finally:
            memory.close()

    def test_stale_manual_handoff_request_does_not_affect_new_session(self):
        self.directory.mkdir()
        control = RunControl(self.directory, hotkey=False)
        control.start()
        try:
            write_json(self.directory / ".stop", {"session_id": "old-session", "kind": "handoff"})
            control.check()
        finally:
            control.close()

    def test_monitoring_stops_before_emulator_resources_close(self):
        control = RunControl(self.directory, hotkey=False)

        class CloseFixture(ReplayEmulator):
            def close(self):
                if control.watcher and control.watcher.is_alive():
                    raise AssertionError("Monitor is still using native target resources")

        run(self.config, CloseFixture(self.images), DemoAgent(), turns=1, control=control)
