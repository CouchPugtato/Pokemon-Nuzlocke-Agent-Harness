import json
import tempfile
import threading
import unittest
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

from pokemon_harness.agent.lmstudio import LMStudio
from pokemon_harness.agent.prompts import messages
from pokemon_harness.agent.schemas import validate_decision
from pokemon_harness.config import load_config
from pokemon_harness.cli import emulator_for
from pokemon_harness.control.diagnostics import input_test
from pokemon_harness.demo import DemoAgent, create_images
from pokemon_harness.emulator.interface import Capture, ReplayEmulator
from pokemon_harness.emulator.mgba import MGBA
from pokemon_harness.emulator.vbam import VBAM
from pokemon_harness.memory.database import Memory
from pokemon_harness.memory.retrieval import ledger_context, retrieve
from pokemon_harness.runner import run
from pokemon_harness.telemetry.recorder import RunLock, write_json


def decision(observation_id=1):
    return DemoAgent().decide(messages({"observation_id": observation_id}, b"png"))[0]


def event(key="route101:first", kind="encounter", confidence="high", corrects=""):
    return {"key": key, "type": kind, "subject": "Zigzagoon", "area": "Route 101",
            "details": "Visible encounter", "confidence": confidence, "corrects_event_key": corrects}


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        path = self.root / "config.json"
        write_json(path, {"run_dir": "run", "objective": "Synthetic test", "rules": ["Retire fainted Pokemon"],
                          "model": {"name": "test"}, "emulator": {"backend": "replay"}})
        self.config = load_config(path)

    def memory(self):
        memory = Memory(self.root / "memory.sqlite")
        self.addCleanup(memory.close)
        memory.initialize(self.config)
        memory.observe("screenshots/one.png", "hash", "replay")
        return memory

    def test_action_validation(self):
        value = decision()
        validate_decision(value, 1)
        with self.assertRaises(ValueError):
            validate_decision(value, 2)
        value["action"]["buttons"] = ["UP", "DOWN"]
        with self.assertRaises(ValueError):
            validate_decision(value, 1)
        value["action"]["buttons"] = ["UP"]
        value["action"]["hold_frames"] = 61
        with self.assertRaises(ValueError):
            validate_decision(value, 1)

    def test_memory_revisions_and_atomic_ledger(self):
        memory = self.memory()
        first = decision()
        first["events"] = [event()]
        turn = memory.plan(1, first)
        memory.status(turn, "completed")
        memory.observe("two.png", "hash2", "replay")
        second = decision(2)
        second["memory_updates"][0]["claim"] = "Updated landmark"
        second["events"] = [event()]  # identical event must be deduplicated
        memory.plan(2, second)
        self.assertEqual(len(memory.events()), 1)
        self.assertEqual(memory.facts()[0]["claim"], "Updated landmark")
        memory.observe("three.png", "hash3", "replay")
        conflict = decision(3)
        conflict["events"] = [event("new-event"), dict(event(), details="Changed history")]
        with self.assertRaises(ValueError):
            memory.plan(3, conflict)
        self.assertEqual(len(memory.events()), 1)  # entire transaction rolled back
        self.assertEqual(memory.facts()[0]["claim"], "Updated landmark")

    def test_crash_recovery_never_replays_pending_actions(self):
        memory = self.memory()
        turn = memory.plan(1, decision())
        memory.status(turn, "executing")
        self.assertEqual(memory.recover(), [{"id": turn, "status": "executing"}])
        self.assertEqual(memory.recent_turns()[0]["status"], "unknown")
        self.assertEqual(memory.recover(), [])

    def test_run_identity_cannot_silently_change(self):
        memory = self.memory()
        config = dict(self.config, rules=["Different rules"])
        with self.assertRaises(ValueError):
            memory.initialize(config)

    def test_end_to_end_and_resume(self):
        images = create_images(self.root / "fixtures")
        emulator = ReplayEmulator(images)
        result = run(self.config, emulator, DemoAgent(), turns=3)
        self.assertEqual(result, {"completed_actions": 2, "reason": "agent_stop"})
        self.assertEqual(len(emulator.actions), 2)
        database = Memory(Path(self.config["run_dir"]) / "memory.sqlite")
        before = database.previous_observation()["id"]
        self.assertEqual(len(database.facts()), 1)
        database.close()
        run(self.config, ReplayEmulator(images), DemoAgent(), turns=1)
        database = Memory(Path(self.config["run_dir"]) / "memory.sqlite")
        self.assertGreater(database.previous_observation()["id"], before)
        self.assertEqual(len(database.recent_turns()), 4)
        database.close()

    def test_uncertain_event_blocks_buttons_but_can_be_corrected(self):
        images = create_images(self.root / "fixtures")

        class UncertainAgent(DemoAgent):
            def decide(self, messages):
                value, trace = super().decide(messages)
                value["events"] = [event(confidence="low")]
                return value, trace

        emulator = ReplayEmulator(images)
        result = run(self.config, emulator, UncertainAgent(), turns=1)
        self.assertEqual(result["reason"], "unresolved_events")
        self.assertEqual(emulator.actions, [])

        class CorrectionAgent(DemoAgent):
            def decide(self, messages):
                value, trace = super().decide(messages)
                value["events"] = [event("correction:1", "correction", "high", "route101:first")]
                return value, trace

        result = run(self.config, ReplayEmulator(images), CorrectionAgent(), turns=1)
        self.assertEqual(result["completed_actions"], 1)

    def test_input_failure_is_unknown_and_no_retry(self):
        class BrokenEmulator(ReplayEmulator):
            def execute(self, action):
                self.actions.append(action)
                raise TimeoutError("Input may have happened")

        emulator = BrokenEmulator(create_images(self.root / "fixtures"))
        with self.assertRaises(TimeoutError):
            run(self.config, emulator, DemoAgent(), turns=3)
        self.assertEqual(len(emulator.actions), 1)
        memory = Memory(Path(self.config["run_dir"]) / "memory.sqlite")
        self.assertEqual(memory.recent_turns()[0]["status"], "unknown")
        memory.close()

    def test_old_deaths_survive_bounded_event_context(self):
        events = [event("death", "faint")] + [event(f"progress-{i}", "progress") for i in range(100)]
        ledger = ledger_context(events, "Route 102", limit=4)
        self.assertIn("Zigzagoon", ledger["retired_pokemon"])
        self.assertEqual(len(ledger["recent_or_local_events"]), 4)

    def test_run_lock(self):
        path = self.root / ".lock"
        with RunLock(path):
            with self.assertRaises(RuntimeError):
                with RunLock(path):
                    pass
        with RunLock(path):
            pass

    def test_stale_model_decision_never_reaches_controller(self):
        class StaleAgent(DemoAgent):
            def decide(self, messages):
                value, trace = super().decide(messages)
                value["observation_id"] += 10
                return value, trace

        emulator = ReplayEmulator(create_images(self.root / "fixtures"))
        with self.assertRaises(ValueError):
            run(self.config, emulator, StaleAgent(), turns=1)
        self.assertEqual(emulator.actions, [])

    def test_retrieval_prioritizes_objectives_and_exposes_fact_age(self):
        facts = []
        for index, category in enumerate(["objective", "world", "party"], 1):
            fact = dict(decision()["memory_updates"][0], category=category,
                        evidence_observation_id=index)
            facts.append(fact)
        selected = retrieve(facts, "doorway", observation_id=10, limit=1)
        self.assertEqual(selected[0]["category"], "objective")
        self.assertEqual(selected[0]["age_observations"], 9)

    def test_desktop_input_failure_releases_keys_without_frontend_commands(self):
        # Exercise the real execute/finally logic without generating OS input.
        class FakeMGBA(MGBA):
            def __init__(self):
                self.config = {}
                self.bindings = {"UP": "UP"}
                self.held = set()
                self.released = False
                self.fps = 60
                self.min_hold_frames = 6
                self.input_mode = "scancode"

            def _check_focus(self):
                pass

            def _key(self, key, up=False):
                self.held.add(key)
                raise RuntimeError("Simulated focus loss after key down")

            def _release(self):
                self.held.clear()
                self.released = True

        emulator = FakeMGBA()
        with self.assertRaises(RuntimeError):
            emulator.execute(decision()["action"])
        self.assertTrue(emulator.released)
        self.assertFalse(emulator.held)
        emulator.close()

    def test_win32_input_structure_matches_platform_abi(self):
        import ctypes
        import os

        if os.name != "nt":
            self.skipTest("Windows ABI check")
        emulator = MGBA.__new__(MGBA)
        emulator.user32 = ctypes.WinDLL("user32", use_last_error=True)
        emulator._configure_api()
        self.assertEqual(ctypes.sizeof(emulator.Input), 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28)

    def test_desktop_backend_selection_preserves_settings(self):
        for backend, target in [("mgba", "pokemon_harness.emulator.mgba.MGBA"),
                                ("vbam", "pokemon_harness.emulator.vbam.VBAM")]:
            with self.subTest(backend=backend), patch(target) as factory:
                settings = {"backend": backend, "window_title": "Custom title"}
                result = emulator_for({"emulator": settings}, self.root / "config.json")
                factory.assert_called_once_with(settings)
                self.assertIs(result, factory.return_value)

    def test_user_controls_accept_caps_lock_and_standalone_shift(self):
        # VBA-M displays Caps Lock as CAPITAL and standalone Shift as 306:4.
        self.assertEqual(MGBA._combo("CAPSLOCK"), [0x14])
        self.assertEqual(MGBA._combo("CAPITAL"), [0x14])
        self.assertEqual(MGBA._combo("SHIFT"), [0x10])

    def test_runner_uses_capture_timing_for_vbam(self):
        class DesktopFixture(ReplayEmulator):
            def capture(self):
                return Capture(super().capture().image, "live; game continues during inference")

        config = dict(self.config, emulator={"backend": "vbam"})
        emulator = DesktopFixture(create_images(self.root / "fixtures"))
        run(config, emulator, DemoAgent(), turns=1)
        context_path = next((Path(config["run_dir"]) / "turns").glob("*.context.json"))
        context = json.loads(context_path.read_text(encoding="utf-8"))
        self.assertEqual(context["timing"], "live; game continues during inference")

    def test_lmstudio_http_contract(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                body = json.dumps({"choices": [{"finish_reason": "stop", "message": {
                    "content": json.dumps(decision())}}]}).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            agent = LMStudio({"name": "local-vision", "base_url": f"http://127.0.0.1:{server.server_port}/v1"})
            value, trace = agent.decide(messages({"observation_id": 1}, b"fake-png"))
            validate_decision(value, 1)
            self.assertEqual(received[0][0], "/v1/chat/completions")
            payload = received[0][1]
            self.assertEqual(payload["response_format"]["type"], "json_schema")
            self.assertTrue(payload["messages"][1]["content"][-1]["image_url"]["url"].startswith("data:image/png;base64,"))
            self.assertNotIn("messages", trace["request_settings"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


    def test_lmstudio_timeout_identifies_inference_and_does_not_retry(self):
        agent = LMStudio({"name": "test", "timeout_seconds": 180})
        with patch("pokemon_harness.agent.lmstudio.urlopen", side_effect=TimeoutError("timed out")) as request:
            with self.assertRaisesRegex(TimeoutError, "LM Studio.*180s"):
                agent.decide(messages({"observation_id": 1}, b"fake-png"))
        self.assertEqual(request.call_count, 1)

    def test_minimum_press_preserves_wait_duration_and_reports_actual_input(self):
        clock = SimpleNamespace(value=0.0)

        def sleep(seconds):
            clock.value += seconds

        class FakeDesktop(MGBA):
            def __init__(self):
                self.config = {}
                self.bindings = {"START": "SHIFT"}
                self.held = set()
                self.fps = 60
                self.min_hold_frames = 6
                self.input_mode = "scancode"
                self.events = []
                self.config["focus_game_area"] = True

            def _check_focus(self):
                pass

            def _focus_game_area(self):
                self.events.append(("canvas", "focus", clock.value))

            def _key(self, key, up=False):
                self.events.append((key, "up" if up else "down", clock.value))
                if up:
                    self.held.discard(key)
                else:
                    self.held.add(key)

        emulator = FakeDesktop()
        action = {"kind": "press", "buttons": ["START"], "hold_frames": 1, "settle_frames": 0}
        with patch("pokemon_harness.emulator.desktop.time.monotonic", side_effect=lambda: clock.value), \
                patch("pokemon_harness.emulator.desktop.time.sleep", side_effect=sleep):
            report = emulator.execute(action)
            down = next(e[2] for e in emulator.events if e[1] == "down")
            up = next(e[2] for e in emulator.events if e[1] == "up")
            self.assertGreaterEqual(up - down, 0.1)
            self.assertEqual(report["effective_action"]["hold_frames"], 6)
            self.assertEqual(action["hold_frames"], 1)
            self.assertEqual(report["game_timing"], "continuous")
            self.assertEqual([e[0] for e in emulator.events if e[1] in {"down", "up"}], [0x10, 0x10])
            report = emulator.execute(dict(action, kind="wait", buttons=[]))
            self.assertEqual(report["effective_action"]["hold_frames"], 1)
            self.assertFalse(emulator.held)

    def test_shift_and_extended_keys_encode_real_scan_codes(self):
        import ctypes
        import os

        if os.name != "nt":
            self.skipTest("Windows ABI check")
        emulator = MGBA.__new__(MGBA)
        emulator.user32 = ctypes.WinDLL("user32", use_last_error=True)
        emulator._configure_api()
        emulator.user32 = SimpleNamespace(MapVirtualKeyW=lambda key, _: {0x10: 0x2A, 0x25: 0xE04B}[key])
        emulator.input_mode = "scancode"
        shift = emulator._keyboard_input(0x10)
        self.assertEqual((shift.ki.wVk, shift.ki.wScan, shift.ki.dwFlags), (0, 0x2A, 8))
        release = emulator._keyboard_input(0x10, up=True)
        self.assertEqual(release.ki.dwFlags, 8 | 2)
        left = emulator._keyboard_input(0x25)
        self.assertEqual((left.ki.wScan, left.ki.dwFlags), (0x4B, 8 | 1))
        emulator.input_mode = "virtual_key"
        shift = emulator._keyboard_input(0x10)
        self.assertEqual((shift.ki.wVk, shift.ki.wScan, shift.ki.dwFlags), (0x10, 0x2A, 0))

    def test_input_diagnostic_archives_before_after_without_touching_memory(self):
        emulator = ReplayEmulator(create_images(self.root / "fixtures"))
        report = input_test(self.config, emulator, "START", 12, 45)
        self.assertEqual(report["status"], "submitted")
        self.assertEqual(len(emulator.actions), 1)
        self.assertFalse((Path(self.config["run_dir"]) / "memory.sqlite").exists())
        directory = next((Path(self.config["run_dir"]) / "input-tests").iterdir())
        self.assertTrue((directory / report["before"]).is_file())
        self.assertTrue((directory / report["after"]).is_file())


    def test_capture_and_close_do_not_access_keyboard_or_menu_controls(self):
        from PIL import Image

        def rect(_hwnd, result):
            result._obj.right, result._obj.bottom = 240, 160
            return 1

        for backend in (MGBA, VBAM):
            with self.subTest(backend=backend.__name__):
                emulator = backend.__new__(backend)
                emulator.config = {}
                emulator.hwnd = 1
                emulator.held = set()
                emulator._focus = lambda: None
                emulator._check_focus = lambda: None
                # Only capture geometry APIs are available. Any keyboard/menu
                # operation during capture or shutdown must fail this test.
                emulator.user32 = SimpleNamespace(GetClientRect=rect, ClientToScreen=lambda *_: 1)
                with patch("PIL.ImageGrab.grab", return_value=Image.new("RGB", (240, 160))):
                    capture = emulator.capture()
                self.assertIn("game continues during inference", capture.timing)
                self.assertTrue(capture.image.startswith(b"\x89PNG"))
                emulator.close()


if __name__ == "__main__":
    unittest.main()
