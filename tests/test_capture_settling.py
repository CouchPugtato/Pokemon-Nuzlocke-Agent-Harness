import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from pokemon_harness.config import load_config
from pokemon_harness.control.capture import capture_settings, settled_capture
from pokemon_harness.control.handoff import ControlLost
from pokemon_harness.control.session import RunControl, StopRequested, request_stop
from pokemon_harness.demo import DemoAgent, create_images
from pokemon_harness.emulator.interface import Capture, ReplayEmulator
from pokemon_harness.memory.archive import archive_settings, retrieve_images
from pokemon_harness.memory.database import Memory
from pokemon_harness.runner import run
from pokemon_harness.telemetry.recorder import write_json


def png(color, size=(4, 4)):
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, format="PNG")
    return stream.getvalue()


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.waits = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


class SamplingEmulator:
    def __init__(self, images, clock=None, latency=0):
        self.images = images
        self.calls = 0
        self.clock = clock
        self.latency = latency
        self.held = set()

    def capture(self):
        if self.clock:
            self.clock.now += self.latency
        image = self.images[min(self.calls, len(self.images) - 1)]
        self.calls += 1
        return Capture(image, "fixture timing")


class CaptureSettlingTests(unittest.TestCase):
    def test_post_input_delay_prevents_reviewing_a_stable_old_frame(self):
        clock = FakeClock()
        black, white = png("black"), png("white")

        class DelayedResponse(SamplingEmulator):
            def capture(self):
                self.calls += 1
                return Capture(white if clock.now >= 0.5 else black, "fixture timing")

        emulator = DelayedResponse([])
        settings = capture_settings({"capture_settling": {"enabled": True}})
        capture, report = settled_capture(emulator, settings, scene="dialogue", clock=clock,
                                         sleep=clock.sleep, after_input=True)
        self.assertEqual(capture.image, white)
        self.assertEqual(report["after_input_delay_ms"], 750)
        self.assertEqual(report["status"], "stable")
        self.assertGreaterEqual(clock.now, 0.9)
        self.assertLessEqual(clock.now, 1.0)

    def test_post_input_delay_is_configurable_and_not_applied_to_other_captures(self):
        settings = capture_settings({"capture_settling": {"enabled": True, "after_input_delay_ms": 100}})
        for after_input, expected in ((False, 0), (True, 100)):
            clock = FakeClock()
            _, report = settled_capture(SamplingEmulator([png("black")]), settings, clock=clock,
                                        sleep=clock.sleep, after_input=after_input)
            self.assertEqual(report["after_input_delay_ms"], expected)
        for value in (-1, 5001, True):
            with self.assertRaises(ValueError):
                capture_settings({"capture_settling": {"after_input_delay_ms": value}})

    def sample(self, images, scene="unknown", **changes):
        clock = FakeClock()
        emulator = SamplingEmulator(images)
        settings = capture_settings({"capture_settling": dict(enabled=True, min_delay_ms=0, interval_ms=10,
            max_ms=100, scene_max_ms={}, **changes)})
        result = settled_capture(emulator, settings, scene=scene, clock=clock, sleep=clock.sleep)
        return result, emulator, clock

    def test_stability_requires_consecutive_comparisons_and_returns_last_sample(self):
        black, white = png("black"), png("white")
        (capture, report), emulator, clock = self.sample([black, white, white, white])
        self.assertEqual(report["status"], "stable")
        self.assertEqual(emulator.calls, 4)
        self.assertEqual(capture.image, white)
        self.assertEqual(report["differences"], [None, 255, 0, 0])
        self.assertFalse(report["semantic_completion_verified"])
        self.assertLessEqual(clock.now, 0.1)

    def test_continuous_animation_stops_at_sample_limit(self):
        (_, report), emulator, _ = self.sample([png("black"), png("white")] * 10, max_samples=5)
        self.assertEqual(report["status"], "sample_limit")
        self.assertEqual(emulator.calls, 5)

    def test_sampling_stops_at_deadline_without_an_extra_capture(self):
        clock = FakeClock()
        emulator = SamplingEmulator([png("black"), png("white")] * 10)
        settings = capture_settings({"capture_settling": {"enabled": True, "min_delay_ms": 0,
            "interval_ms": 10, "max_ms": 25, "scene_max_ms": {}}})
        _, report = settled_capture(emulator, settings, clock=clock, sleep=clock.sleep)
        self.assertEqual(report["status"], "time_limit")
        self.assertEqual(emulator.calls, 3)
        self.assertAlmostEqual(clock.now, 0.025)

    def test_scene_limit_cannot_expand_the_global_budget(self):
        settings = capture_settings({"capture_settling": {"enabled": True, "max_ms": 200,
            "min_delay_ms": 0, "scene_max_ms": {"battle": 20, "transition": 1000}}})
        for scene, expected in (("battle", 20), ("transition", 200)):
            clock = FakeClock()
            emulator = SamplingEmulator([png("black"), png("white")] * 20)
            _, report = settled_capture(emulator, settings, scene=scene, clock=clock, sleep=clock.sleep)
            self.assertEqual(report["budget_ms"], expected)
            self.assertLessEqual(report["elapsed_ms"], expected)

    def test_slow_native_capture_reports_overrun_without_retry(self):
        clock = FakeClock()
        emulator = SamplingEmulator([png("black")], clock, latency=1)
        settings = capture_settings({"capture_settling": {"enabled": True, "min_delay_ms": 0}})
        _, report = settled_capture(emulator, settings, clock=clock, sleep=clock.sleep)
        self.assertEqual(report["status"], "time_limit")
        self.assertTrue(report["deadline_exceeded"])
        self.assertEqual(emulator.calls, 1)

    def test_minimum_delay_and_disabled_mode_are_distinct(self):
        clock = FakeClock()
        settings = capture_settings({"capture_settling": {"enabled": True, "min_delay_ms": 50}})
        _, report = settled_capture(SamplingEmulator([png("black")]), settings, clock=clock, sleep=clock.sleep)
        self.assertGreaterEqual(report["elapsed_ms"], 200)
        clock = FakeClock()
        _, report = settled_capture(SamplingEmulator([png("black")]), dict(settings, enabled=False), clock=clock, sleep=clock.sleep)
        self.assertEqual(report["status"], "disabled")
        self.assertEqual(report["samples"], 1)
        self.assertEqual(clock.waits, [])

    def test_masks_are_logged_and_do_not_modify_returned_pixels(self):
        black = png("black")
        image = Image.new("RGB", (4, 4), "black")
        image.putpixel((0, 0), (255, 255, 255))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        changed = stream.getvalue()
        (capture, report), _, _ = self.sample([black, changed, changed], masks=[[0, 0, 1, 1]])
        self.assertEqual(report["status"], "stable")
        self.assertEqual(report["policy"]["masks"], [[0, 0, 1, 1]])
        self.assertEqual(capture.image, changed)
        for masks in ([[0, 0, 5, 1]], [[0, 0, 4, 4]]):
            with self.assertRaises(ValueError):
                self.sample([black], masks=masks)

    def test_dimension_changes_reset_the_stability_streak(self):
        (_, report), emulator, _ = self.sample([png("black"), png("black", (8, 8))])
        self.assertEqual(emulator.calls, 4)
        self.assertEqual(report["differences"][1], None)

    def test_stop_and_focus_loss_interrupt_sampling(self):
        for exception in (StopRequested("stop"), ControlLost("focus lost")):
            clock = FakeClock()
            emulator = SamplingEmulator([png("black")])

            class Control:
                def check(self):
                    if clock.now >= 0.025:
                        raise exception

            with self.assertRaises(type(exception)):
                settled_capture(emulator, capture_settings({"emulator": {"backend": "vbam"}}),
                                Control(), clock=clock, sleep=clock.sleep)
            self.assertEqual(emulator.calls, 0)

    def test_held_keys_prevent_sampling(self):
        emulator = SamplingEmulator([png("black")])
        emulator.held.add("A")
        with self.assertRaisesRegex(RuntimeError, "held"):
            settled_capture(emulator, capture_settings({}))
        self.assertEqual(emulator.calls, 0)

    def test_configuration_rejects_invalid_bounds_and_masks(self):
        for changes in ({"enabled": 1}, {"max_samples": True}, {"interval_ms": 0}, {"max_ms": 6000},
                        {"min_delay_ms": 1000, "max_ms": 10}, {"difference_threshold": float("nan")},
                        {"masks": [[0, 0, 1, False]]}, {"scene_max_ms": {"bad": 100}}, {"unknown": 1}):
            with self.assertRaises(ValueError):
                capture_settings({"capture_settling": changes})


class CaptureIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        config_path = self.root / "config.json"
        write_json(config_path, {"run_dir": "run", "objective": "Test", "rules": [],
            "model": {"name": "fixture"}, "emulator": {"backend": "replay"},
            "capture_settling": {"enabled": True, "min_delay_ms": 0, "interval_ms": 1}})
        self.config = load_config(config_path)
        self.directory = Path(self.config["run_dir"])
        self.images = create_images(self.root / "fixtures")

    def test_final_observation_and_input_timing_are_persisted_and_retrievable(self):
        run(self.config, ReplayEmulator(self.images), DemoAgent(), turns=1)
        context = json.loads(next((self.directory / "turns").glob("*.context.json")).read_text())
        self.assertEqual(context["observation_capture"]["status"], "stable")
        self.assertGreaterEqual(context["observation_capture"]["age_seconds_at_request_build"], 0)
        with contextlib.closing(Memory(self.directory / "memory.sqlite")) as memory:
            self.assertEqual(memory.previous_observation()["id"], 2)
            report = memory.capture_metadata(2)
            self.assertEqual(report["samples"], 3)
            self.assertEqual(report["after_input"]["effective_action"]["buttons"], ["UP"])
            self.assertGreaterEqual(report["after_input"]["observation_age_at_input_seconds"], 0)
            _, records = retrieve_images(memory, self.directory, [2], 2, archive_settings({}))
            self.assertEqual(records[0]["capture_metadata"]["status"], "stable")

    def test_capture_metadata_and_observation_commit_atomically(self):
        self.directory.mkdir()
        with contextlib.closing(Memory(self.directory / "memory.sqlite")) as memory:
            with patch.object(memory.conversation, "append", side_effect=OSError("journal failure")):
                with self.assertRaises(OSError):
                    memory.observe("frame.png", "hash", "test", capture_metadata={"status": "stable"})
            self.assertIsNone(memory.previous_observation())
            self.assertEqual(memory.connection.execute("SELECT COUNT(*) FROM observation_capture").fetchone()[0], 0)

    def test_stop_during_result_sampling_keeps_input_unknown_and_cancels_remaining_moves(self):
        directory = self.directory

        class Emulator(ReplayEmulator):
            def capture(self):
                result = super().capture()
                if self.actions:
                    request_stop(directory)
                return result

        class Agent(DemoAgent):
            def decide(self, request):
                value, trace = super().decide(request)
                value["additional_actions"] = [dict(value["action"])]
                return value, trace

        emulator = Emulator(self.images)
        result = run(self.config, emulator, Agent(), turns=1, control=RunControl(directory, hotkey=False))
        self.assertEqual(result["reason"], "user_stop")
        self.assertEqual(len(emulator.actions), 1)
        with contextlib.closing(Memory(directory / "memory.sqlite")) as memory:
            self.assertEqual(memory.previous_observation()["id"], 1)
            self.assertEqual([step["status"] for step in memory.recent_turns()[0]["steps"]], ["unknown", "cancelled"])
