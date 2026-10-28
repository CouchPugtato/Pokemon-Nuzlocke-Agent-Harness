import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from pokemon_harness.agent.prompts import messages
from pokemon_harness.agent.schemas import validate_decision
from pokemon_harness.config import load_config
from pokemon_harness.demo import DemoAgent, create_images
from pokemon_harness.emulator.interface import ReplayEmulator
from pokemon_harness.memory.database import Memory
from pokemon_harness.runner import run
from pokemon_harness.telemetry.recorder import write_json


def sequence(observation_id):
    value = DemoAgent().decide(messages({"observation_id": observation_id}, b"png"))[0]
    value["action"] = {"kind": "press", "buttons": ["UP"], "hold_frames": 6, "settle_frames": 0}
    value["additional_actions"] = [dict(value["action"], buttons=["RIGHT"]),
                                   dict(value["action"], buttons=["A"])]
    return value


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config_path = self.root / "config.json"
        self.settings = {"run_dir": "run", "objective": "Test action sequences",
                         "model": {"name": "fixture"}, "emulator": {"backend": "replay"},
                         "max_sequence_length": 3, "sequence_delay_ms": 125}
        write_json(self.config_path, self.settings)
        self.config = load_config(self.config_path)
        self.images = create_images(self.root / "images")

    def memory(self):
        memory = Memory(Path(self.config["run_dir"]) / "memory.sqlite")
        self.addCleanup(memory.close)
        return memory

    def agent(self):
        class Agent:
            def __init__(self):
                self.contexts = []
                self.requests = []

            def decide(self, request):
                context = json.loads(request[1]["content"][0]["text"])
                self.contexts.append(context)
                self.requests.append(request)
                value = sequence(context["observation_id"])
                if len(self.contexts) > 1:
                    value["action"] = dict(value["action"], kind="stop", buttons=[])
                    value["additional_actions"] = []
                return value, {"decision": value}

        return Agent()

    def test_sequence_order_delay_screenshots_and_next_model_context(self):
        emulator = ReplayEmulator(self.images)
        agent = self.agent()
        with patch("pokemon_harness.runner.time.sleep") as sleep:
            result = run(self.config, emulator, agent, turns=2)
        self.assertEqual(result, {"completed_actions": 3, "reason": "agent_stop"})
        self.assertEqual([a["buttons"] for a in emulator.actions], [["UP"], ["RIGHT"], ["A"]])
        self.assertEqual(sleep.call_args_list, [call(0.125), call(0.125)])
        self.assertEqual(len(agent.contexts), 2)
        context = agent.contexts[1]
        self.assertEqual(context["max_sequence_length"], 3)
        self.assertEqual(context["sequence_delay_ms"], 125)
        self.assertEqual(len(context["sequence_observations"]), 3)
        self.assertEqual([s["status"] for s in context["recent_turns"][-1]["steps"]], ["completed"] * 3)
        labels = [item.get("text", "") for item in agent.requests[1][1]["content"]]
        self.assertTrue(any("Historical sequence result: move 1" in text for text in labels))
        memory = self.memory()
        steps = memory.recent_turns()[0]["steps"]
        self.assertEqual([step["observation_id"] for step in steps], [2, 3, 4])

    def test_all_moves_are_validated_before_any_input(self):
        for invalid in ("too_long", "opposite_directions"):
            with self.subTest(invalid=invalid):
                value = sequence(1)
                if invalid == "opposite_directions":
                    value["additional_actions"][-1]["buttons"] = ["LEFT", "RIGHT"]
                    cap = 3
                else:
                    cap = 2
                with self.assertRaises(ValueError):
                    validate_decision(value, 1, max_sequence_length=cap)
        emulator = ReplayEmulator(self.images)
        config = dict(self.config, max_sequence_length=2)
        with self.assertRaisesRegex(ValueError, "sequence exceeds"):
            run(config, emulator, self.agent(), turns=1)
        self.assertEqual(emulator.actions, [])

    def test_stop_and_search_cannot_queue_moves(self):
        for kind in ("stop", "search_memory"):
            value = sequence(1)
            value["action"] = dict(value["action"], kind=kind, buttons=[])
            with self.assertRaisesRegex(ValueError, "cannot queue"):
                validate_decision(value, 1)

    def test_failure_keeps_completed_steps_and_cancels_remaining_moves(self):
        class BrokenEmulator(ReplayEmulator):
            def execute(self, action):
                if len(self.actions) == 1:
                    self.actions.append(action)
                    raise TimeoutError("Second move outcome unknown")
                super().execute(action)

        emulator = BrokenEmulator(self.images)
        with patch("pokemon_harness.runner.time.sleep"), self.assertRaises(TimeoutError):
            run(self.config, emulator, self.agent(), turns=1)
        self.assertEqual(len(emulator.actions), 2)
        turn = self.memory().recent_turns()[0]
        self.assertEqual(turn["status"], "unknown")
        self.assertEqual([s["status"] for s in turn["steps"]], ["completed", "unknown", "cancelled"])

    def test_interrupt_during_delay_never_starts_the_next_move(self):
        emulator = ReplayEmulator(self.images)
        with patch("pokemon_harness.runner.time.sleep", side_effect=KeyboardInterrupt), \
                self.assertRaises(KeyboardInterrupt):
            run(self.config, emulator, self.agent(), turns=1)
        self.assertEqual(len(emulator.actions), 1)
        self.assertEqual([s["status"] for s in self.memory().recent_turns()[0]["steps"]],
                         ["completed", "cancelled", "cancelled"])

    def test_recovery_preserves_completed_steps_without_replaying_a_plan(self):
        directory = Path(self.config["run_dir"])
        directory.mkdir()
        memory = Memory(directory / "memory.sqlite")
        memory.initialize(self.config)
        observation_id = memory.observe("fixture.png", "hash", "replay")
        turn_id = memory.plan(observation_id, sequence(observation_id))
        memory.status(turn_id, "executing")
        memory.step_status(turn_id, 1, "completed", observation_id)
        memory.step_status(turn_id, 2, "executing")
        memory.close()
        memory = self.memory()
        self.assertEqual(memory.recover(), [{"id": turn_id, "status": "executing"}])
        self.assertEqual([s["status"] for s in memory.recent_turns()[0]["steps"]],
                         ["completed", "unknown", "cancelled"])
        self.assertEqual(memory.recover(), [])

    def test_sequence_config_rejects_invalid_limits_and_delays(self):
        for changes in ({"max_sequence_length": 0}, {"max_sequence_length": 21},
                        {"sequence_delay_ms": -1}, {"sequence_delay_ms": True}):
            write_json(self.config_path, dict(self.settings, **changes))
            with self.assertRaises(ValueError):
                load_config(self.config_path)


if __name__ == "__main__":
    unittest.main()
