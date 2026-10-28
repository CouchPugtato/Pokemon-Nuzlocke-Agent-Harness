import json
import tempfile
import unittest
from pathlib import Path

from pokemon_harness.agent.prompts import messages
from pokemon_harness.agent.schemas import validate_decision
from pokemon_harness.config import load_config
from pokemon_harness.demo import DemoAgent, create_images
from pokemon_harness.emulator.interface import ReplayEmulator
from pokemon_harness.memory.database import Memory
from pokemon_harness.runner import run
from pokemon_harness.telemetry.recorder import write_json


def decision(observation_id):
    return DemoAgent().decide(messages({"observation_id": observation_id}, b"png"))[0]


def request(**changes):
    return dict({"query": "starter hp", "scope": "facts", "category": "party",
                 "location": "", "include_history": False, "offset": 0, "limit": 5}, **changes)


def search_decision(observation_id, **changes):
    value = decision(observation_id)
    value.update(memory_updates=[], events=[], memory_search=request(**changes))
    value["action"] = {"kind": "search_memory", "buttons": [], "hold_frames": 1, "settle_frames": 0}
    return value


class MemorySearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        path = self.root / "config.json"
        write_json(path, {"run_dir": "run", "objective": "Synthetic memory test",
                          "model": {"name": "test"}, "emulator": {"backend": "replay"},
                          "memory_limit": 1, "memory_search_rounds": 2})
        self.config = load_config(path)

    def seed(self):
        directory = Path(self.config["run_dir"])
        directory.mkdir()
        memory = Memory(directory / "memory.sqlite")
        memory.initialize(self.config)
        for hp in ("18/24", "24/24"):
            observation_id = memory.observe("unused.png", hp, "replay")
            value = decision(observation_id)
            value["memory_updates"] = [dict(value["memory_updates"][0], category="party",
                key="starter:hp", claim=f"Starter HP was {hp}", location="Oldale Town", status="resolved")]
            turn = memory.plan(observation_id, value)
            memory.status(turn, "completed")
        # A newer active objective crowds the resolved party fact out of automatic recall.
        observation_id = memory.observe("unused.png", "new", "replay")
        value = decision(observation_id)
        value["memory_updates"] = [dict(value["memory_updates"][0], category="objective", key="current")]
        value["events"] = [
            {"key": "faint:starter:1", "type": "faint", "subject": "Starter", "area": "Route 101",
             "details": "Initially believed fainted", "confidence": "high", "corrects_event_key": ""},
            {"key": "correct:faint:1", "type": "correction", "subject": "Starter", "area": "Route 101",
             "details": "Correction: starter was not fainted", "confidence": "high", "corrects_event_key": "faint:starter:1"},
        ]
        turn = memory.plan(observation_id, value)
        memory.status(turn, "completed")
        memory.close()

    def open_memory(self):
        memory = Memory(Path(self.config["run_dir"]) / "memory.sqlite")
        self.addCleanup(memory.close)
        return memory

    def test_search_finds_resolved_fact_and_earlier_revisions_after_reopen(self):
        self.seed()
        memory = self.open_memory()
        latest = memory.search(request(), observation_id=10)
        self.assertEqual(latest["total_matches"], 1)
        self.assertEqual(latest["results"][0]["claim"], "Starter HP was 24/24")
        self.assertTrue(latest["results"][0]["is_latest"])
        self.assertEqual(latest["results"][0]["age_observations"], 8)
        history = memory.search(request(include_history=True, limit=1), observation_id=10)
        self.assertEqual(history["total_matches"], 2)
        self.assertTrue(history["has_more"])
        old = memory.search(request(include_history=True, limit=1, offset=1), observation_id=10)
        self.assertEqual(old["results"][0]["claim"], "Starter HP was 18/24")
        self.assertFalse(old["results"][0]["is_latest"])
        self.assertEqual(old["results"][0]["evidence_observation_id"], 1)

    def test_event_search_preserves_corrections_and_filters_location(self):
        self.seed()
        memory = self.open_memory()
        result = memory.search(request(query="faint:starter:1", scope="events", category="",
                                       location="route 101"), observation_id=10)
        self.assertEqual(result["results"][0]["key"], "faint:starter:1")
        self.assertEqual(result["results"][0]["corrected_by"], ["correct:faint:1"])
        missing = memory.search(request(location="Rustboro"), observation_id=10)
        self.assertEqual(missing["total_matches"], 0)
        self.assertFalse(missing["has_more"])

    def test_search_validation_prevents_mixing_searches_and_mutations(self):
        value = search_decision(1)
        validate_decision(value, 1)
        value["memory_updates"] = decision(1)["memory_updates"]
        with self.assertRaisesRegex(ValueError, "read-only"):
            validate_decision(value, 1)
        value = decision(1)
        value["memory_search"] = request()
        with self.assertRaises(ValueError):
            validate_decision(value, 1)

    def test_search_before_action_has_fresh_observation_and_no_extra_input(self):
        self.seed()
        images = create_images(self.root / "images")
        # Supply an actual last screenshot for the resumed runner.
        memory = self.open_memory()
        memory.observe(str(images[0]), "fixture", "replay")
        contexts = []

        class SearchAgent:
            def decide(self, model_messages):
                context = json.loads(model_messages[1]["content"][0]["text"])
                contexts.append(context)
                if len(contexts) == 1:
                    value = search_decision(context["observation_id"])
                else:
                    value = decision(context["observation_id"])
                return value, {"decision": value}

        emulator = ReplayEmulator(images)
        result = run(self.config, emulator, SearchAgent(), turns=1)
        self.assertEqual(result["completed_actions"], 1)
        self.assertEqual(len(emulator.actions), 1)
        self.assertEqual(len(contexts), 2)
        self.assertGreater(contexts[1]["observation_id"], contexts[0]["observation_id"])
        self.assertFalse(any(f["key"] == "starter:hp" for f in contexts[0]["memory"]))
        found = contexts[1]["memory_search_results"][0]["results"][0]
        self.assertEqual(found["key"], "starter:hp")
        self.assertEqual(found["evidence_observation_id"], 2)
        self.assertEqual(found["age_observations"], contexts[1]["observation_id"] - 2)
        self.assertEqual(contexts[1]["memory_available"]["searches_remaining"], 1)
        self.assertEqual(len(memory.recent_turns()), 4)  # 3 seeded turns + 1 input, no search turn
        self.assertEqual(memory.connection.execute("SELECT COUNT(*) FROM notes WHERE kind='memory_search'").fetchone()[0], 1)

    def test_search_budget_stops_repeated_searches_before_any_input(self):
        images = create_images(self.root / "images")

        class SearchingForever:
            def decide(self, model_messages):
                context = json.loads(model_messages[1]["content"][0]["text"])
                value = search_decision(context["observation_id"])
                return value, {"decision": value}

        emulator = ReplayEmulator(images)
        with self.assertRaisesRegex(ValueError, "budget exhausted"):
            run(self.config, emulator, SearchingForever(), turns=1)
        self.assertEqual(emulator.actions, [])
        memory = self.open_memory()
        self.assertEqual(memory.recent_turns(), [])
        self.assertEqual(memory.connection.execute("SELECT COUNT(*) FROM notes WHERE kind='memory_search'").fetchone()[0], 2)


if __name__ == "__main__":
    unittest.main()
