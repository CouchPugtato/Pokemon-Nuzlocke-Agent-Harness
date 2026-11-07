import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from jsonschema import ValidationError

from pokemon_harness.agent.context_budget import fit_request
from pokemon_harness.agent.prompts import messages
from pokemon_harness.agent.schemas import MEMORY_SEARCH_SCHEMA
from jsonschema import Draft202012Validator
from pokemon_harness.config import load_config
from pokemon_harness.demo import DemoAgent, create_images
from pokemon_harness.emulator.interface import ReplayEmulator
from pokemon_harness.memory.archive import archive_settings, retrieve_images
from pokemon_harness.memory.database import Memory
from pokemon_harness.runner import run
from pokemon_harness.telemetry.recorder import write_json


def query(**changes):
    return dict({"query": "", "scope": "turns", "category": "", "location": "",
                 "include_history": False, "offset": 0, "limit": 10}, **changes)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.memory = Memory(self.root / "memory.sqlite")
        self.addCleanup(self.memory.close)
        self.memory.initialize({"objective": "Test", "rules": []})
        self.images = create_images(self.root / "screenshots")
        self.ids = [self.memory.observe(path.relative_to(self.root), hashlib.sha256(path.read_bytes()).hexdigest(), "test")
                    for path in self.images]
        self.settings = archive_settings({})

    def fetch(self, ids, **changes):
        return retrieve_images(self.memory, self.root, ids, 3, dict(self.settings, **changes))

    def test_verified_images_keep_source_identity_without_mutating_memory(self):
        before = self.memory.export()
        frames, records = self.fetch([1, 3])
        self.assertEqual([frame["observation_id"] for frame in frames], [1, 3])
        self.assertEqual(frames[0]["image"], self.images[0].read_bytes())
        self.assertEqual([record["read_status"] for record in records], ["verified", "verified"])
        self.assertEqual(self.memory.export(), before)

    def test_missing_and_oversized_images_are_explicit_without_replacement(self):
        self.images[0].unlink()
        frames, records = self.fetch([1, 2], max_bytes=1)
        self.assertEqual(frames, [])
        self.assertEqual([record["read_status"] for record in records], ["missing", "byte_limit"])

    def test_hash_mismatch_stops_retrieval(self):
        self.images[0].write_bytes(self.images[1].read_bytes())
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.fetch([1])

    def test_invalid_and_future_ids_cannot_select_files(self):
        for ids in ([True], [0], [-1], [4], [999]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.fetch(ids)
        with self.assertRaisesRegex(ValueError, "belong"):
            retrieve_images(self.memory, self.root, [4], 10, self.settings)

    def test_paths_must_resolve_within_the_run(self):
        for path in ("../outside.png", str(self.root.parent / "outside.png")):
            with self.memory.connection:
                self.memory.connection.execute("UPDATE observations SET image_path=? WHERE id=1", (path,))
            with self.assertRaisesRegex(ValueError, "escapes"):
                self.fetch([1])

    def test_dimensions_are_checked_before_image_decode(self):
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.fetch([1], max_image_pixels=1)

    def test_corrupt_png_with_matching_hash_is_rejected(self):
        self.images[0].write_bytes(b"not a png")
        with self.memory.connection:
            self.memory.connection.execute("UPDATE observations SET image_hash=? WHERE id=1",
                                           (hashlib.sha256(b"not a png").hexdigest(),))
        with self.assertRaises(OSError):
            self.fetch([1])

    def test_cumulative_limits_and_deduplication_apply_across_searches(self):
        frames, _ = self.fetch([1])
        again, _ = retrieve_images(self.memory, self.root, [1], 3, self.settings, frames)
        self.assertEqual(len(again), 1)
        with self.assertRaisesRegex(ValueError, "image limit"):
            retrieve_images(self.memory, self.root, [2], 3, dict(self.settings, max_images=1), frames)
        limited, records = retrieve_images(self.memory, self.root, [2], 3,
            dict(self.settings, max_bytes=len(frames[0]["image"])), frames)
        self.assertEqual(len(limited), 1)
        self.assertEqual(records[0]["read_status"], "byte_limit")

    def test_search_contract_supports_old_requests_and_bounds_explicit_ids(self):
        validator = Draft202012Validator(MEMORY_SEARCH_SCHEMA)
        validator.validate(query())
        validator.validate(query(observation_ids=[1, 2, 3, 4]))
        for ids in ([1, 1], [0], [True], list(range(1, 6)), ["../path"]):
            with self.assertRaises(ValidationError):
                validator.validate(query(observation_ids=ids))
        for settings in ({"max_images": 5}, {"max_bytes": True}, {"url": "bad"}):
            with self.assertRaises(ValueError):
                archive_settings({"archive_retrieval": settings})

    def test_turn_search_returns_sources_and_actual_steps_without_raw_rationales(self):
        value = DemoAgent().decide(messages({"observation_id": 1}, self.images[0].read_bytes()))[0]
        value["decision_reason"] = "private speculative narrative"
        value["expected_result"] = "unobserved prediction"
        turn = self.memory.plan(1, value)
        effective = dict(value["action"], buttons=["B"])
        self.memory.step_status(turn, 1, "completed", 2, {"effective_action": effective})
        self.memory.status(turn, "completed")
        result = self.memory.search(query(query=f"turn:{turn}"), 3)
        self.assertEqual(result["total_matches"], 1)
        self.assertEqual(result["results"][0]["detail"]["steps"][0]["observation_id"], 2)
        self.assertEqual(result["results"][0]["detail"]["steps"][0]["action"], effective)
        self.assertNotIn("private speculative narrative", json.dumps(result))
        self.assertNotIn("unobserved prediction", json.dumps(result))
        self.assertEqual(self.memory.search(query(query="doorway"), 3)["total_matches"], 1)
        self.assertEqual(self.memory.search(query(query="doorway", offset=1, limit=1), 3)["results"], [])

    def test_turn_recall_withholds_interpretations_after_a_source_dispute(self):
        value = DemoAgent().decide(messages({"observation_id": 1}, self.images[0].read_bytes()))[0]
        turn = self.memory.plan(1, value)
        self.memory.status(turn, "stopped")
        with self.memory.connection:
            self.memory.contradictions.apply([{
                "operation": "open", "key": "doorway conflict", "expected_revision": 0,
                "reason": "Fresh evidence conflicts", "evidence_observation_ids": [3],
                "fact_ids": [1], "event_keys": [], "replacement_facts": [], "correction_event_keys": []
            }], 3)
        record = self.memory.search(query(query=f"turn:{turn}"), 3)["results"][0]
        self.assertEqual(record["memory_validity"], "invalidated")
        self.assertIsNone(record["strategy_target"])
        self.assertEqual(record["detail"]["decision"]["scene"], "unknown")
        frames, _ = self.fetch([1])
        self.assertEqual(len(frames), 1)
        self.assertEqual(self.memory.contradictions.context()["open_count"], 1)

    def test_archive_pixels_are_labeled_and_budget_omissions_are_explicit(self):
        frames, _ = self.fetch([1])
        context = {"observation_id": 3, "requested_archive_observation_ids": [1]}
        model = {"name": "fixture", "context_length": 32768, "max_tokens": 8192}
        selected, request, _ = fit_request(context, model, self.images[2].read_bytes(), outcome_images=frames)
        self.assertEqual(selected["archive_visual_evidence_ids"], [1])
        self.assertIn("Archived historical evidence observation 1", json.dumps(request))
        large = Image.new("RGB", (2, 2))
        stream = io.BytesIO()
        large.save(stream, format="PNG")
        model["context_budget"] = {"image_tokens": 5000}
        selected, _, report = fit_request(context, model, stream.getvalue(), outcome_images=frames)
        self.assertEqual(selected["archive_visual_evidence_ids"], [])
        self.assertEqual(report["omitted"]["archive_images"], 1)
        self.assertEqual(selected["outcome_visual_evidence_ids"], [3])


class ArchiveIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        path = self.root / "config.json"
        write_json(path, {"run_dir": "run", "objective": "Test", "rules": [],
                          "model": {"name": "fixture"}, "emulator": {"backend": "replay"}})
        self.config = load_config(path)
        self.images = create_images(self.root / "fixtures")

    def agent(self, invalid=False, source_id=None):
        requests = []

        class Agent:
            def decide(self, request):
                context = json.loads(request[1]["content"][0]["text"])
                requests.append((context, request))
                value, trace = DemoAgent().decide(request)
                if len(requests) == 1:
                    value.update(memory_updates=[], events=[], outcome_reviews=[],
                                 memory_search=query(observation_ids=[999 if invalid else source_id or context["observation_id"]]))
                    value["action"] = {"kind": "search_memory", "buttons": [], "hold_frames": 1, "settle_frames": 0}
                return value, trace

        return Agent(), requests

    def test_search_images_arrive_with_fresh_current_observation_and_no_extra_input(self):
        agent, requests = self.agent()
        emulator = ReplayEmulator(self.images)
        run(self.config, emulator, agent, turns=1)
        self.assertEqual(len(requests), 2)
        self.assertEqual(len(emulator.actions), 1)
        first, second = requests[0][0], requests[1][0]
        self.assertGreater(second["observation_id"], first["observation_id"])
        self.assertEqual(second["archive_visual_evidence_ids"], [first["observation_id"]])
        self.assertEqual(second["memory_search_results"][0]["archive_images"][0]["read_status"], "verified")
        with contextlib.closing(Memory(Path(self.config["run_dir"]) / "memory.sqlite")) as memory:
            archived = memory.conversation.request(2)
            self.assertNotIn("base64,", json.dumps(archived))
            search = next(row for row in memory.conversation.history(100) if row["kind"] == "search_completed")
            self.assertEqual(search["payload"]["archive_images"][0]["observation_id"], first["observation_id"])

    def test_invalid_archive_request_stops_before_controller_input(self):
        agent, requests = self.agent(invalid=True)
        emulator = ReplayEmulator(self.images)
        with self.assertRaisesRegex(ValueError, "future"):
            run(self.config, emulator, agent, turns=1)
        self.assertEqual(len(requests), 1)
        self.assertEqual(emulator.actions, [])

    def test_resume_can_request_original_pixels_from_an_earlier_session(self):
        run(self.config, ReplayEmulator(self.images), DemoAgent(), turns=1)
        agent, requests = self.agent(source_id=1)
        emulator = ReplayEmulator(self.images)
        run(self.config, emulator, agent, turns=1)
        self.assertGreater(requests[0][0]["observation_id"], 1)
        self.assertEqual(requests[1][0]["archive_visual_evidence_ids"], [1])
        record = requests[1][0]["memory_search_results"][0]["archive_images"][0]
        self.assertGreater(record["age_observations_at_retrieval"], 0)
        self.assertEqual(len(emulator.actions), 1)
