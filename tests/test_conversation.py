import contextlib
import copy
import hashlib
import io
import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from pokemon_harness.agent.prompts import messages
from pokemon_harness.agent.schemas import validate_decision
from pokemon_harness.cli import main
from pokemon_harness.config import load_config
from pokemon_harness.control.session import RunControl, request_stop
from pokemon_harness.demo import DemoAgent, create_images
from pokemon_harness.emulator.interface import ReplayEmulator
from pokemon_harness.memory.database import Memory
from pokemon_harness.runner import run
from pokemon_harness.telemetry.backup import create_backup
from pokemon_harness.telemetry.recorder import write_json
from pokemon_harness.telemetry.restore import restore_backup


class ConversationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.memory = Memory(self.root / "memory.sqlite")
        self.addCleanup(self.memory.close)
        self.memory.initialize({"objective": "Test", "rules": ["No items"]})
        self.png = create_images(self.root / "images")[0].read_bytes()

    def begin(self):
        observation = self.memory.observe("image.png", hashlib.sha256(self.png).hexdigest(), "test")
        request = messages({"observation_id": observation}, self.png)
        request_id = self.memory.begin_request(request, observation, {"approximate": True}, "fixture")
        value = DemoAgent().decide(request)[0]
        value["action"].update(kind="stop", buttons=[])
        return request_id, observation, value

    def validated(self):
        request_id, observation, value = self.begin()
        self.memory.record_response(request_id, value, "turns/response.model.json")
        validate_decision(value, observation)
        self.memory.validate_response(request_id)
        return request_id, observation, value

    def completed(self):
        request_id, observation, value = self.validated()
        turn = self.memory.plan(observation, value, request_id)
        self.memory.status(turn, "stopped")
        return request_id, observation, turn

    def test_archived_request_uses_foreign_key_image_and_deduplicated_prompt_references(self):
        request_id, _, _ = self.begin()
        archived = self.memory.conversation.request(request_id)
        self.assertNotIn("base64,", json.dumps(archived))
        source = archived["messages"][0]["content_reference"]
        self.assertIn("You play Pokemon", archived["prompts"][source])
        block = archived["messages"][1]["content"][-1]
        self.assertEqual(block["type"], "observation_image")
        self.assertEqual(block["observation_id"], 1)
        self.begin()
        self.assertEqual(self.memory.connection.execute("SELECT COUNT(*) FROM conversation_prompts").fetchone()[0], 1)
        self.assertEqual(self.memory.connection.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_unmatched_image_rolls_back_the_entire_request(self):
        observation = self.memory.observe("image.png", hashlib.sha256(self.png).hexdigest(), "test")
        with self.assertRaisesRegex(ValueError, "matching observation"):
            self.memory.begin_request(messages({"observation_id": observation}, b"unknown"), observation, {}, "fixture")
        self.assertEqual(self.memory.connection.execute("SELECT COUNT(*) FROM conversation_requests").fetchone()[0], 0)
        self.assertEqual(self.memory.connection.execute("SELECT COUNT(*) FROM conversation_prompts").fetchone()[0], 0)

    def test_only_the_matching_validated_response_can_commit_a_plan(self):
        request_id, observation, value = self.validated()
        changed = copy.deepcopy(value)
        changed["expected_result"] = "A different proposal"
        with self.assertRaisesRegex(ValueError, "differs from"):
            self.memory.plan(observation, changed, request_id)
        self.assertEqual(self.memory.recent_turns(), [])
        self.assertEqual(self.memory.facts(), [])
        self.assertEqual(self.memory.conversation.request(request_id)["status"], "validated")
        turn = self.memory.plan(observation, value, request_id)
        self.memory.status(turn, "stopped")
        self.assertEqual(self.memory.conversation.request(request_id)["status"], "stopped")

    def test_checkpoint_refuses_pending_requests_and_recovery_never_replays_them(self):
        request_id, _, _ = self.begin()
        with self.assertRaisesRegex(ValueError, "unfinished conversation"):
            self.memory.save_checkpoint()
        self.memory.recover()
        self.assertEqual(self.memory.conversation.request(request_id)["status"], "interrupted")
        self.assertEqual(self.memory.recent_turns(), [])
        self.memory.save_checkpoint()
        with self.assertRaisesRegex(ValueError, "no longer accepting"):
            self.memory.record_response(request_id, {}, "late.model.json")

    def test_raw_rationale_is_archived_but_excluded_from_model_recall(self):
        request_id, observation, value = self.begin()
        value["decision_reason"] = "Private speculative narrative"
        value["expected_result"] = "Unobserved future result"
        self.memory.record_response(request_id, value, "trace.json")
        self.memory.validate_response(request_id)
        turn = self.memory.plan(observation, value, request_id)
        self.memory.status(turn, "stopped")
        self.assertIn("Private speculative narrative", json.dumps(self.memory.conversation.history()))
        recall = json.dumps(self.memory.conversation.context())
        self.assertNotIn("Private speculative narrative", recall)
        self.assertNotIn("Unobserved future result", recall)

    def test_checkpoint_boundary_and_snapshot_publish_atomically(self):
        self.completed()
        first = self.memory.save_checkpoint()
        before = self.memory.conversation.active_checkpoint()
        self.memory.observe("image.png", hashlib.sha256(self.png).hexdigest(), "new")
        original = self.memory.conversation.checkpoint

        def fail(snapshot):
            original(snapshot)
            raise OSError("Interrupted checkpoint commit")

        with patch.object(self.memory.conversation, "checkpoint", side_effect=fail):
            with self.assertRaisesRegex(OSError, "Interrupted checkpoint"):
                self.memory.save_checkpoint()
        self.assertEqual(self.memory.conversation.active_checkpoint(), before)
        self.assertEqual(self.memory.connection.execute("SELECT COUNT(*) FROM conversation_checkpoints").fetchone()[0], 1)
        self.assertEqual(self.memory.conversation.context()["checkpoint"]["id"], first)
        self.assertGreater(self.memory.conversation.context()["recent"][-1]["id"], before["boundary_id"])

    def test_checkpoint_checksum_and_source_references_are_verified(self):
        self.completed()
        self.memory.save_checkpoint()
        checkpoint = self.memory.conversation.active_checkpoint()
        with self.memory.connection:
            self.memory.connection.execute("UPDATE conversation_checkpoints SET payload='{}'")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.memory.conversation.active_checkpoint()
        payload = copy.deepcopy(checkpoint["payload"])
        payload["source_watermarks"]["facts"] = 999
        encoded = json.dumps(payload)
        with self.memory.connection:
            self.memory.connection.execute("UPDATE conversation_checkpoints SET payload=?,payload_hash=?",
                                           (encoded, hashlib.sha256(encoded.encode()).hexdigest()))
        with self.assertRaisesRegex(ValueError, "unavailable source"):
            self.memory.save_checkpoint()

    def test_checkpoint_restore_context_is_bounded_and_preserves_suffix_order(self):
        self.completed()
        self.memory.save_checkpoint()
        for index in range(30):
            self.memory.observe("image.png", hashlib.sha256(self.png).hexdigest(), str(index))
        context = self.memory.conversation.context()
        ids = [row["id"] for row in context["recent"]]
        self.assertEqual(len(ids), 12)
        self.assertEqual(ids, sorted(set(ids)))
        with contextlib.closing(Memory(self.root / "memory.sqlite")) as reopened:
            self.assertEqual(reopened.conversation.context(), context)

    def test_readonly_search_journal_finishes_without_a_controller_turn(self):
        request_id, observation, value = self.begin()
        value.update(memory_updates=[], events=[])
        value["action"].update(kind="search_memory", buttons=[])
        value["memory_search"] = {"query": "party", "scope": "facts", "category": "", "location": "",
                                  "include_history": False, "offset": 0, "limit": 2}
        self.memory.record_response(request_id, value, "trace.json")
        validate_decision(value, observation)
        self.memory.validate_response(request_id)
        result = self.memory.search(value["memory_search"], observation)
        self.memory.record_search(request_id, observation, result)
        self.assertEqual(self.memory.conversation.request(request_id)["status"], "search_completed")
        self.assertEqual(self.memory.recent_turns(), [])
        self.memory.save_checkpoint()


class ConversationIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        config_path = self.root / "config.json"
        write_json(config_path, {"run_dir": "run", "objective": "Test", "model": {"name": "fixture"},
                                 "emulator": {"backend": "replay"}, "sequence_delay_ms": 0})
        self.config = load_config(config_path)
        self.directory = Path(self.config["run_dir"])
        self.images = create_images(self.root / "fixtures")

    def test_request_is_committed_before_generation_and_resume_has_checkpoint_plus_fresh_evidence(self):
        requests = []
        database = self.directory / "memory.sqlite"

        class Agent:
            def decide(self, request):
                requests.append(json.loads(request[1]["content"][0]["text"]))
                with contextlib.closing(sqlite3.connect(database)) as connection:
                    self_status = connection.execute("SELECT status FROM conversation_requests ORDER BY id DESC LIMIT 1").fetchone()[0]
                    if self_status != "requested":
                        raise AssertionError("request must be durable before generation")
                return DemoAgent().decide(request)

        run(self.config, ReplayEmulator(self.images), Agent(), turns=1)
        with contextlib.closing(Memory(database)) as memory:
            first = memory.conversation.active_checkpoint()
        emulator = ReplayEmulator(self.images)
        run(self.config, emulator, Agent(), turns=1)
        self.assertEqual(len(emulator.actions), 1)
        self.assertGreater(requests[1]["observation_id"], first["observation_id"])
        self.assertEqual(requests[1]["conversation"]["checkpoint"]["id"], first["id"])
        self.assertTrue(any(row["kind"] == "input_step" for row in requests[1]["conversation"]["recent"]))

    def test_cancelled_late_response_never_writes_journal_or_enters_checkpoint(self):
        release, finished = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        directory = self.directory

        class Agent:
            def decide(self, request):
                request_stop(directory)
                release.wait(5)
                result = DemoAgent().decide(request)
                finished.set()
                return result

        emulator = ReplayEmulator(self.images)
        result = run(self.config, emulator, Agent(), turns=1, control=RunControl(directory, hotkey=False))
        self.assertEqual(result["reason"], "user_stop")
        with contextlib.closing(Memory(directory / "memory.sqlite")) as memory:
            before = memory.conversation.history()
            checkpoint = memory.conversation.active_checkpoint()
            self.assertEqual(memory.conversation.request(1)["status"], "interrupted")
        release.set()
        self.assertTrue(finished.wait(2))
        with contextlib.closing(Memory(directory / "memory.sqlite")) as memory:
            self.assertEqual(memory.conversation.history(), before)
            self.assertEqual(memory.conversation.active_checkpoint(), checkpoint)
            self.assertEqual(memory.recent_turns(), [])
        self.assertEqual(emulator.actions, [])

    def test_restore_retains_journal_and_checkpoint_without_bypassing_game_review(self):
        run(self.config, ReplayEmulator(self.images), DemoAgent(), turns=2)
        with contextlib.closing(Memory(self.directory / "memory.sqlite")) as memory:
            history = memory.conversation.history(100)
            checkpoint = memory.conversation.active_checkpoint()
        backup = create_backup(self.directory, self.root / "backups")
        target = self.root / "restored"
        restore_backup(backup, target)
        with contextlib.closing(Memory(target / "memory.sqlite")) as memory:
            self.assertEqual(memory.conversation.active_checkpoint(), checkpoint)
            self.assertEqual(memory.conversation.history(100), history)
            self.assertTrue(memory.escalations.review(memory.events())["blocked_cases"])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["conversation", str(target), "--history", "--request", "1", "--checkpoint"]), 0)
        self.assertEqual(json.loads(output.getvalue())["checkpoint"], checkpoint)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["checkpoint", str(target)]), 0)
        with contextlib.closing(Memory(target / "memory.sqlite")) as memory:
            self.assertGreater(memory.conversation.active_checkpoint()["id"], checkpoint["id"])
            self.assertTrue(memory.escalations.review(memory.events())["blocked_cases"])
