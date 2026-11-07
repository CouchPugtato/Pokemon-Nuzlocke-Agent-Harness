import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import ValidationError

from pokemon_harness.agent.prompts import messages
from pokemon_harness.agent.schemas import validate_decision
from pokemon_harness.cli import main
from pokemon_harness.config import load_config
from pokemon_harness.demo import DemoAgent, create_images, demo_outcome_reviews
from pokemon_harness.emulator.interface import ReplayEmulator
from pokemon_harness.memory.database import Memory
from pokemon_harness.memory.retrieval import recent_turn_context
from pokemon_harness.runner import run
from pokemon_harness.telemetry.backup import create_backup
from pokemon_harness.telemetry.recorder import write_json
from pokemon_harness.telemetry.restore import restore_backup


def decision(observation_id):
    return DemoAgent().decide(messages({"observation_id": observation_id}, b"png"))[0]


class OutcomeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.memory = Memory(self.root / "memory.sqlite")
        self.addCleanup(self.memory.close)
        self.memory.initialize({"objective": "Beat first gym", "rules": []})
        self.memory.observe("before.png", "before", "test")

    def attempted(self, partial=False, unknown=False):
        value = decision(1)
        if partial:
            value["additional_actions"] = [dict(value["action"]), dict(value["action"])]
        turn = self.memory.plan(1, value)
        self.memory.status(turn, "executing")
        self.memory.step_status(turn, 1, "executing")
        if not unknown:
            result = self.memory.observe("after.png", "after", "test")
            self.memory.step_status(turn, 1, "completed", result)
        if partial or unknown:
            if partial:
                self.memory.step_status(turn, 2, "executing")
            self.memory.status(turn, "unknown", "Interrupted")
        else:
            self.memory.status(turn, "completed")
        return turn

    def review(self):
        observation = self.memory.observe("review.png", "review", "test")
        reviews = demo_outcome_reviews({"observation_id": observation, "outcome_review": self.memory.outcomes.context()})
        return observation, reviews

    def test_startup_and_unexecuted_plans_have_explicit_empty_review_context(self):
        self.assertEqual(self.memory.outcomes.context(), {"pending_count": 0, "next_plan": None})
        turn = self.memory.plan(1, decision(1))
        self.memory.status(turn, "blocked")
        self.assertEqual(self.memory.outcomes.pending_count(), 0)

    def test_input_completion_requires_review_before_more_input(self):
        turn = self.attempted()
        self.assertEqual(self.memory.outcomes.pending_count(), 1)
        observation, reviews = self.review()
        with self.assertRaisesRegex(ValueError, "Review pending"):
            self.memory.plan(observation, decision(observation))
        value = decision(observation)
        reviews[0]["steps"][0].update(assessment="no_progress_observed", explanation="The menu stayed open")
        value["outcome_reviews"] = reviews
        self.memory.plan(observation, value)
        self.assertEqual(self.memory.outcomes.pending_count(), 0)
        self.assertEqual(self.memory.outcomes.latest(turn)["steps"][0]["assessment"], "no_progress_observed")
        self.assertEqual(self.memory.recent_turns()[0]["status"], "completed")

    def test_unknown_and_partial_sequence_requires_exactly_attempted_steps(self):
        turn = self.attempted(partial=True)
        observation, reviews = self.review()
        self.assertEqual([s["input_status"] for s in self.memory.outcomes.context()["next_plan"]["steps"]],
                         ["completed", "unknown", "cancelled"])
        self.assertEqual([s["step_index"] for s in reviews[0]["steps"]], [1, 2])
        self.assertEqual(reviews[0]["steps"][1]["before_observation_id"], 2)
        for invalid in (reviews[0]["steps"][:1], reviews[0]["steps"] * 2,
                        [*reviews[0]["steps"], dict(reviews[0]["steps"][1], step_index=3)]):
            changed = copy.deepcopy(reviews)
            changed[0]["steps"] = invalid
            with self.assertRaisesRegex(ValueError, "every attempted step"):
                self.memory.review_outcomes(observation, changed)
        self.memory.review_outcomes(observation, reviews)
        self.assertEqual(self.memory.outcomes.pending_count(), 0)
        self.assertEqual(self.memory.recent_turns()[0]["steps"][2]["status"], "cancelled")
        self.assertEqual(self.memory.outcomes.latest(turn)["steps"][1]["assessment"], "uncertain")

    def test_missing_result_capture_cannot_claim_progress_or_failure(self):
        self.attempted(unknown=True)
        observation, reviews = self.review()
        for assessment in ("progress_observed", "no_progress_observed"):
            invalid = copy.deepcopy(reviews)
            invalid[0]["steps"][0]["assessment"] = assessment
            with self.assertRaisesRegex(ValueError, "Missing step capture"):
                self.memory.review_outcomes(observation, invalid)
        self.memory.review_outcomes(observation, reviews)
        self.assertEqual(self.memory.recent_turns()[0]["status"], "unknown")

    def test_review_rejects_wrong_before_future_and_unrelated_result(self):
        self.attempted()
        observation, reviews = self.review()
        for key, value in (("before_observation_id", 2), ("after_observation_id", 999),
                           ("after_observation_id", observation), ("after_observation_id", 1)):
            invalid = copy.deepcopy(reviews)
            invalid[0]["steps"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.memory.review_outcomes(observation, invalid)
        invalid = copy.deepcopy(reviews)
        invalid[0]["turn_id"] = 999
        with self.assertRaisesRegex(ValueError, "missing turn"):
            self.memory.review_outcomes(observation, invalid)
        self.assertEqual(self.memory.outcomes.history(), [])

    def test_uncaptured_step_review_requires_a_fresh_reconciliation_observation(self):
        self.attempted(unknown=True)
        observation, reviews = self.review()
        fresh = self.memory.observe("fresh.png", "fresh", "test")
        with self.assertRaisesRegex(ValueError, "current observation"):
            self.memory.review_outcomes(fresh, reviews)
        reviews[0]["steps"][0]["after_observation_id"] = fresh
        self.memory.review_outcomes(fresh, reviews)

    def test_reviews_are_append_only_and_stale_revision_is_rejected(self):
        turn = self.attempted()
        observation, reviews = self.review()
        self.memory.review_outcomes(observation, reviews)
        with self.assertRaisesRegex(ValueError, "stale revision"):
            self.memory.review_outcomes(observation, reviews)
        revised = copy.deepcopy(reviews)
        revised[0]["expected_revision"] = 1
        revised[0]["steps"][0].update(assessment="progress_observed", explanation="The next menu became visible")
        self.memory.review_outcomes(observation, revised)
        self.assertEqual(self.memory.outcomes.history()[-1]["steps"][0]["assessment"], "uncertain")
        self.assertEqual(self.memory.outcomes.latest(turn)["revision"], 2)
        with contextlib.closing(Memory(self.root / "memory.sqlite")) as reopened:
            self.assertEqual(reopened.outcomes.latest(turn), self.memory.outcomes.latest(turn))

    def test_invalid_decision_rolls_back_review_and_new_plan(self):
        self.attempted()
        observation, reviews = self.review()
        value = decision(observation)
        value["outcome_reviews"] = reviews
        value["working_state_update"] = {"expected_revision": 9, "location": None, "uncertainties": [],
                                         "next_check": "", "subgoal_change": None}
        with self.assertRaisesRegex(ValueError, "stale revision"):
            self.memory.plan(observation, value)
        self.assertEqual(self.memory.outcomes.history(), [])
        self.assertEqual(self.memory.outcomes.pending_count(), 1)
        self.assertEqual(len(self.memory.recent_turns()), 1)
        self.assertEqual(len(self.memory.facts(True)), 1)

    def test_stop_can_leave_unreviewed_outcome_without_claiming_success(self):
        self.attempted()
        observation, _ = self.review()
        value = decision(observation)
        value["action"].update(kind="stop", buttons=[])
        turn = self.memory.plan(observation, value)
        self.memory.status(turn, "stopped")
        self.assertEqual(self.memory.outcomes.pending_count(), 1)

    def test_model_contract_keeps_search_readonly_and_review_only_bounded(self):
        self.attempted()
        observation, reviews = self.review()
        value = decision(observation)
        value.update(outcome_reviews=reviews, memory_updates=[], events=[])
        value["action"].update(kind="review_outcomes", buttons=[])
        validate_decision(value, observation)
        invalid = copy.deepcopy(value)
        invalid["outcome_reviews"] *= 2
        with self.assertRaises(ValidationError):
            validate_decision(invalid, observation)
        invalid = copy.deepcopy(value)
        invalid["memory_updates"] = decision(observation)["memory_updates"]
        with self.assertRaisesRegex(ValueError, "only writes"):
            validate_decision(invalid, observation)
        value["action"]["kind"] = "search_memory"
        value["memory_search"] = {"query": "", "scope": "all", "category": "", "location": "",
                                  "include_history": False, "offset": 0, "limit": 1}
        with self.assertRaisesRegex(ValueError, "read-only"):
            validate_decision(value, observation)

    def test_review_retains_original_subgoal_without_changing_ledger_or_cases(self):
        value = decision(1)
        definition = {"description": "Inspect party", "objective_relation": "Gym readiness",
                      "completion_condition": "Party menu visible", "abandonment_condition": "Unsafe battle"}
        value["working_state_update"] = {"expected_revision": 0, "location": None, "uncertainties": [],
            "next_check": "Party menu", "subgoal_change": {"operation": "start", "id": "party:inspect",
            "definition": definition, "reason": "Check readiness", "evidence_observation_ids": [1]}}
        turn = self.memory.plan(1, value)
        result = self.memory.observe("after.png", "after", "test")
        self.memory.step_status(turn, 1, "completed", result)
        self.memory.status(turn, "completed")
        plan = self.memory.outcomes.context()["next_plan"]
        self.assertEqual(plan["subgoal_at_input"]["definition"], definition)
        self.assertEqual(plan["working_revision_at_input"], 1)
        case = self.memory.escalations.open("uncertain", "Uncertain party history", [result])
        observation, reviews = self.review()
        self.memory.review_outcomes(observation, reviews)
        self.assertEqual(self.memory.events(), [])
        self.assertEqual(self.memory.escalations.review([])["blocked_cases"][0]["id"], case)
        self.assertEqual(recent_turn_context(self.memory.recent_turns())[0]["outcome_review"]["revision"], 1)


class OutcomeIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        path = self.root / "config.json"
        write_json(path, {"run_dir": "run", "objective": "Test", "model": {"name": "test"},
                          "emulator": {"backend": "replay"}, "outcome_review_rounds": 1})
        self.config = load_config(path)
        self.directory = Path(self.config["run_dir"])
        self.images = create_images(self.root / "fixtures")

    def backlog(self):
        run(self.config, ReplayEmulator(self.images), DemoAgent(), turns=2)
        with contextlib.closing(Memory(self.directory / "memory.sqlite")) as memory:
            with memory.connection:
                memory.connection.execute("DELETE FROM outcome_reviews")
            self.assertEqual(memory.outcomes.pending_count(), 2)

    def test_missing_review_stops_before_input_and_retains_pending_plan_on_resume(self):
        run(self.config, ReplayEmulator(self.images), DemoAgent(), turns=1)

        class MissingReview:
            def decide(self, request):
                context = json.loads(request[1]["content"][0]["text"])
                value = decision(context["observation_id"])
                return value, {"decision": value}

        emulator = ReplayEmulator(self.images)
        with self.assertRaisesRegex(ValueError, "Review pending"):
            run(self.config, emulator, MissingReview(), turns=1)
        self.assertEqual(emulator.actions, [])
        with contextlib.closing(Memory(self.directory / "memory.sqlite")) as memory:
            self.assertEqual(memory.outcomes.pending_count(), 1)

    def test_backlog_batches_do_not_send_input_and_budget_is_persistent_in_effect(self):
        self.backlog()
        requests = []

        class Reviewer:
            def decide(self, request):
                requests.append(request)
                value, trace = DemoAgent().decide(request)
                value.update(memory_updates=[], events=[])
                value["action"].update(kind="review_outcomes", buttons=[])
                return value, trace

        emulator = ReplayEmulator(self.images)
        with self.assertRaisesRegex(ValueError, "review budget exhausted"):
            run(self.config, emulator, Reviewer(), turns=1)
        self.assertEqual(emulator.actions, [])
        context = json.loads(requests[1][1]["content"][0]["text"])
        self.assertEqual(context["outcome_review"]["pending_count"], 1)
        self.assertEqual(context["outcome_review"]["review_batches_remaining"], 0)
        labels = [item.get("text", "") for item in requests[0][1]["content"]]
        self.assertTrue(any("Outcome evidence observation 1" in label for label in labels))
        result = run(self.config, ReplayEmulator(self.images), DemoAgent(), turns=1)
        self.assertEqual(result["completed_actions"], 1)
        with contextlib.closing(Memory(self.directory / "memory.sqlite")) as memory:
            self.assertEqual(len(memory.outcomes.history()), 2)
            self.assertEqual(memory.outcomes.pending_count(), 1)

    def test_review_only_then_controller_plan_uses_separate_budgets(self):
        self.backlog()
        contexts = []

        class Reviewer:
            def decide(self, request):
                context = json.loads(request[1]["content"][0]["text"])
                contexts.append(context)
                value, trace = DemoAgent().decide(request)
                if len(contexts) == 1:
                    value.update(memory_updates=[], events=[])
                    value["action"].update(kind="review_outcomes", buttons=[])
                return value, trace

        emulator = ReplayEmulator(self.images)
        result = run(self.config, emulator, Reviewer(), turns=1)
        self.assertEqual(result["completed_actions"], 1)
        self.assertEqual(len(emulator.actions), 1)
        self.assertEqual(contexts[1]["memory_available"]["searches_remaining"], 3)

    def test_backups_restore_review_history_pending_work_and_case_gate(self):
        run(self.config, ReplayEmulator(self.images), DemoAgent(), turns=2)
        with contextlib.closing(Memory(self.directory / "memory.sqlite")) as memory:
            original = memory.outcomes.history()
            pending = memory.outcomes.context()
        backup = create_backup(self.directory, self.root / "backups")
        target = self.root / "restored"
        restore_backup(backup, target)
        with contextlib.closing(Memory(target / "memory.sqlite")) as restored:
            self.assertEqual(restored.outcomes.history(), original)
            self.assertEqual(restored.outcomes.context(), pending)
            self.assertTrue(restored.escalations.review([])["blocked_cases"])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["outcomes", str(target), "--history", "--limit", "1"]), 0)
        self.assertEqual(json.loads(output.getvalue())["history"], original[:1])
