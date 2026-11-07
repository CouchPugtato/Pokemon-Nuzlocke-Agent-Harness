import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from pokemon_harness.agent.prompts import messages
from pokemon_harness.agent.schemas import validate_decision
from pokemon_harness.cli import main
from pokemon_harness.config import load_config
from pokemon_harness.demo import DemoAgent, create_images, demo_outcome_reviews
from pokemon_harness.emulator.interface import ReplayEmulator
from pokemon_harness.memory.database import Memory
from pokemon_harness.memory.retrieval import ledger_context, recent_turn_context, retrieve
from pokemon_harness.runner import run
from pokemon_harness.telemetry.backup import create_backup
from pokemon_harness.telemetry.recorder import write_json
from pokemon_harness.telemetry.restore import restore_backup


def fact(key="location", basis="observed", **changes):
    return dict({"category": "world", "key": key, "claim": "A visible landmark", "location": "Town",
                 "basis": basis, "confidence": "high", "status": "active"}, **changes)


def conflict(facts=(), events=(), key="location:conflict", **changes):
    return dict({"operation": "open", "key": key, "expected_revision": 0,
                 "reason": "Fresh screenshot conflicts with the earlier interpretation",
                 "evidence_observation_ids": [], "fact_ids": list(facts), "event_keys": list(events),
                 "replacement_facts": [], "correction_event_keys": []}, **changes)


def working(revision=0, operation="start"):
    definition = {"description": "Find an exit", "objective_relation": "Reach gym",
                  "completion_condition": "Exit visible", "abandonment_condition": "Unsafe route"}
    return {"expected_revision": revision, "location": None, "uncertainties": [], "next_check": "Inspect exit",
            "subgoal_change": {"operation": operation, "id": "find:exit", "definition": definition,
                               "reason": "Inspect exit evidence", "evidence_observation_ids": []}}


class ContradictionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.memory = Memory(self.root / "memory.sqlite")
        self.addCleanup(self.memory.close)
        self.memory.initialize({"objective": "Beat gym", "rules": []})

    def save(self, facts=(), links=(), changes=(), state=None, working_sources=None, events=(), working_events=None):
        observation = self.memory.observe("unused.png", "hash", "test")
        value = DemoAgent().decide(messages({"observation_id": observation}, b"png"))[0]
        value["action"].update(kind="stop", buttons=[])
        value.update(memory_updates=list(facts), events=list(events), working_state_update=state,
                     memory_dependencies={"facts": list(links), "working_state": working_sources, "working_state_events": working_events},
                     contradiction_updates=copy.deepcopy(list(changes)))
        for change in value["contradiction_updates"]:
            change["evidence_observation_ids"] = [observation]
        turn = self.memory.plan(observation, value)
        self.memory.status(turn, "stopped")
        return observation

    def dependency(self, key, sources):
        return {"fact": {"category": "world", "key": key}, "depends_on_fact_ids": sources, "depends_on_event_keys": []}

    def graph(self, include_working=True):
        self.save([fact()])
        self.save([fact("route", "inferred")], [self.dependency("route", [1])])
        self.save([fact("gym", "inferred")], [self.dependency("gym", [2])])
        if include_working:
            self.save(state=working(), working_sources=[3])
        return [1, 2, 3]

    def query(self, scope="facts", history=True, query=""):
        return self.memory.search({"query": query, "scope": scope, "category": "", "location": "",
                                   "include_history": history, "offset": 0, "limit": 10},
                                  self.memory.previous_observation()["id"])

    def resolve(self):
        return conflict(key="location:conflict", operation="resolve", expected_revision=1,
                        replacement_facts=[{"category": "world", "key": "location"}])

    def test_transitive_disputes_remove_default_recall_but_keep_original_evidence(self):
        self.graph()
        self.save(changes=[conflict([1])])
        facts = self.memory.facts()
        by_id = {item["fact_id"]: item for item in facts}
        self.assertEqual([by_id[index]["memory_validity"] for index in (1, 2, 3)],
                         ["disputed", "invalidated", "invalidated"])
        self.assertEqual(retrieve(facts, "gym", 6), [])
        results = self.query()["results"]
        self.assertEqual(len(results), 3)
        self.assertEqual(by_id[1]["evidence_observation_id"], 1)
        self.assertEqual(by_id[3]["depends_on_fact_ids"], [2])
        self.assertEqual(self.memory.contradictions.context()["open_count"], 1)
        projected = self.memory.working_context()
        self.assertEqual(projected["memory_validity"], "invalidated")
        self.assertIsNone(projected["subgoal"]["definition"])
        self.assertEqual(projected["subgoal"]["id"], "find:exit")
        self.assertEqual(self.memory.working.current()["subgoal"]["definition"]["description"], "Find an exit")

    def test_new_claim_using_disputed_dependency_is_also_invalidated(self):
        self.graph(False)
        self.save(changes=[conflict([1])])
        self.save([fact("safety", "inferred")], [self.dependency("safety", [3])])
        latest = self.memory.facts()[0]
        self.assertEqual(latest["memory_validity"], "invalidated")
        self.assertEqual(latest["contradiction_keys"], ["location:conflict"])

    def test_overwriting_disputed_key_does_not_resolve_the_contradiction(self):
        self.graph(False)
        self.save(changes=[conflict([1])])
        self.save([fact(claim="A renamed interpretation")])
        self.assertEqual(self.memory.facts()[0]["memory_validity"], "disputed")
        self.assertEqual(self.memory.contradictions.context()["open_count"], 1)

    def test_resolution_restores_only_fresh_replacement_not_old_dependents(self):
        self.graph()
        self.save(changes=[conflict([1])])
        self.save([fact(claim="Freshly inspected location")], changes=[self.resolve()])
        by_id = {item["fact_id"]: item for item in self.memory.facts(True)}
        self.assertEqual(by_id[4]["memory_validity"], "supported")
        self.assertEqual(by_id[1]["memory_validity"], "disputed")
        self.assertEqual(by_id[2]["memory_validity"], "invalidated")
        self.assertEqual(self.memory.working_context()["memory_validity"], "invalidated")
        self.assertEqual(self.memory.contradictions.context()["open_count"], 0)
        self.save([fact("route", "inferred")], [self.dependency("route", [4])])
        self.assertEqual(self.memory.facts()[0]["memory_validity"], "inferred")
        self.save(state=working(1, "revise"), working_sources=[5])
        self.assertNotIn("memory_validity", self.memory.working_context())
        self.assertEqual(self.memory.contradictions.history()[-1]["operation"], "open")
        self.assertEqual(self.memory.contradictions.history()[0]["operation"], "resolve")

    def test_resolving_cannot_reuse_old_or_inferred_or_uncertain_replacements(self):
        self.graph(False)
        self.save(changes=[conflict([1])])
        for replacement in (None, fact(basis="inferred"), fact(confidence="medium"), fact(status="uncertain")):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                self.save([replacement] if replacement else [], changes=[self.resolve()])
        self.assertEqual(len(self.memory.facts(True)), 3)
        self.assertEqual(self.memory.contradictions.context()["open_count"], 1)

    def test_resolving_cannot_ignore_a_second_open_dispute(self):
        self.graph(False)
        self.save(changes=[conflict([1]), conflict([1], key="other:conflict")])
        with self.assertRaisesRegex(ValueError, "other disputes"):
            self.save([fact()], changes=[self.resolve()])
        self.assertEqual(self.memory.contradictions.context()["open_count"], 2)

    def test_dependency_validation_rejects_missing_self_and_same_plan_sources(self):
        for source in (99, 1):
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.save([fact("route", "inferred")], [self.dependency("route", [source])])
        with self.assertRaises(ValueError):
            self.save([fact(), fact("route", "inferred")], [self.dependency("route", [1])])
        self.assertEqual(self.memory.facts(), [])

    def test_dependency_links_require_matching_new_inferred_facts(self):
        self.save([fact()])
        invalid = [([fact("route")], [self.dependency("route", [1])]),
                   ([], [self.dependency("route", [1])]),
                   ([fact("route", "inferred")], [self.dependency("route", [1])] * 2)]
        for facts, links in invalid:
            with self.subTest(links=links), self.assertRaises(ValueError):
                self.save(facts, links)
        self.assertEqual(len(self.memory.facts(True)), 1)

    def test_stale_working_dependencies_are_inherited_and_cannot_be_erased_with_goal_unchanged(self):
        self.graph()
        self.save(changes=[conflict([1])])
        update = working(1)
        update["subgoal_change"] = None
        self.save(state=update)
        self.assertEqual(self.memory.working_context()["memory_validity"], "invalidated")
        update["expected_revision"] = 2
        with self.assertRaisesRegex(ValueError, "stale subgoal"):
            self.save(state=update, working_sources=[])
        self.assertEqual(self.memory.working.current()["revision"], 2)

    def test_stale_revision_and_duplicate_operations_roll_back_all_writes_and_cases(self):
        self.save([fact()])
        change = conflict([1])
        with self.assertRaisesRegex(ValueError, "unique per decision"):
            self.save([fact("new")], changes=[change, change])
        self.assertEqual(self.memory.contradictions.context()["open_count"], 0)
        self.assertEqual(len(self.memory.facts()), 1)
        self.save(changes=[change])
        with self.assertRaisesRegex(ValueError, "stale revision"):
            self.save([fact()], changes=[conflict(operation="resolve", replacement_facts=[{"category": "world", "key": "location"}])])
        self.assertEqual(len(self.memory.facts(True)), 1)

    def test_disputing_an_event_creates_a_blocking_case_without_hiding_ledger_history(self):
        event = {"key": "faint:starter", "type": "faint", "subject": "Starter", "area": "Route 101",
                 "details": "Fainting was initially reported", "confidence": "high", "corrects_event_key": ""}
        self.save(events=[event])
        self.save(changes=[conflict(events=[event["key"]])])
        events = self.memory.events()
        self.assertEqual(events[0]["memory_validity"], "disputed")
        self.assertIn("Starter", ledger_context(events, "Route 101")["retired_pokemon"])
        case = self.memory.escalations.review(events)["blocked_cases"][0]
        self.memory.escalations.resolve(case["id"], "user-assertion", "User reviewed uncertainty", allow_continue=True)
        self.assertEqual(self.memory.events()[0]["memory_validity"], "disputed")
        self.assertEqual(self.memory.contradictions.context()["open_count"], 1)

    def test_event_resolution_requires_a_fresh_linked_correction(self):
        event = {"key": "capture:starter", "type": "capture", "subject": "Starter", "area": "Route 101",
                 "details": "An acquisition was reported", "confidence": "high", "corrects_event_key": ""}
        self.save(events=[event])
        self.save(changes=[conflict(events=[event["key"]])])
        resolution = conflict(operation="resolve", expected_revision=1, correction_event_keys=["correct:starter"])
        with self.assertRaisesRegex(ValueError, "fresh high confidence corrections"):
            self.save(changes=[resolution])
        correction = dict(event, key="correct:starter", type="correction", corrects_event_key=event["key"],
                          details="Fresh party inspection contradicts acquisition")
        self.save(events=[correction], changes=[resolution])
        review = self.memory.escalations.review(self.memory.events())
        self.assertEqual(review["blocked_cases"], [])
        self.assertEqual(self.memory.contradictions.context()["open_count"], 0)
        self.assertEqual(len(self.memory.events()), 2)

    def test_case_creation_is_rolled_back_when_later_working_update_fails(self):
        event = {"key": "badge:first", "type": "badge", "subject": "Badge", "area": "Gym",
                 "details": "Badge was reported", "confidence": "high", "corrects_event_key": ""}
        self.save(events=[event])
        with self.assertRaisesRegex(ValueError, "stale revision"):
            self.save(changes=[conflict(events=[event["key"]])], state=working(99))
        self.assertEqual(self.memory.escalations.cases(), [])
        self.assertEqual(self.memory.contradictions.history(), [])

    def test_search_returns_dispute_history_and_preserves_fact_provenance(self):
        self.graph(False)
        self.save(changes=[conflict([1])])
        self.save([fact()], changes=[self.resolve()])
        records = self.query("contradictions")["results"]
        self.assertEqual({record["operation"] for record in records}, {"open", "resolve"})
        self.assertEqual({record["record_type"] for record in records}, {"contradiction"})
        latest = self.query("contradictions", False)["results"]
        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0]["revision"], 2)
        old = next(record for record in self.query()["results"] if record["fact_id"] == 1)
        self.assertEqual(old["memory_validity"], "disputed")
        self.assertEqual(old["evidence_observation_id"], 1)

    def test_search_and_review_contracts_cannot_write_conflicts_or_dependencies(self):
        for kind in ("search_memory", "review_outcomes"):
            value = DemoAgent().decide(messages({"observation_id": 1}, b"png"))[0]
            value.update(memory_updates=[], events=[])
            value["action"].update(kind=kind, buttons=[])
            if kind == "search_memory":
                value["memory_search"] = {"query": "", "scope": "contradictions", "category": "", "location": "",
                                          "include_history": False, "offset": 0, "limit": 1}
            else:
                value["outcome_reviews"] = [{"turn_id": 1, "expected_revision": 0, "steps": [{"step_index": 1,
                    "before_observation_id": 1, "after_observation_id": 2, "assessment": "uncertain", "explanation": "Unknown"}]}]
            value["memory_dependencies"]["working_state"] = []
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_decision(value, 1)

    def test_event_dispute_invalidates_direct_and_transitive_dependents_and_working_state(self):
        event = {"key": "capture:starter", "type": "capture", "subject": "Starter", "area": "Route 101",
                 "details": "Acquisition reported", "confidence": "high", "corrects_event_key": ""}
        self.save(events=[event])
        link = self.dependency("party", [])
        link["depends_on_event_keys"] = [event["key"]]
        self.save([fact("party", "inferred")], [link])
        self.save([fact("readiness", "inferred")], [self.dependency("readiness", [1])])
        self.save(state=working(), working_events=[event["key"]])
        self.save(changes=[conflict(events=[event["key"]])])
        self.assertEqual({item["memory_validity"] for item in self.memory.facts()}, {"invalidated"})
        self.assertEqual(self.memory.working_context()["memory_validity"], "invalidated")
        self.assertEqual(self.memory.facts()[1]["depends_on_event_keys"], [event["key"]])

    def test_ledger_correction_invalidates_dependents_without_an_explicit_dispute(self):
        event = {"key": "capture:starter", "type": "capture", "subject": "Starter", "area": "Route 101",
                 "details": "Acquisition reported", "confidence": "high", "corrects_event_key": ""}
        self.save(events=[event])
        link = self.dependency("party", [])
        link["depends_on_event_keys"] = [event["key"]]
        self.save([fact("party", "inferred")], [link])
        self.save([fact("readiness", "inferred")], [self.dependency("readiness", [1])])
        correction = dict(event, key="correction:starter", type="correction", corrects_event_key=event["key"])
        self.save(events=[correction])
        self.assertEqual({item["memory_validity"] for item in self.memory.facts()}, {"invalidated"})
        self.assertEqual(self.memory.contradictions.context()["open_count"], 0)
        self.assertEqual(len(self.memory.contradictions.history()), 2)
        self.assertEqual(self.memory.events()[0]["memory_validity"], "disputed")
        with self.assertRaisesRegex(ValueError, "earlier saved events"):
            self.save([fact("new", "inferred")], [dict(link, fact={"category": "world", "key": "new"})])

    def test_disputed_goal_is_withheld_from_outcome_review_and_progress_credit(self):
        self.graph()
        observation = self.memory.observe("before.png", "before", "test")
        value = DemoAgent().decide(messages({"observation_id": observation}, b"png"))[0]
        value["memory_updates"] = []
        turn = self.memory.plan(observation, value)
        after = self.memory.observe("after.png", "after", "test")
        self.memory.step_status(turn, 1, "completed", after)
        self.memory.status(turn, "completed")
        reviews = demo_outcome_reviews({"observation_id": after, "outcome_review": self.memory.outcomes.context()})
        reviews[0]["steps"][0]["assessment"] = "progress_observed"
        self.memory.review_outcomes(after, reviews)
        self.assertEqual(self.memory.monitor_progress()["no_progress_attempts"], 0)
        self.save(changes=[conflict([1])])
        plan = self.memory.outcomes.plan_context(turn)
        self.assertIsNone(plan["subgoal_at_input"]["definition"])
        self.assertEqual(plan["subgoal_memory_validity"], "invalidated")
        self.assertEqual(self.memory.monitor_progress()["no_progress_attempts"], 1)
        recalled = next(item for item in recent_turn_context(self.memory.recent_turns(20)) if item["id"] == turn)
        self.assertEqual(recalled["decision"]["scene"], "unknown")
        self.assertEqual(recalled["outcome_review"]["steps"], [])
        self.assertEqual(self.memory.outcomes.latest(turn)["steps"][0]["assessment"], "progress_observed")


class ContradictionIntegrationTests(unittest.TestCase):
    def test_fact_graph_and_stale_working_projection_survive_backup_and_restore(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            write_json(config_path, {"run_dir": "run", "objective": "Test", "model": {"name": "test"},
                                     "emulator": {"backend": "replay"}})
            config = load_config(config_path)
            directory = Path(config["run_dir"])
            images = create_images(root / "fixtures")
            requests = []

            class Agent:
                def decide(self, request):
                    context = json.loads(request[1]["content"][0]["text"])
                    requests.append(context)
                    value, trace = DemoAgent().decide(request)
                    if len(requests) == 1:
                        value["memory_updates"] = [fact()]
                    elif len(requests) == 2:
                        value["memory_updates"] = [fact("route", "inferred")]
                        value["memory_dependencies"]["facts"] = [{"fact": {"category": "world", "key": "route"},
                            "depends_on_fact_ids": [1], "depends_on_event_keys": []}]
                        value["working_state_update"] = working()
                        value["memory_dependencies"]["working_state"] = [1]
                    else:
                        value["action"].update(kind="stop", buttons=[])
                        value["memory_updates"] = []
                        value["contradiction_updates"] = [conflict([1], evidence_observation_ids=[context["observation_id"]])]
                    return value, trace

            run(config, ReplayEmulator(images), Agent(), turns=3)
            with contextlib.closing(Memory(directory / "memory.sqlite")) as memory:
                facts = memory.facts(True)
                projected = memory.working_context()
                self.assertEqual(projected["memory_validity"], "invalidated")
            backup = create_backup(directory, root / "backups")
            target = root / "restored"
            restore_backup(backup, target)
            with contextlib.closing(Memory(target / "memory.sqlite")) as restored:
                self.assertEqual(restored.facts(True), facts)
                self.assertEqual(restored.working_context(), projected)
                self.assertEqual(retrieve(restored.facts(), "route", 4), [])
                self.assertEqual(restored.working.current()["subgoal"]["definition"]["description"], "Find an exit")

    def test_runner_backup_restore_and_inspection_preserve_blocking_disputes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            write_json(config_path, {"run_dir": "run", "objective": "Test", "model": {"name": "test"},
                                     "emulator": {"backend": "replay"}})
            config = load_config(config_path)
            directory = Path(config["run_dir"])
            images = create_images(root / "fixtures")
            requests = []

            class Agent:
                def decide(self, request):
                    context = json.loads(request[1]["content"][0]["text"])
                    requests.append(context)
                    value, trace = DemoAgent().decide(request)
                    if len(requests) == 1:
                        value["events"] = [{"key": "faint:starter", "type": "faint", "subject": "Starter",
                            "area": "Route 101", "details": "Faint was reported", "confidence": "high", "corrects_event_key": ""}]
                    else:
                        value["contradiction_updates"] = [conflict(events=["faint:starter"],
                            evidence_observation_ids=[context["observation_id"]])]
                    return value, trace

            emulator = ReplayEmulator(images)
            result = run(config, emulator, Agent(), turns=2)
            self.assertEqual(result["reason"], "unresolved_events")
            self.assertEqual(len(emulator.actions), 1)
            with contextlib.closing(Memory(directory / "memory.sqlite")) as memory:
                original = memory.contradictions.history()
                facts = memory.facts(True)
            backup = create_backup(directory, root / "backups")
            target = root / "restored"
            restore_backup(backup, target)
            with contextlib.closing(Memory(target / "memory.sqlite")) as restored:
                self.assertEqual(restored.contradictions.history(), original)
                self.assertEqual(restored.facts(True), facts)
                self.assertEqual(restored.events()[0]["memory_validity"], "disputed")
                self.assertTrue(restored.escalations.review(restored.events())["blocked_cases"])
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["contradictions", str(target), "--history", "--limit", "1"]), 0)
            self.assertEqual(json.loads(output.getvalue())["history"], original[:1])
