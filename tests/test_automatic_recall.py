import contextlib
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pokemon_harness.agent.context_budget import fit_request
from pokemon_harness.agent.prompts import messages
from pokemon_harness.config import load_config
from pokemon_harness.demo import DemoAgent, create_images
from pokemon_harness.emulator.interface import ReplayEmulator
from pokemon_harness.memory.database import Memory
from pokemon_harness.memory.recall import automatic_recall, recall_settings, terms
from pokemon_harness.runner import run
from pokemon_harness.telemetry.recorder import write_json


def fact(source_id, **changes):
    return dict({"fact_id": source_id, "category": "world", "key": f"fact:{source_id}",
                 "claim": "Old memory", "location": "", "basis": "observed", "confidence": "low",
                 "status": "active", "memory_validity": "supported", "evidence_observation_id": source_id}, **changes)


def event(source_id, **changes):
    return dict({"event_id": source_id, "key": f"event:{source_id}", "type": "encounter",
                 "subject": "Kip", "area": "Route 101", "details": "An encounter was seen",
                 "corrects_event_key": "", "confidence": "low", "evidence_observation_id": source_id}, **changes)


def state(**changes):
    return dict({"location": None, "uncertainties": [], "next_check": "", "subgoal": None}, **changes)


def recall(facts, events=(), working=None, recent=(), disputes=(), limit=6, **settings):
    return automatic_recall(facts, events, working or state(), recent, disputes, "Beat gym", 100,
                            limit, recall_settings({"automatic_recall": settings}))


class AutomaticRecallTests(unittest.TestCase):
    def test_subgoal_and_party_survive_objective_crowding_without_a_model_query(self):
        facts = [fact(index, category="objective", claim="Beat gym") for index in range(10, 50)]
        facts += [fact(1, claim="The parcel is upstairs"), fact(2, category="party", key="kip:hp", claim="Kip HP was low")]
        working = state(subgoal={"status": "active", "definition": {"description": "Collect parcel"}})
        selected, _ = recall(facts, working=working, limit=3)
        self.assertIn(1, [item["fact_id"] for item in selected])
        self.assertIn(2, [item["fact_id"] for item in selected])
        self.assertEqual(len(selected), 3)

    def test_uncertainty_and_next_check_find_related_lessons(self):
        working = state(uncertainties=[{"claim": "Which doorway is safe"}], next_check="Check stairs")
        selected, _ = recall([fact(1, category="lesson", claim="The stairs led downstairs"),
                              fact(2, claim="Doorway is blocked")], working=working)
        self.assertIn("uncertainty", selected[0]["recall_reasons"])
        self.assertTrue(any("lesson" in item["recall_reasons"] for item in selected))

    def test_aliases_and_normalization_match_search_hints_without_changing_claims(self):
        working = state(subgoal={"status": "active", "definition": {"description": "Inspect Mudkip"}})
        original = fact(1, category="party", key="kip:hp", claim="Kip HP was low")
        selected, _ = recall([original], working=working, aliases=[["Mudkip", "Kip"]])
        self.assertIn("subgoal", selected[0]["recall_reasons"])
        self.assertEqual(selected[0]["claim"], original["claim"])
        self.assertEqual(selected[0]["confidence"], "low")
        self.assertEqual(terms("Pokémon Route101 OldaleTown"), terms("Pokemon Route 101 Oldale Town"))

    def test_ambiguous_locations_are_bounded_source_linked_hypotheses(self):
        working = state(location={"claim": "Route101", "basis": "inferred", "evidence_observation_ids": [1]})
        recent = [{"id": 3, "observation_id": 3, "decision": {"location": "Oldale Town"}}]
        _, report = recall([fact(2, location="Rustboro")], working=working, recent=recent, location_limit=2)
        self.assertEqual([item["claim"] for item in report["location_candidates"]], ["Route101", "Oldale Town"])
        self.assertTrue(all(item["hypothesis_only"] for item in report["location_candidates"]))
        self.assertEqual(report["omitted"]["location_candidates"], 1)

    def test_invalidated_state_and_disputed_turns_cannot_seed_location_or_goal(self):
        working = state(memory_validity="invalidated", location={"claim": "Secret Town"},
                        subgoal={"status": "active", "definition": {"description": "Secret"}})
        recent = [{"id": 2, "observation_id": 2, "contradiction_keys": ["bad"], "decision": {"location": "Secret"}}]
        selected, report = recall([fact(1, memory_validity="disputed"), fact(2, memory_validity="invalidated"),
                                   fact(3, status="resolved"), fact(101)], working=working, recent=recent)
        self.assertEqual(selected, [])
        self.assertEqual(report["location_candidates"], [])

    def test_ledger_evidence_keeps_corrections_confidence_and_original_source_ids(self):
        events = [event(1), event(2, type="correction", corrects_event_key="event:1", details="Earlier encounter was mistaken")]
        _, report = recall([fact(3, category="party", key="kip:hp")], events=events)
        records = {item["event_id"]: item for item in report["evidence"]}
        self.assertTrue(records[1]["superseded"])
        self.assertEqual(records[1]["corrected_by"], ["event:2"])
        self.assertEqual(records[1]["confidence"], "low")
        self.assertEqual(records[1]["evidence_observation_id"], 1)
        self.assertEqual(records[1]["age_observations"], 99)

    def test_disputes_are_separate_from_active_facts_and_offer_evidence_ids(self):
        dispute = {"operation": "open", "key": "conflict", "contradiction_revision_id": 1,
                   "evidence_observation_id": 2, "fact_ids": [1]}
        selected, report = recall([fact(1, memory_validity="disputed")], disputes=[dispute])
        self.assertEqual(selected, [])
        self.assertEqual(report["evidence"][0]["record_type"], "contradiction")
        self.assertEqual(report["image_candidate_observation_ids"], [2])
        self.assertNotIn("image", report)

    def test_limits_deduplicate_sources_and_report_omissions_without_mutation(self):
        facts = [fact(index, category="party", key="kip:hp", claim="Kip") for index in range(1, 20)]
        events = [event(index) for index in range(1, 20)]
        before = copy.deepcopy((facts, events))
        selected, report = recall(facts, events, limit=3, evidence_limit=2, image_candidate_limit=1)
        self.assertEqual(len({item["fact_id"] for item in selected}), 3)
        self.assertEqual(len(report["evidence"]), 2)
        self.assertEqual(len(report["image_candidate_observation_ids"]), 1)
        self.assertEqual(report["omitted"]["facts"], 16)
        self.assertEqual((facts, events), before)

    def test_disputes_do_not_crowd_out_ledger_evidence_or_import_future_corrections(self):
        disputes = [{"operation": "open", "key": f"conflict:{index}", "evidence_observation_id": index}
                    for index in range(1, 10)]
        events = [event(1), event(101, type="correction", corrects_event_key="event:1")]
        _, report = recall([fact(2, category="party", key="kip:hp")], events=events, disputes=disputes, evidence_limit=2)
        self.assertEqual([item["record_type"] for item in report["evidence"]], ["contradiction", "event"])
        self.assertFalse(report["evidence"][1]["superseded"])

    def test_invalid_settings_are_rejected(self):
        for value in (None, {"evidence_limit": True}, {"location_limit": 0}, {"image_candidate_limit": 99},
                      {"aliases": ["Kip"]}, {"aliases": [["Kip", ""]]}, {"unknown": 1}):
            with self.assertRaises(ValueError):
                recall_settings({"automatic_recall": value})

    def test_automatic_evidence_can_be_trimmed_without_displacing_rules_or_ledger(self):
        context = {"observation_id": 1, "rules": ["No items"], "ledger": {"retired_pokemon": {"Kip": "high"}},
                   "automatic_recall": {"evidence": [{"details": "x" * 100000}]}}
        model = {"name": "fixture", "context_length": 32768, "max_tokens": 8192}
        selected, _, report = fit_request(context, model, b"png")
        self.assertEqual(report["omitted"]["automatic_evidence"], 1)
        self.assertEqual(selected["ledger"], context["ledger"])
        self.assertEqual(selected["rules"], context["rules"])


class AutomaticRecallIntegrationTests(unittest.TestCase):
    def test_resumed_run_recalls_subgoal_sources_without_an_extra_search_or_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "config.json"
            write_json(path, {"run_dir": "run", "objective": "Beat gym", "rules": [], "memory_limit": 1,
                              "model": {"name": "fixture"}, "emulator": {"backend": "replay"}})
            config = load_config(path)
            directory = Path(config["run_dir"])
            images = create_images(directory / "screenshots")
            with contextlib.closing(Memory(directory / "memory.sqlite")) as memory:
                memory.initialize(config)
                observation = memory.observe(images[0].relative_to(directory), hashlib.sha256(images[0].read_bytes()).hexdigest(), "test")
                value = DemoAgent().decide(messages({"observation_id": observation}, images[0].read_bytes()))[0]
                value["action"].update(kind="stop", buttons=[])
                value["memory_updates"] = [{key: item for key, item in fact(1, key="parcel", claim="Parcel upstairs").items()
                                             if key in {"category", "key", "claim", "location", "basis", "confidence", "status"}}]
                value["working_state_update"] = {
                    "expected_revision": 0, "location": None, "uncertainties": [], "next_check": "Find parcel",
                    "subgoal_change": {"operation": "start", "id": "parcel", "reason": "Retrieve parcel", "evidence_observation_ids": [observation],
                        "definition": {"description": "Collect parcel", "objective_relation": "Advance", "completion_condition": "Parcel obtained",
                                       "abandonment_condition": "Unsafe"}}}
                turn = memory.plan(observation, value)
                memory.status(turn, "stopped")
                memory.save_checkpoint()
            contexts = []

            class Agent:
                def decide(self, request):
                    contexts.append(json.loads(request[1]["content"][0]["text"]))
                    return DemoAgent().decide(request)

            emulator = ReplayEmulator(images)
            run(config, emulator, Agent(), turns=1)
            self.assertEqual(len(contexts), 1)
            self.assertEqual(len(emulator.actions), 1)
            self.assertEqual(contexts[0]["memory"][0]["key"], "parcel")
            self.assertIn("subgoal", contexts[0]["memory"][0]["recall_reasons"])
            self.assertEqual(contexts[0]["automatic_recall"]["image_candidate_observation_ids"], [1])
