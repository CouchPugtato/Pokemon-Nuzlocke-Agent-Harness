import copy
import unittest

from jsonschema import Draft202012Validator

from pokemon_harness.agent.prompts import messages
from pokemon_harness.agent.schemas import DECISION_SCHEMA, schema_for_context
from pokemon_harness.demo import DemoAgent


def context():
    return {"observation_id": 16, "action_limit_frames": 60, "progress": {
        "requires_recovery": True, "episode_id": 1, "recovery_attempts_remaining": 2,
        "last_input_signature": [["wait", []]], "recovery_signatures": []}}


def decision(state):
    value = DemoAgent().decide(messages(state, b"png"))[0]
    value.update(memory_updates=[], scene="dialogue", recovery_plan={
        "episode_id": 1, "explanation": "Unchanged complete dialogue suggests a confirmation press"})
    value["action"].update(buttons=["A"])
    value["strategy"]["kind"] = "advance_dialogue"
    return value


class RecoverySchemaTests(unittest.TestCase):
    def test_recovery_requires_current_episode_and_single_alternative(self):
        state = context()
        schema = schema_for_context(state)
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        valid = decision(state)
        validator.validate(valid)
        variants = []
        for plan in (None, {"episode_id": 2, "explanation": "Alternative"},
                     {"episode_id": 1, "explanation": ""}):
            value = copy.deepcopy(valid)
            value["recovery_plan"] = plan
            variants.append(value)
        value = copy.deepcopy(valid)
        value["additional_actions"] = [copy.deepcopy(value["action"])]
        variants.append(value)
        value = copy.deepcopy(valid)
        value["strategy"] = None
        variants.append(value)
        for value in variants:
            self.assertFalse(validator.is_valid(value))

    def test_repeat_signatures_cannot_be_evaded_by_retiming(self):
        state = context()
        state["progress"]["recovery_signatures"] = [[["press", ["A"]]]]
        validator = Draft202012Validator(schema_for_context(state))
        value = decision(state)
        value["action"]["hold_frames"] = 30
        self.assertFalse(validator.is_valid(value))
        value["action"].update(kind="wait", buttons=[])
        self.assertFalse(validator.is_valid(value))
        value["action"].update(kind="press", buttons=["B"])
        validator.validate(value)

    def test_uncertain_and_battle_scenes_only_allow_fresh_wait_tests(self):
        state = context()
        state["progress"]["last_input_signature"] = [["press", ["B"]]]
        validator = Draft202012Validator(schema_for_context(state))
        for scene in ("unknown", "battle", "transition"):
            value = decision(state)
            value["scene"] = scene
            self.assertFalse(validator.is_valid(value))
            value["action"].update(kind="wait", buttons=[])
            validator.validate(value)

    def test_exhausted_budget_only_allows_stop(self):
        state = context()
        state["progress"]["recovery_attempts_remaining"] = 0
        validator = Draft202012Validator(schema_for_context(state))
        value = decision(state)
        self.assertFalse(validator.is_valid(value))
        value.update(strategy=None, recovery_plan=None)
        value["action"].update(kind="stop", buttons=[], hold_frames=1, settle_frames=0)
        validator.validate(value)

    def test_search_remains_readonly_and_respects_search_budget(self):
        state = context()
        state["memory_available"] = {"searches_remaining": 1}
        state["progress"]["recovery_attempts_remaining"] = 0
        value = decision(state)
        value.update(strategy=None, recovery_plan=None, memory_search={
            "query": "dialogue", "scope": "all", "category": "", "location": "",
            "include_history": False, "offset": 0, "limit": 3, "observation_ids": []})
        value["action"].update(kind="search_memory", buttons=[], hold_frames=1, settle_frames=0)
        validator = Draft202012Validator(schema_for_context(state))
        validator.validate(value)
        state["memory_available"]["searches_remaining"] = 0
        self.assertFalse(Draft202012Validator(schema_for_context(state)).is_valid(value))
        value["memory_updates"] = DemoAgent().decide(messages(state, b"png"))[0]["memory_updates"]
        self.assertFalse(validator.is_valid(value))

    def test_recovery_respects_frame_limits_and_forbids_state_writes(self):
        state = context()
        state["action_limit_frames"] = 10
        validator = Draft202012Validator(schema_for_context(state))
        value = decision(state)
        value["action"]["hold_frames"] = 11
        self.assertFalse(validator.is_valid(value))
        value["action"]["hold_frames"] = 6
        validator.validate(value)
        value["memory_updates"] = DemoAgent().decide(messages(state, b"png"))[0]["memory_updates"]
        self.assertFalse(validator.is_valid(value))

    def test_review_phase_takes_priority_and_base_schema_is_unchanged(self):
        from test_review_schema import context as pending_context, review

        before = copy.deepcopy(DECISION_SCHEMA)
        state = context()
        schema_for_context(state)
        state.update(pending_context())
        validator = Draft202012Validator(schema_for_context(state))
        validator.validate(review(state))
        self.assertFalse(validator.is_valid(decision(state)))
        self.assertEqual(DECISION_SCHEMA, before)
