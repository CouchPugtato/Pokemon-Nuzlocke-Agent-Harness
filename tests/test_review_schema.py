import copy
import io
import json
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator, ValidationError

from pokemon_harness.agent.context_budget import assess_request, fit_request
from pokemon_harness.agent.lmstudio import InvalidModelResponse, LMStudio
from pokemon_harness.agent.prompts import messages
from pokemon_harness.agent.schemas import DECISION_SCHEMA, request_schema, schema_for_context
from pokemon_harness.demo import DemoAgent


def context():
    return {"observation_id": 7, "outcome_visual_evidence_ids": [5, 6, 7], "outcome_review": {
        "pending_count": 1, "next_plan": {"turn_id": 3, "expected_revision": 0, "steps": [
            {"step_index": 1, "input_status": "completed", "before_observation_id": 5,
             "after_observation_id": 6}]}}}


def review(value):
    decision = DemoAgent().decide(messages(value, b"png"))[0]
    decision.update(memory_updates=[], events=[], strategy=None, recovery_plan=None)
    decision["action"] = {"kind": "review_outcomes", "buttons": [], "hold_frames": 1, "settle_frames": 0}
    return decision


class ReviewSchemaTests(unittest.TestCase):
    def test_action_phase_excludes_unrequested_review_revisions(self):
        state = {"observation_id": 23, "outcome_review": {"pending_count": 0, "next_plan": None}}
        before = copy.deepcopy(DECISION_SCHEMA)
        schema = schema_for_context(state)
        validator = Draft202012Validator(schema)
        value = DemoAgent().decide(messages(state, b"png"))[0]
        validator.validate(value)
        value["outcome_reviews"] = review(context())["outcome_reviews"]
        self.assertFalse(validator.is_valid(value))
        value["outcome_reviews"] = []
        value["action"].update(kind="review_outcomes", buttons=[])
        self.assertFalse(validator.is_valid(value))
        self.assertEqual(DECISION_SCHEMA, before)

    def test_client_rejects_a_server_that_ignores_the_action_phase_schema(self):
        state = {"observation_id": 23, "outcome_review": {"pending_count": 0, "next_plan": None}}
        request = messages(state, b"png")
        value = DemoAgent().decide(request)[0]
        value["outcome_reviews"] = review(context())["outcome_reviews"]
        response = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}).encode()
        with patch("pokemon_harness.agent.lmstudio.urlopen", return_value=io.BytesIO(response)):
            with self.assertRaises(InvalidModelResponse) as error:
                LMStudio({"name": "fixture"}).decide(request)
        self.assertIn("response", error.exception.trace)

    def test_pending_phase_allows_review_or_stop_but_no_input_or_writes(self):
        state = context()
        schema = schema_for_context(state)
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        value = review(state)
        validator.validate(value)
        for kind in ("press", "wait", "search_memory"):
            invalid = copy.deepcopy(value)
            invalid["action"]["kind"] = kind
            with self.assertRaises(ValidationError):
                validator.validate(invalid)
        invalid = copy.deepcopy(value)
        invalid["outcome_reviews"] = []
        with self.assertRaises(ValidationError):
            validator.validate(invalid)
        invalid = copy.deepcopy(value)
        invalid["memory_updates"] = DemoAgent().decide(messages(state, b"png"))[0]["memory_updates"]
        with self.assertRaises(ValidationError):
            validator.validate(invalid)
        value["action"]["kind"] = "stop"
        value["outcome_reviews"] = []
        validator.validate(value)

    def test_review_references_are_bound_to_the_pending_plan(self):
        state = context()
        validator = Draft202012Validator(schema_for_context(state))
        for field, wrong in (("turn_id", 1), ("expected_revision", 2)):
            value = review(state)
            value["outcome_reviews"][0][field] = wrong
            with self.assertRaises(ValidationError):
                validator.validate(value)
        for field in ("step_index", "before_observation_id", "after_observation_id"):
            value = review(state)
            value["outcome_reviews"][0]["steps"][0][field] = 999
            with self.assertRaises(ValidationError):
                validator.validate(value)

    def test_missing_images_constrain_assessments_to_uncertain(self):
        state = context()
        state["outcome_visual_evidence_ids"] = [7]
        value = review(state)
        validator = Draft202012Validator(schema_for_context(state))
        validator.validate(value)
        value["outcome_reviews"][0]["steps"][0]["assessment"] = "progress_observed"
        with self.assertRaises(ValidationError):
            validator.validate(value)

    def test_supplied_delayed_endpoint_is_preferred_to_an_early_capture(self):
        state = context()
        state["outcome_review"]["next_plan"]["steps"][0]["review_after_observation_ids"] = [6, 7]
        value = review(state)
        validator = Draft202012Validator(schema_for_context(state))
        self.assertFalse(validator.is_valid(value))
        value["outcome_reviews"][0]["steps"][0].update(after_observation_id=7, assessment="progress_observed")
        validator.validate(value)
        state["outcome_visual_evidence_ids"] = [5, 7]
        Draft202012Validator(schema_for_context(state)).validate(value)
        from pokemon_harness.agent.context_budget import validate_visual_reviews
        from pokemon_harness.evaluation import validate_context_gates

        validate_visual_reviews(value, state)
        validate_context_gates(value, state)

    def test_normal_phase_keeps_the_original_schema_and_does_not_mutate_it(self):
        before = copy.deepcopy(DECISION_SCHEMA)
        schema_for_context(context())
        self.assertEqual(DECISION_SCHEMA, before)
        self.assertIs(schema_for_context({}), DECISION_SCHEMA)
        self.assertEqual(request_schema(messages(context(), b"png")), schema_for_context(context()))

    def test_budget_and_http_payload_use_the_same_selected_schema(self):
        model = {"name": "fixture", "context_length": 32768, "max_tokens": 8192}
        state, request, report = fit_request(context(), model, b"png", outcome_images=[
            {"observation_id": 5, "image": b"before", "purpose": "outcome"},
            {"observation_id": 6, "image": b"after", "purpose": "outcome"}])
        value = review(state)
        response = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}).encode()
        with patch("pokemon_harness.agent.lmstudio.urlopen", return_value=io.BytesIO(response)) as http:
            result, trace = LMStudio(model).decide(request)
        payload = json.loads(http.call_args.args[0].data)
        self.assertEqual(payload["response_format"]["json_schema"]["schema"], request_schema(request))
        self.assertEqual(trace["context_budget"]["estimated_input_tokens"], report["estimated_input_tokens"])
        self.assertEqual(assess_request(request, model)["estimated_input_tokens"], report["estimated_input_tokens"])
        self.assertEqual(result, value)

    def test_client_rejects_a_server_that_ignores_the_readonly_schema_with_trace(self):
        request = messages(context(), b"png")
        invalid = DemoAgent().decide(request)[0]
        response = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(invalid)}}]}).encode()
        with patch("pokemon_harness.agent.lmstudio.urlopen", return_value=io.BytesIO(response)):
            with self.assertRaises(InvalidModelResponse) as error:
                LMStudio({"name": "fixture"}).decide(request)
        self.assertIn("response", error.exception.trace)
