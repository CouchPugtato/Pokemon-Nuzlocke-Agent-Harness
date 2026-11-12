import contextlib
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from pokemon_harness.agent.context_budget import (ContextBudgetError, assess_request, budget_settings,
    calibrate_usage, calibration_key, estimate_request, fit_request, validate_visual_reviews)
from pokemon_harness.agent.lmstudio import InvalidModelResponse, LMStudio
from pokemon_harness.agent.prompts import messages
from pokemon_harness.config import load_config
from pokemon_harness.demo import DemoAgent, create_images
from pokemon_harness.emulator.interface import ReplayEmulator
from pokemon_harness.memory.database import Memory
from pokemon_harness.memory.retrieval import ledger_context
from pokemon_harness.runner import run
from pokemon_harness.telemetry.backup import create_backup
from pokemon_harness.telemetry.recorder import write_json
from pokemon_harness.telemetry.restore import restore_backup


def context():
    return {"observation_id": 21, "objective": "Beat gym", "rules": ["No items", "Retire fainted Pokemon"],
            "working_state": {"revision": 8, "uncertainties": ["Party level unknown"]},
            "ledger": {"retired_pokemon": {"Starter": "high"}, "encounter_areas": {"Route 101": ["first"]},
                       "badge_events": [{"key": "stone", "evidence_observation_id": 1}],
                       "recent_or_local_events": []},
            "contradictions": {"open_count": 1, "open": [{"key": "dispute", "revision": 1}]},
            "progress": {"episode_id": 2, "recovery_attempts_remaining": 1},
            "run_review": {"validity": "unverifiable", "authorized_overrides": [{"case_id": 1}]},
            "outcome_review": {"pending_count": 0, "next_plan": None},
            "recent_turns": [], "memory": [], "memory_search_results": []}


class ContextBudgetTests(unittest.TestCase):
    def test_pending_review_keeps_before_and_latest_delayed_endpoint(self):
        state = context()
        state["outcome_review"] = {"pending_count": 1, "next_plan": {
            "turn_id": 1, "expected_revision": 0, "steps": [{"step_index": 1,
                "input_status": "completed", "before_observation_id": 19,
                "after_observation_id": 20, "review_after_observation_ids": [20, 21]}]}}
        frames = [{"observation_id": i, "image": data, "purpose": "outcome"}
                  for i, data in ((19, b"before"), (20, b"early"))]
        model = dict(self.model, context_length=65536)
        _, _, baseline = fit_request(state, model, b"current", outcome_images=frames)
        model["context_length"] = baseline["estimated_input_tokens"] + 8192 + 4096 - 1000
        selected, _, report = fit_request(state, model, b"current", outcome_images=frames)
        self.assertEqual(selected["outcome_visual_evidence_ids"], [19, 21])
        self.assertEqual(report["omitted"]["outcome_images"], 1)

    def test_compacted_continuity_is_optional_and_does_not_displace_authoritative_state(self):
        original = context()
        original["conversation"] = {"recent": [], "continuity": {
            "historical_only": True, "reviews": [{"source_review_id": 1, "steps": "x" * 100000}]}}
        saved = copy.deepcopy(original)
        selected, _, report = fit_request(original, self.model, b"current")
        self.assertEqual(report["omitted"]["continuity_reviews"], 1)
        self.assertEqual(selected["conversation"]["continuity"]["reviews"], [])
        self.assertEqual(original, saved)
        for key in ("ledger", "working_state", "progress", "contradictions", "rules"):
            self.assertEqual(selected[key], original[key])

    def setUp(self):
        self.model = {"name": "fixture", "context_length": 32768, "max_tokens": 32768}

    def test_whole_request_estimate_counts_schema_text_and_images_without_base64_as_text(self):
        policy = budget_settings(self.model)
        small = estimate_request(messages(context(), b"small"), policy)
        large = estimate_request(messages(context(), b"large" * 10000), policy)
        self.assertEqual(small, large)
        self.assertEqual(small["image_count"], 1)
        self.assertGreater(small["estimated_text_tokens"], 1000)

    def test_output_is_clamped_with_reserved_space_and_exact_final_report_consistency(self):
        selected, request, report = fit_request(context(), self.model, b"current")
        self.assertLess(report["effective_output_tokens"], 32768)
        self.assertGreaterEqual(report["effective_output_tokens"], 8192)
        self.assertLessEqual(report["estimated_input_tokens"] + report["effective_output_tokens"] +
                             report["safety_margin_tokens"], 32768)
        self.assertEqual(selected["request_budget"]["estimated_input_tokens"], report["estimated_input_tokens"])
        measured = assess_request(request, self.model)
        self.assertEqual(measured["effective_output_tokens"], report["effective_output_tokens"])
        self.assertTrue(report["approximate"])

    def test_long_history_is_pruned_oldest_first_without_mutating_authoritative_records(self):
        original = context()
        original["recent_turns"] = [{"id": index, "observation": "Earlier observation " * 100} for index in range(80)]
        saved = copy.deepcopy(original)
        selected, _, report = fit_request(original, self.model, b"current")
        self.assertGreater(report["omitted"]["recent_turns"], 0)
        self.assertEqual(original, saved)
        self.assertEqual(selected["recent_turns"], original["recent_turns"][-len(selected["recent_turns"]):])
        for key in ("objective", "rules", "ledger", "memory", "working_state", "contradictions", "progress", "run_review"):
            self.assertEqual(selected[key], original[key])

    def test_current_evidence_survives_omissions_and_missing_step_images_require_uncertainty(self):
        frames = [{"observation_id": index, "image": f"frame {index}".encode(), "purpose": "outcome"}
                  for index in range(1, 21)]
        selected, request, report = fit_request(context(), self.model, b"current", outcome_images=frames)
        self.assertGreater(report["omitted"]["outcome_images"], 0)
        self.assertIn(21, selected["outcome_visual_evidence_ids"])
        self.assertNotIn(1, selected["outcome_visual_evidence_ids"])
        self.assertEqual(request[1]["content"][-1]["image_url"]["url"], "data:image/png;base64,Y3VycmVudA==")
        value = {"outcome_reviews": [{"steps": [{"before_observation_id": 1, "after_observation_id": 21,
                                                  "assessment": "progress_observed"}]}]}
        with self.assertRaisesRegex(ValueError, "remain uncertain"):
            validate_visual_reviews(value, selected)
        value["outcome_reviews"][0]["steps"][0]["assessment"] = "uncertain"
        validate_visual_reviews(value, selected)

    def test_identical_images_have_labels_without_repeated_image_cost(self):
        frame = {"observation_id": 2, "image": b"same", "purpose": "outcome"}
        selected, _, report = fit_request(context(), self.model, b"same", previous_image=b"same", outcome_images=[frame])
        self.assertEqual(report["image_count"], 1)
        self.assertIn(2, selected["outcome_visual_evidence_ids"])

    def test_required_input_overflow_fails_without_dropping_rules_or_eligibility(self):
        original = context()
        original["rules"] = ["A mandatory rule " * 10000]
        with self.assertRaises(ContextBudgetError) as error:
            fit_request(original, self.model, b"current")
        self.assertIn("no controller input", str(error.exception))
        self.assertGreater(error.exception.report["estimated_input_tokens"], error.exception.report["input_token_limit"])
        self.assertEqual(original["ledger"]["retired_pokemon"], {"Starter": "high"})

    def test_older_search_bundles_can_be_omitted_but_latest_search_is_required(self):
        original = context()
        original["memory_search_results"] = [{"request": {"query": "old"}, "results": [{"claim": "x" * 100000}]},
                                               {"request": {"query": "latest"}, "results": []}]
        selected, _, report = fit_request(original, self.model, b"current")
        self.assertEqual(selected["memory_search_results"], original["memory_search_results"][-1:])
        self.assertEqual(report["omitted"]["older_search_bundles"], 1)
        original["memory_search_results"] = original["memory_search_results"][:1]
        with self.assertRaises(ContextBudgetError):
            fit_request(original, self.model, b"current")

    def test_only_nonconsequential_ledger_details_are_optional_and_badges_stay_available(self):
        original = context()
        important = {"type": "faint", "key": "faint", "details": "An important history"}
        original["ledger"]["recent_or_local_events"] = [important, {"type": "progress", "details": "x" * 100000}]
        selected, _, report = fit_request(original, self.model, b"current")
        self.assertEqual(selected["ledger"]["recent_or_local_events"], [important])
        self.assertEqual(selected["ledger"]["badge_events"], original["ledger"]["badge_events"])
        self.assertEqual(report["omitted"]["recent_ledger_details"], 1)
        badge = {"type": "badge", "key": "stone", "corrects_event_key": "", "area": "Gym"}
        self.assertEqual(ledger_context([badge], "", 1)["badge_events"], [badge])

    def test_usage_calibration_only_increases_and_is_separated_by_model_and_policy(self):
        report = assess_request(messages(context(), b"image"), self.model)
        higher = calibrate_usage(report, {"prompt_tokens": report["base_estimated_input_tokens"] * 2})
        self.assertGreater(higher, 2)
        self.assertEqual(calibrate_usage(dict(report, calibration_factor=higher), {"prompt_tokens": 1}), higher)
        for value in (None, "bad", True, -1):
            self.assertEqual(calibrate_usage(report, {"prompt_tokens": value}), 1)
        self.assertNotEqual(calibration_key(self.model), calibration_key(dict(self.model, name="different")))
        self.assertNotEqual(calibration_key(self.model), calibration_key(dict(self.model, context_budget={"image_tokens": 2048})))

    def test_invalid_policy_and_calibration_are_rejected(self):
        for policy in ({"image_tokens": True}, {"safety_margin_tokens": 0}, {"unknown": 1}, {"text_bytes_per_token": 5}):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                budget_settings(dict(self.model, context_budget=policy))
        for factor in (0, True, float("nan"), float("inf")):
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                assess_request([], self.model, factor=factor)


class ContextBudgetIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        config_path = self.root / "config.json"
        write_json(config_path, {"run_dir": "run", "objective": "Test", "model": {"name": "fixture"},
                                 "emulator": {"backend": "replay"}})
        self.config = load_config(config_path)
        self.directory = Path(self.config["run_dir"])
        self.images = create_images(self.root / "fixtures")

    def test_client_rejects_oversized_direct_requests_before_http(self):
        agent = LMStudio(dict(self.config["model"], context_length=4096))
        with patch("pokemon_harness.agent.lmstudio.urlopen") as http:
            with self.assertRaises(ContextBudgetError):
                agent.decide(messages(context(), b"image"))
        http.assert_not_called()

    def test_client_uses_verified_context_and_sends_only_the_effective_output_limit(self):
        agent = LMStudio(self.config["model"])
        agent.loaded_context_length = 65536
        value = DemoAgent().decide(messages(context(), b"image"))[0]
        response = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}
        with patch("pokemon_harness.agent.lmstudio.urlopen") as http:
            http.return_value.__enter__.return_value.read.return_value = json.dumps(response).encode()
            _, trace = agent.decide(messages(context(), b"image"))
        payload = json.loads(http.call_args.args[0].data)
        report = trace["context_budget"]
        self.assertEqual(report["context_length"], 65536)
        self.assertEqual(report["context_source"], "verified_loaded")
        self.assertEqual(report["configured_context_length"], 32768)
        self.assertEqual(payload["max_tokens"], report["effective_output_tokens"])
        self.assertNotIn("context_budget", payload)
        self.assertNotIn("messages", trace["request_settings"])

    def test_budget_failure_archives_diagnostics_without_model_or_controller_input(self):
        self.config["model"]["context_length"] = 4096
        emulator = ReplayEmulator(self.images)

        class Agent:
            def decide(self, request):
                raise AssertionError("generation should not start")

        with self.assertRaises(ContextBudgetError):
            run(self.config, emulator, Agent(), turns=1)
        self.assertEqual(emulator.actions, [])
        report = json.loads(next((self.directory / "turns").glob("*.budget.json")).read_text())
        self.assertEqual(report["context_length"], 4096)
        with contextlib.closing(Memory(self.directory / "memory.sqlite")) as memory:
            self.assertEqual(memory.recent_turns(), [])

    def test_calibration_is_saved_from_failed_generation_and_survives_restore(self):
        agent = LMStudio(self.config["model"])
        response = {"usage": {"prompt_tokens": 20000, "completion_tokens": 1},
                    "choices": [{"finish_reason": "length", "message": {"content": "{"}}]}
        with patch("pokemon_harness.agent.lmstudio.urlopen") as http:
            http.return_value.__enter__.return_value.read.return_value = json.dumps(response).encode()
            with self.assertRaises(InvalidModelResponse):
                run(self.config, ReplayEmulator(self.images), agent, turns=1)
        key = calibration_key(self.config["model"])
        with contextlib.closing(Memory(self.directory / "memory.sqlite")) as memory:
            factor = memory.budgets.factor(key)
            self.assertGreater(factor, 1)
            memory.budgets.record(key, {"calibration_suggestion": 1}, 1)
            self.assertEqual(memory.budgets.factor(key), factor)
        backup = create_backup(self.directory, self.root / "backups")
        target = self.root / "restored"
        restore_backup(backup, target)
        with contextlib.closing(Memory(target / "memory.sqlite")) as memory:
            self.assertEqual(memory.budgets.factor(key), factor)

    def test_final_budget_trace_matches_supplied_context_for_a_successful_run(self):
        run(self.config, ReplayEmulator(self.images), DemoAgent(), turns=1)
        report = json.loads(next((self.directory / "turns").glob("*.budget.json")).read_text())
        selected = json.loads(next((self.directory / "turns").glob("*.context.json")).read_text())
        self.assertEqual(selected["request_budget"]["estimated_input_tokens"], report["estimated_input_tokens"])
        self.assertLessEqual(report["estimated_input_tokens"], report["input_token_limit"])
        self.assertEqual(selected["outcome_visual_evidence_ids"], [1])

    def test_oversized_required_review_stops_before_another_request_or_input(self):
        paths = []
        for index in range(21):
            path = self.root / f"frame-{index}.png"
            Image.new("RGB", (32, 32), (index * 10,) * 3).save(path)
            paths.append(path)
        self.config["max_sequence_length"] = 20
        self.config["sequence_delay_ms"] = 0
        requests = []

        class Agent:
            def decide(self, request):
                requests.append(json.loads(request[1]["content"][0]["text"]))
                value, trace = DemoAgent().decide(request)
                if len(requests) == 1:
                    value["additional_actions"] = [dict(value["action"])] * 19
                else:
                    for review in value["outcome_reviews"]:
                        for step in review["steps"]:
                            step["assessment"] = "progress_observed"
                return value, trace

        emulator = ReplayEmulator(paths)
        with self.assertRaises(ContextBudgetError):
            run(self.config, emulator, Agent(), turns=2)
        self.assertEqual(len(emulator.actions), 20)
        self.assertEqual(len(requests), 1)
        with contextlib.closing(Memory(self.directory / "memory.sqlite")) as memory:
            self.assertEqual(len(memory.recent_turns()), 1)
            self.assertEqual(memory.outcomes.pending_count(), 1)
