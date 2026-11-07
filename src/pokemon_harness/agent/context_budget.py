"""conservative whole request budgeting without claiming exact token counts"""

import copy
import hashlib
import json
import math

from .prompts import messages
from .schemas import DECISION_SCHEMA


DEFAULTS = {"output_reserve_tokens": 8192, "safety_margin_tokens": 4096,
            "image_tokens": 1536, "text_bytes_per_token": 3}


class ContextBudgetError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__(
            f"Context budget cannot fit required input: estimated {report['estimated_input_tokens']} tokens, "
            f"input allowance {report['input_token_limit']}, context {report['context_length']}, "
            f"output reserve {report['output_reserve_tokens']}, safety {report['safety_margin_tokens']}. "
            "Increase --context-length or reduce the request size; no controller input was sent")


def budget_settings(model):
    supplied = model.get("context_budget", {})
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
        raise ValueError("model.context_budget must contain only supported budget settings")
    settings = dict(DEFAULTS, **supplied)
    for key, lower, upper in (("output_reserve_tokens", 128, 131072), ("safety_margin_tokens", 256, 131072),
                              ("image_tokens", 64, 32768), ("text_bytes_per_token", 1, 4)):
        if type(settings[key]) is not int or not lower <= settings[key] <= upper:
            raise ValueError(f"model.context_budget.{key} must be an integer between {lower} and {upper}")
    return settings


def calibration_key(model):
    payload = {"name": model["name"], "base_url": model.get("base_url", "http://127.0.0.1:1234/v1").rstrip("/"),
               "policy": budget_settings(model), "schema": DECISION_SCHEMA}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def estimate_request(request, policy):
    text_bytes = len(json.dumps(DECISION_SCHEMA, ensure_ascii=False).encode("utf-8"))
    images = 0
    for message in request:
        text_bytes += len(str(message["role"]).encode("utf-8"))
        content = message["content"]
        if isinstance(content, str):
            text_bytes += len(content.encode("utf-8"))
        else:
            for item in content:
                if item["type"] == "text":
                    text_bytes += len(item["text"].encode("utf-8"))
                elif item["type"] == "image_url":
                    images += 1
                else:
                    raise ValueError("Unsupported message content for context budgeting")
    text_tokens = math.ceil(text_bytes / policy["text_bytes_per_token"])
    overhead = 64 * len(request) + 256
    return {"estimated_text_tokens": text_tokens, "image_count": images,
            "estimated_image_tokens": images * policy["image_tokens"], "message_overhead_tokens": overhead,
            "base_estimated_input_tokens": text_tokens + images * policy["image_tokens"] + overhead}


def assess_request(request, model, context_length=None, factor=1.0):
    if not isinstance(factor, (int, float)) or isinstance(factor, bool) or not math.isfinite(factor) or factor < 1:
        raise ValueError("Context calibration factor must be finite and at least one")
    policy = budget_settings(model)
    length = model.get("context_length", 32768) if context_length is None else context_length
    requested = model.get("max_tokens", 32768)
    if type(length) is not int or length < 1024 or type(requested) is not int or requested < 1:
        raise ValueError("Context length and output limit must be valid positive token counts")
    reserve = min(requested, policy["output_reserve_tokens"])
    measured = estimate_request(request, policy)
    estimated = math.ceil(measured["base_estimated_input_tokens"] * factor)
    available = length - estimated - policy["safety_margin_tokens"]
    report = dict(measured, approximate=True, estimator="utf8 text and configured image allowance",
                  context_length=length, configured_context_length=model.get("context_length", 32768),
                  context_source="verified_loaded" if context_length is not None else "configured_unverified",
                  calibration_factor=factor, estimated_input_tokens=estimated,
                  input_token_limit=length - reserve - policy["safety_margin_tokens"],
                  requested_output_tokens=requested, output_reserve_tokens=reserve,
                  effective_output_tokens=max(0, min(requested, available)),
                  safety_margin_tokens=policy["safety_margin_tokens"])
    if available < reserve:
        raise ContextBudgetError(report)
    return report


def calibrate_usage(report, usage):
    actual = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    if type(actual) is int and actual > 0:
        return max(report["calibration_factor"], actual / report["base_estimated_input_tokens"] * 1.1)
    return report["calibration_factor"]


def fit_request(context, model, current_image, previous_image=None, sequence_images=None,
                outcome_images=None, context_length=None, factor=1.0):
    selected = copy.deepcopy(context)
    selected["request_budget"] = {"approximate": True, "omitted": {}}
    sequence = list(sequence_images or [])
    evidence = list(outcome_images or [])
    omissions = {}

    def omitted(name, count=1):
        omissions[name] = omissions.get(name, 0) + count

    def trim():
        if selected.get("recent_turns"):
            selected["recent_turns"].pop(0)
            omitted("recent_turns")
        elif any(event.get("type") == "progress" for event in selected.get("ledger", {}).get("recent_or_local_events", [])):
            entries = selected["ledger"]["recent_or_local_events"]
            index = next(index for index, event in enumerate(entries) if event.get("type") == "progress")
            entries.pop(index)
            omitted("recent_ledger_details")
        elif len(selected.get("memory_search_results", [])) > 1:
            selected["memory_search_results"].pop(0)
            omitted("older_search_bundles")
        elif sequence:
            sequence.pop(0)
            omitted("sequence_images")
        elif any(frame.get("purpose") == "loop" for frame in evidence):
            index = next(index for index, frame in enumerate(evidence) if frame.get("purpose") == "loop")
            evidence.pop(index)
            omitted("loop_images")
        elif len(evidence) > 2:
            evidence.pop(0)
            omitted("outcome_images")
        else:
            return False
        return True

    while True:
        visible = {selected["observation_id"], *(frame["observation_id"] for frame in evidence),
                   *(frame["observation_id"] for frame in sequence)}
        selected["outcome_visual_evidence_ids"] = sorted(visible)
        selected["request_budget"]["omitted"] = dict(omissions)
        request = messages(selected, current_image, previous_image, sequence, evidence)
        try:
            report = assess_request(request, model, context_length, factor)
        except ContextBudgetError as exc:
            if trim():
                continue
            if previous_image is not None:
                previous_image = None
                omitted("previous_images")
                continue
            if evidence:
                evidence.pop(0)
                omitted("outcome_images")
                continue
            exc.report["omitted"] = dict(omissions)
            raise
        fitted = True
        for _ in range(8):
            metadata = {key: report[key] for key in ("approximate", "context_length", "estimated_input_tokens",
                        "effective_output_tokens", "safety_margin_tokens", "calibration_factor")}
            metadata["omitted"] = dict(omissions)
            if selected["request_budget"] == metadata:
                break
            selected["request_budget"] = metadata
            request = messages(selected, current_image, previous_image, sequence, evidence)
            try:
                report = assess_request(request, model, context_length, factor)
            except ContextBudgetError as exc:
                if trim():
                    fitted = False
                elif previous_image is not None:
                    previous_image = None
                    omitted("previous_images")
                    fitted = False
                elif evidence:
                    evidence.pop(0)
                    omitted("outcome_images")
                    fitted = False
                else:
                    exc.report["omitted"] = dict(omissions)
                    raise
                break
        else:
            raise ValueError("Context budget metadata did not stabilize before generation")
        if not fitted:
            continue
        report["omitted"] = dict(omissions)
        return selected, request, report


def validate_visual_reviews(decision, context):
    visible = set(context["outcome_visual_evidence_ids"])
    for review in decision["outcome_reviews"]:
        for step in review["steps"]:
            if step["assessment"] != "uncertain" and not {
                    step["before_observation_id"], step["after_observation_id"]}.issubset(visible):
                raise ValueError("Outcome review without both supplied screenshots must remain uncertain")
