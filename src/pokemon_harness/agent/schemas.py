"""one output contract for generation and validation"""

from jsonschema import Draft202012Validator
import copy
import json

BUTTONS = ["A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT", "L", "R"]
MEMORY_CATEGORIES = ["working", "objective", "world", "party", "lesson"]


def text(limit=600):
    return {"type": "string", "maxLength": limit}


def obj(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


MEMORY_SEARCH_SCHEMA = obj({
    "query": text(200),
    "scope": {"type": "string", "enum": ["all", "facts", "events", "contradictions", "turns"]},
    "category": {"type": "string", "enum": ["", *MEMORY_CATEGORIES]},
    "location": text(120),
    "include_history": {"type": "boolean"},
    "offset": {"type": "integer", "minimum": 0, "maximum": 1000000},
    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
})
MEMORY_SEARCH_SCHEMA["properties"]["observation_ids"] = {
    "type": "array", "maxItems": 4, "uniqueItems": True,
    "items": {"type": "integer", "minimum": 1}}

EVIDENCE_IDS = {"type": "array", "items": {"type": "integer", "minimum": 1},
                "uniqueItems": True, "maxItems": 8}
EVIDENCE_CLAIM = obj({
    "claim": {**text(), "minLength": 1},
    "basis": {"type": "string", "enum": ["observed", "inferred", "unknown"]},
    "evidence_observation_ids": EVIDENCE_IDS,
})
SUBGOAL_DEFINITION = obj({
    key: {**text(), "minLength": 1}
    for key in ("description", "objective_relation", "completion_condition", "abandonment_condition")
})
WORKING_STATE_UPDATE = obj({
    "expected_revision": {"type": "integer", "minimum": 0},
    "location": {"anyOf": [{"type": "null"}, EVIDENCE_CLAIM]},
    "uncertainties": {"type": "array", "maxItems": 8, "items": EVIDENCE_CLAIM},
    "next_check": text(),
    "subgoal_change": {"anyOf": [{"type": "null"}, obj({
        "operation": {"type": "string", "enum": ["start", "revise", "complete", "abandon"]},
        "id": {**text(120), "minLength": 1},
        "definition": {"anyOf": [{"type": "null"}, SUBGOAL_DEFINITION]},
        "reason": {**text(), "minLength": 1},
        "evidence_observation_ids": EVIDENCE_IDS,
    })]},
})


MOVE_SCHEMA = obj({
    "kind": {"type": "string", "enum": ["press", "wait"]},
    "buttons": {"type": "array", "items": {"type": "string", "enum": BUTTONS},
                "uniqueItems": True, "maxItems": 3},
    "hold_frames": {"type": "integer", "minimum": 1, "maximum": 120},
    "settle_frames": {"type": "integer", "minimum": 0, "maximum": 120},
})

OUTCOME_REVIEW = obj({
    "turn_id": {"type": "integer", "minimum": 1},
    "expected_revision": {"type": "integer", "minimum": 0},
    "steps": {"type": "array", "minItems": 1, "maxItems": 20, "items": obj({
        "step_index": {"type": "integer", "minimum": 1, "maximum": 20},
        "before_observation_id": {"type": "integer", "minimum": 1},
        "after_observation_id": {"type": "integer", "minimum": 1},
        "assessment": {"type": "string", "enum": [
            "progress_observed", "no_progress_observed", "uncertain"]},
        "explanation": {**text(), "minLength": 1},
    })},
})

FACT_REFERENCE = obj({
    "category": {"type": "string", "enum": MEMORY_CATEGORIES},
    "key": {**text(120), "minLength": 1},
})
FACT_IDS = {"type": "array", "maxItems": 8, "uniqueItems": True,
            "items": {"type": "integer", "minimum": 1}}
EVENT_KEYS = {"type": "array", "maxItems": 8, "uniqueItems": True,
              "items": {**text(120), "minLength": 1}}
MEMORY_DEPENDENCIES = obj({
    "facts": {"type": "array", "maxItems": 12, "items": obj({
        "fact": FACT_REFERENCE, "depends_on_fact_ids": FACT_IDS, "depends_on_event_keys": EVENT_KEYS,
    })},
    "working_state": {"anyOf": [{"type": "null"}, FACT_IDS]},
    "working_state_events": {"anyOf": [{"type": "null"}, EVENT_KEYS]},
})
CONTRADICTION_UPDATE = obj({
    "operation": {"type": "string", "enum": ["open", "resolve"]},
    "key": {**text(120), "minLength": 1},
    "expected_revision": {"type": "integer", "minimum": 0},
    "reason": {**text(), "minLength": 1},
    "evidence_observation_ids": EVIDENCE_IDS,
    "fact_ids": FACT_IDS,
    "event_keys": EVENT_KEYS,
    "replacement_facts": {"type": "array", "maxItems": 8, "items": FACT_REFERENCE},
    "correction_event_keys": EVENT_KEYS,
})


DECISION_SCHEMA = obj({
    "observation_id": {"type": "integer", "minimum": 1},
    "scene": {"type": "string", "enum": ["overworld", "dialogue", "menu", "battle", "transition", "unknown"]},
    "location": text(120),
    "observation": text(1200),
    "decision_reason": {**text(4000), "description":
        "Explanation of the chosen action sequence, search, or stop. Empty is allowed; use only as much detail as helpful."},
    "action": obj({
        "kind": {"type": "string", "enum": ["press", "wait", "stop", "search_memory", "review_outcomes"]},
        "buttons": {"type": "array", "items": {"type": "string", "enum": BUTTONS},
                    "uniqueItems": True, "maxItems": 3},
        "hold_frames": {"type": "integer", "minimum": 1, "maximum": 120},
        "settle_frames": {"type": "integer", "minimum": 0, "maximum": 120},
    }),
    "expected_result": text(),
    "additional_actions": {"type": "array", "items": MOVE_SCHEMA, "maxItems": 19},
    "memory_search": {"anyOf": [{"type": "null"}, MEMORY_SEARCH_SCHEMA]},
    "working_state_update": {"anyOf": [{"type": "null"}, WORKING_STATE_UPDATE]},
    "outcome_reviews": {"type": "array", "maxItems": 1, "items": OUTCOME_REVIEW},
    "memory_dependencies": MEMORY_DEPENDENCIES,
    "contradiction_updates": {"type": "array", "maxItems": 8, "items": CONTRADICTION_UPDATE},
    "strategy": {"anyOf": [{"type": "null"}, obj({
        "kind": {"type": "string", "enum": ["advance_dialogue", "inspect_menu", "move", "interact",
                                             "battle", "capture", "flee", "wait", "other"]},
        "target": {**text(120), "minLength": 1},
        "intended_effect": {**text(), "minLength": 1},
    })]},
    "recovery_plan": {"anyOf": [{"type": "null"}, obj({
        "episode_id": {"type": "integer", "minimum": 1},
        "explanation": {**text(), "minLength": 1},
    })]},
    "memory_updates": {"type": "array", "maxItems": 12, "items": obj({
        "category": {"type": "string", "enum": MEMORY_CATEGORIES},
        "key": {"type": "string", "minLength": 1, "maxLength": 120},
        "claim": text(),
        "location": text(120),
        "basis": {"type": "string", "enum": ["observed", "inferred"]},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "status": {"type": "string", "enum": ["active", "resolved", "uncertain"]},
    })},
    "events": {"type": "array", "maxItems": 8, "items": obj({
        "key": {"type": "string", "minLength": 1, "maxLength": 120},
        "type": {"type": "string", "enum": ["encounter", "capture", "faint", "badge", "progress", "correction"]},
        "subject": text(120),
        "area": text(120),
        "details": text(),
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "corrects_event_key": text(120),
    })},
})


def restricted_schema():
    schema = copy.deepcopy(DECISION_SCHEMA)
    fields = schema["properties"]
    empty = {"type": "array", "maxItems": 0}
    for key in ("additional_actions", "memory_updates", "events", "contradiction_updates"):
        fields[key] = copy.deepcopy(empty)
    for key in ("memory_search", "working_state_update", "strategy", "recovery_plan"):
        fields[key] = {"type": "null"}
    fields["memory_dependencies"] = obj({"facts": copy.deepcopy(empty), "working_state": {"type": "null"},
                                         "working_state_events": {"type": "null"}})
    return schema


def recovery_schema(context):
    progress = context["progress"]
    base = restricted_schema()
    base["properties"]["outcome_reviews"] = {"type": "array", "maxItems": 0}
    stop = copy.deepcopy(base)
    stop["properties"]["action"] = obj({"kind": {"type": "string", "enum": ["stop"]},
        "buttons": {"type": "array", "maxItems": 0}, "hold_frames": {"type": "integer", "enum": [1]},
        "settle_frames": {"type": "integer", "enum": [0]}})
    choices = []
    if progress.get("recovery_attempts_remaining", 0):
        forbidden = [*progress.get("recovery_signatures", []), progress.get("last_input_signature")]

        def permitted(kind, buttons):
            signature = json.dumps(((kind, tuple(sorted(buttons))),))
            return all(value is None or signature != json.dumps(value) for value in forbidden)

        for kind in ("press", "wait"):
            buttons = [[button] for button in BUTTONS if permitted(kind, [button])] if kind == "press" else ([[]] if permitted(kind, []) else [])
            if not buttons:
                continue
            test = copy.deepcopy(base)
            fields = test["properties"]
            maximum = context.get("action_limit_frames", 60)
            fields["action"] = obj({"kind": {"type": "string", "enum": [kind]}, "buttons": {"type": "array", "enum": buttons},
                "hold_frames": {"type": "integer", "minimum": 1, "maximum": maximum},
                "settle_frames": {"type": "integer", "minimum": 0, "maximum": maximum}})
            if kind == "press":
                fields["scene"]["enum"] = ["overworld", "dialogue", "menu"]
            fields["strategy"] = copy.deepcopy(DECISION_SCHEMA["properties"]["strategy"]["anyOf"][1])
            fields["strategy"]["properties"]["kind"]["enum"] = ["inspect_menu", "move", "interact", "advance_dialogue", "wait"]
            fields["recovery_plan"] = copy.deepcopy(DECISION_SCHEMA["properties"]["recovery_plan"]["anyOf"][1])
            fields["recovery_plan"]["properties"]["episode_id"]["enum"] = [progress["episode_id"]]
            fields["recovery_plan"]["properties"]["explanation"]["minLength"] = 1
            choices.append(test)
    if context.get("memory_available", {}).get("searches_remaining", 0):
        search = copy.deepcopy(stop)
        search["properties"]["action"]["properties"]["kind"]["enum"] = ["search_memory"]
        search["properties"]["memory_search"] = copy.deepcopy(MEMORY_SEARCH_SCHEMA)
        search["properties"]["memory_search"]["properties"]["scope"]["enum"] = ["all"]
        search["properties"]["memory_search"]["properties"]["category"]["enum"] = [""]
        choices.append(search)
    return {"anyOf": [*choices, stop]}


def schema_for_context(context):
    pending = context.get("outcome_review", {})
    if not pending.get("pending_count"):
        return recovery_schema(context) if context.get("progress", {}).get("requires_recovery") else DECISION_SCHEMA
    plan = pending.get("next_plan")
    if not plan:
        raise ValueError("Pending outcome review has no available plan")
    review = restricted_schema()
    fields = review["properties"]
    empty = {"type": "array", "maxItems": 0}
    fields["action"] = obj({"kind": {"type": "string", "enum": ["review_outcomes"]},
                            "buttons": copy.deepcopy(empty), "hold_frames": {"type": "integer", "enum": [1]},
                            "settle_frames": {"type": "integer", "enum": [0]}})
    item = fields["outcome_reviews"]["items"]
    item["properties"]["turn_id"]["enum"] = [plan["turn_id"]]
    item["properties"]["expected_revision"]["enum"] = [plan["expected_revision"]]
    steps = [step for step in plan["steps"] if step["input_status"] in {"completed", "executing", "unknown"}]
    if not steps:
        raise ValueError("Pending outcome review has no attempted steps")
    item["properties"]["steps"]["minItems"] = len(steps)
    item["properties"]["steps"]["maxItems"] = len(steps)
    step_fields = item["properties"]["steps"]["items"]["properties"]
    step_fields["step_index"]["enum"] = [step["step_index"] for step in steps]
    step_fields["before_observation_id"]["enum"] = sorted({step["before_observation_id"] for step in steps})
    visible = context.get("outcome_visual_evidence_ids")
    endpoints = []
    for step in steps:
        candidates = step.get("review_after_observation_ids", [step["after_observation_id"] or context["observation_id"]])
        supplied = [value for value in candidates if visible is None or value in visible]
        endpoints.append(max(supplied or candidates))
    step_fields["after_observation_id"]["enum"] = sorted(set(endpoints))
    if visible is not None and any(step["after_observation_id"] is None or not {
            step["before_observation_id"], after}.issubset(visible) for step, after in zip(steps, endpoints)):
        step_fields["assessment"]["enum"] = ["uncertain"]
    fields["outcome_reviews"]["minItems"] = 1
    stop = copy.deepcopy(review)
    stop["properties"]["action"]["properties"]["kind"]["enum"] = ["stop"]
    stop["properties"]["outcome_reviews"] = copy.deepcopy(empty)
    return {"anyOf": [review, stop]}


def request_schema(request):
    for message in request:
        if message["role"] != "user" or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if block.get("type") == "text":
                try:
                    context = json.loads(block["text"])
                except (ValueError, TypeError):
                    continue
                if isinstance(context, dict):
                    return schema_for_context(context)
    return DECISION_SCHEMA


def validate_decision(value, observation_id, max_frames=60, max_sequence_length=5):
    Draft202012Validator(DECISION_SCHEMA).validate(value)
    if value["observation_id"] != observation_id:
        raise ValueError("Decision refers to a stale observation")
    actions = [value["action"], *value["additional_actions"]]
    if len(actions) > max_sequence_length:
        raise ValueError(f"Action sequence exceeds configured limit of {max_sequence_length} moves")
    if value["action"]["kind"] in {"stop", "search_memory", "review_outcomes"} and value["additional_actions"]:
        raise ValueError("Stop, search_memory and review_outcomes decisions cannot queue controller actions")
    for action in actions:
        validate_action(action, max_frames)
    action = value["action"]
    if action["kind"] in {"press", "wait"} and value["strategy"] is None:
        raise ValueError("Controller plans require a strategy with target and intended effect")
    if action["kind"] not in {"press", "wait"} and value["recovery_plan"] is not None:
        raise ValueError("Only controller plans may propose a recovery test")
    search = value["memory_search"]
    dependency_write = (value["memory_dependencies"]["facts"] or value["memory_dependencies"]["working_state"] is not None
                        or value["memory_dependencies"]["working_state_events"] is not None)
    if (action["kind"] == "search_memory") != (search is not None):
        raise ValueError("Only search_memory actions must supply a memory_search request")
    if search is not None:
        if (value["memory_updates"] or value["events"] or value["working_state_update"] is not None
                or value["outcome_reviews"] or value["contradiction_updates"] or dependency_write):
            raise ValueError("Memory searches are read-only; return empty memory_updates and events and null working_state_update")
        if search["category"] and search["scope"] != "facts":
            raise ValueError("A memory category filter requires scope=facts")
    if action["kind"] == "review_outcomes":
        if not value["outcome_reviews"]:
            raise ValueError("review_outcomes requires an outcome review")
        if (value["memory_updates"] or value["events"] or value["working_state_update"] is not None
                or value["contradiction_updates"] or dependency_write):
            raise ValueError("review_outcomes only writes outcome reviews")
    for event in value["events"]:
        if (event["type"] == "correction") != bool(event["corrects_event_key"]):
            raise ValueError("Only corrections must reference an existing event key")
    return value


def validate_action(action, max_frames):
    if max(action["hold_frames"], action["settle_frames"]) > max_frames:
        raise ValueError(f"Action exceeds configured limit of {max_frames} frames")
    if (action["kind"] == "press") != bool(action["buttons"]):
        raise ValueError("Only press actions may contain buttons; press needs a button")
    for pair in ({"UP", "DOWN"}, {"LEFT", "RIGHT"}):
        if pair.issubset(action["buttons"]):
            raise ValueError("Opposite directions cannot be pressed together")
