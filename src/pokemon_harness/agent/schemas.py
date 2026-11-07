"""one output contract for generation and validation"""

from jsonschema import Draft202012Validator

BUTTONS = ["A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT", "L", "R"]
MEMORY_CATEGORIES = ["working", "objective", "world", "party", "lesson"]


def text(limit=600):
    return {"type": "string", "maxLength": limit}


def obj(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


MEMORY_SEARCH_SCHEMA = obj({
    "query": text(200),
    "scope": {"type": "string", "enum": ["all", "facts", "events"]},
    "category": {"type": "string", "enum": ["", *MEMORY_CATEGORIES]},
    "location": text(120),
    "include_history": {"type": "boolean"},
    "offset": {"type": "integer", "minimum": 0, "maximum": 1000000},
    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
})

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
    if (action["kind"] == "search_memory") != (search is not None):
        raise ValueError("Only search_memory actions must supply a memory_search request")
    if search is not None:
        if (value["memory_updates"] or value["events"] or value["working_state_update"] is not None
                or value["outcome_reviews"]):
            raise ValueError("Memory searches are read-only; return empty memory_updates and events and null working_state_update")
        if search["category"] and search["scope"] != "facts":
            raise ValueError("A memory category filter requires scope=facts")
    if action["kind"] == "review_outcomes":
        if not value["outcome_reviews"]:
            raise ValueError("review_outcomes requires an outcome review")
        if value["memory_updates"] or value["events"] or value["working_state_update"] is not None:
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
