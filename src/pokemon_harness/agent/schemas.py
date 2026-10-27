"""One output contract for generation and application-side validation."""

from jsonschema import Draft202012Validator

BUTTONS = ["A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT", "L", "R"]


def text(limit=600):
    return {"type": "string", "maxLength": limit}


def obj(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


DECISION_SCHEMA = obj({
    "observation_id": {"type": "integer", "minimum": 1},
    "scene": {"type": "string", "enum": ["overworld", "dialogue", "menu", "battle", "transition", "unknown"]},
    "location": text(120),
    "observation": text(1200),
    "action": obj({
        "kind": {"type": "string", "enum": ["press", "wait", "stop"]},
        "buttons": {"type": "array", "items": {"type": "string", "enum": BUTTONS},
                    "uniqueItems": True, "maxItems": 3},
        "hold_frames": {"type": "integer", "minimum": 1, "maximum": 120},
        "settle_frames": {"type": "integer", "minimum": 0, "maximum": 120},
    }),
    "expected_result": text(),
    "memory_updates": {"type": "array", "maxItems": 12, "items": obj({
        "category": {"type": "string", "enum": ["working", "objective", "world", "party", "lesson"]},
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


def validate_decision(value, observation_id, max_frames=60):
    Draft202012Validator(DECISION_SCHEMA).validate(value)
    if value["observation_id"] != observation_id:
        raise ValueError("Decision refers to a stale observation")
    action = value["action"]
    if max(action["hold_frames"], action["settle_frames"]) > max_frames:
        raise ValueError(f"Action exceeds configured limit of {max_frames} frames")
    if (action["kind"] == "press") != bool(action["buttons"]):
        raise ValueError("Only press actions may contain buttons; press needs a button")
    for pair in ({"UP", "DOWN"}, {"LEFT", "RIGHT"}):
        if pair.issubset(action["buttons"]):
            raise ValueError("Opposite directions cannot be pressed together")
    for event in value["events"]:
        if (event["type"] == "correction") != bool(event["corrects_event_key"]):
            raise ValueError("Only corrections must reference an existing event key")
    return value
