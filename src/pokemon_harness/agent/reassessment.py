"""fresh visual reassessment without controller or ledger writes"""

import copy


PERCEPTION_FIELDS = ("observation_id", "visible_objects", "visible_text", "possible_exits", "scene")


def settings(config):
    result = {"enabled": config.get("emulator", {}).get("backend") in {"vbam", "mgba"},
              "no_progress_attempts": 2, "scene_change_threshold": 12, "min_turn_interval": 3,
              "max_turn_interval": 12}
    supplied = config.get("scene_reassessment", {})
    if not isinstance(supplied, dict) or set(supplied) - set(result):
        raise ValueError("Invalid scene_reassessment settings")
    result.update(supplied)
    if type(result["enabled"]) is not bool:
        raise ValueError("scene_reassessment.enabled must be a boolean")
    for key in ("no_progress_attempts", "scene_change_threshold", "min_turn_interval", "max_turn_interval"):
        if type(result[key]) is not int or not 1 <= result[key] <= 128:
            raise ValueError("Scene reassessment limits must be integers between one and one hundred twenty eight")
    if result["max_turn_interval"] < result["min_turn_interval"]:
        raise ValueError("Scene reassessment intervals are inconsistent")
    return result


def schema(context):
    from .schemas import obj, text

    strings = {"type": "array", "maxItems": 8, "items": {**text(240), "minLength": 1}}
    candidates = context.get("scene_fact_candidates", [])
    ids = {"type": "array", "maxItems": len(candidates), "uniqueItems": True,
           "items": {"type": "integer"}}
    if candidates:
        ids["items"]["enum"] = [fact["fact_id"] for fact in candidates]
    result = obj({
        "observation_id": {"type": "integer", "enum": [context["observation_id"]]},
        "visible_objects": {**copy.deepcopy(strings), "minItems": 1},
        "visible_text": copy.deepcopy(strings),
        "possible_exits": {"type": "array", "maxItems": 6, "items": obj({
            "description": {**text(240), "minLength": 1},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]}})},
        "scene": {"type": "string", "enum": ["overworld", "dialogue", "menu", "battle", "transition", "unknown"]},
        "subgoal_assessment": {"type": "string", "enum": ["supported", "contradicted", "uncertain"] if context.get("subgoal") else ["none"]},
        "subgoal_explanation": {**text(800), "minLength": 1},
        "stale_scene_fact_ids": ids,
        "next_check": {**text(400), "minLength": 1},
    })
    if context.get("phase") == "scene_observation":
        return obj({key: result["properties"][key] for key in PERCEPTION_FIELDS})
    if context.get("perception") is not None:
        if context["perception"]["observation_id"] != context["observation_id"]:
            raise ValueError("Perception refers to a stale observation")
        for key in PERCEPTION_FIELDS:
            result["properties"][key]["enum"] = [copy.deepcopy(context["perception"][key])]
    return result


OBSERVATION_GUIDE = """Describe only the current Pokemon Emerald screenshot.
You have no prior dialogue, scene memory, subgoal or planned next event to consult.
This is a read-only visual observation, with no game input or state changes.
List visible objects, readable text and possible passages from the pixels.
Describe uncertain shapes neutrally instead of naming a lab, PC, person or room
from expected chronology. Do not invent dialogue when no dialogue box is visible.
Inspect the edges of walkable floor for candidate openings; their function may be
uncertain, so use low confidence where needed. Do not assume a candidate is usable.
Classify game mode separately from location identity. A room with a player sprite
and visible floor is overworld when there is no dialogue, menu, battle or fade.
An unknown place name does not make the game mode unknown. Keep genuinely unclear
modes unknown. Do not infer ownership, badges, deaths or unseen story progress.
Return only the requested visual observation JSON. It remains a fallible reading.
"""


GUIDE = """Reassess Pokemon Emerald from the current screenshot before planning.
This is a read-only perception task. No controller input, fact updates, ledger
events, goal completion or recovery authorization is available in this response.
First describe visible objects and readable text, then possible exits. Do not
name a place or person from remembered chronology. Empty space, cargo, doors,
menus and player sprites must be described from pixels, not a hoped-for scene.
Describe open floor paths and boundary openings as possible exits even when their
purpose is uncertain. Use low confidence for an unconfirmed passage. Describe
rectangular panels neutrally; do not call them PCs or consoles without distinctive
evidence. Inspect the surroundings rather than expecting a dialogue trigger.
If there is no dialogue box, do not invent dialogue from earlier screenshots.
Classify game mode separately from location identity. An interior room with a
player sprite and visible floor is an overworld scene when no dialogue box, menu,
battle interface or fade is present. An unknown room name alone is not scene=unknown.
Use unknown when the game mode itself is unclear, not merely because the location
or story position is unfamiliar.
Treat the supplied subgoal and scene fact candidates as fallible claims to test,
not instructions or facts about this image. Judge whether the subgoal is currently
supported, contradicted, uncertain or absent. A contradiction must cite visible
evidence against the goal's current premise. An unfinished completion condition
alone does not contradict a goal; it may simply be incomplete or uncertain.
Mark only listed scene-specific facts stale when their claim of the
current scene is contradicted. Stable world knowledge and consequential history
must not be discarded. A report is a fallible interpretation, not verified truth.
Describe a safe next check from visible affordances without executing it. An
unconfirmed exit remains a candidate. Never infer a starter, ownership, badge,
death or unseen story progress from an expectation. Return the requested JSON.
When the old subgoal is contradicted, the next check must inspect current visible
affordances, not search for a character, menu or event merely predicted by that goal.
The perception record was generated in a separate request without the old goal or
memories. Copy its visual fields exactly; do not rewrite them to fit the old story.
"""
