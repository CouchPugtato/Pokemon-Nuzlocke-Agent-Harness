"""Bounded context selection. Durable source records are never summarized away."""

import re


def words(value):
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def recent_turn_context(turns):
    """Recall observations and input outcomes without feeding rationales back in.

    Full decisions, rationale, and controller reports remain in SQLite and traces.
    Predicted results and speculative rationale are deliberately not evidence.
    """
    result = []
    for turn in turns:
        decision = turn["decision"]
        steps = []
        for step in turn.get("steps", []):
            execution = step.get("execution") or {}
            item = {key: step.get(key) for key in ("step_index", "status", "observation_id")}
            item["action"] = execution.get("effective_action", step["action"])
            if step.get("error"):
                item["error"] = step["error"][:200]
            steps.append(item)
        result.append({
            "id": turn["id"], "observation_id": turn["observation_id"], "status": turn["status"],
            "decision": {key: decision[key] for key in ("scene", "location", "observation")},
            "action_kind": decision["action"]["kind"], "steps": steps,
        })
    return result


def retrieve(facts, query, observation_id, limit=24):
    terms = words(query)
    active = [fact for fact in facts if fact["status"] != "resolved"]

    def rank(fact):
        match = len(terms & words(fact["location"] + " " + fact["claim"] + " " + fact["key"]))
        return (fact["category"] == "objective", match, fact["evidence_observation_id"])

    ranked = sorted(active, key=rank, reverse=True)
    return [dict(fact, age_observations=observation_id - fact["evidence_observation_id"])
            for fact in ranked[:limit]]


def ledger_context(events, location, limit=24):
    # Deaths and encounter eligibility stay available even after thousands of turns.
    retired = {}
    encounters = {}
    corrected = {e["corrects_event_key"] for e in events if e["type"] == "correction"}
    for event in events:
        if event["key"] in corrected:
            continue
        if event["type"] == "faint":
            retired[event["subject"]] = event["confidence"]
        if event["type"] in {"encounter", "capture"}:
            encounters.setdefault(event["area"], []).append(event["key"])
    relevant = [e for e in events if location and e["area"].casefold() == location.casefold()]
    chosen = {e["key"]: e for e in events[-limit:]}
    for event in relevant[-limit:]:
        chosen[event["key"]] = event
    return {"retired_pokemon": retired, "encounter_areas": encounters,
            "recent_or_local_events": list(chosen.values())[-2 * limit:],
            "correction_policy": "Corrections mark original events superseded; inspect their details before reuse."}
