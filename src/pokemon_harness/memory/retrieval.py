"""bounded context selection preserving durable source records"""

import re


def words(value):
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def recent_turn_context(turns):
    """recalling observations and input outcomes without speculative rationales"""
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
        review = turn.get("outcome_review")
        conflicts = turn.get("contradiction_keys", [])
        if review:
            review = {"turn_id": review["turn_id"], "revision": review["revision"],
                      "reviewed_observation_id": review["reviewed_observation_id"], "steps": [
                          {**{key: step[key] for key in ("step_index", "before_observation_id",
                                                        "after_observation_id", "assessment")},
                           "explanation_excerpt": step["explanation"][:200]} for step in review["steps"]]}
            if conflicts:
                review["steps"] = []
                review["memory_validity"] = "invalidated"
        result.append({
            "id": turn["id"], "observation_id": turn["observation_id"], "status": turn["status"],
            "decision": ({"scene": "unknown", "location": "", "observation": "Historical interpretation withheld due to disputed source memories"}
                         if conflicts else {key: decision[key] for key in ("scene", "location", "observation")}),
            "contradiction_keys": conflicts,
            "action_kind": decision["action"]["kind"], "steps": steps,
            "outcome_review": review,
        })
    return result


def retrieve(facts, query, observation_id, limit=24):
    terms = words(query)
    active = [fact for fact in facts if fact["status"] != "resolved"
              and fact.get("scene_validity") not in {"stale", "needs_refresh"}
              and fact.get("memory_validity") not in {"disputed", "invalidated"}]

    def rank(fact):
        match = len(terms & words(fact["location"] + " " + fact["claim"] + " " + fact["key"]))
        return (fact["category"] == "objective", match, fact["evidence_observation_id"])

    ranked = sorted(active, key=rank, reverse=True)
    return [dict(fact, age_observations=observation_id - fact["evidence_observation_id"])
            for fact in ranked[:limit]]


def ledger_context(events, location, limit=24):
    # deaths and encounter eligibility staying available across long runs
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
            "badge_events": [event for event in events if event["type"] == "badge" and event["key"] not in corrected],
            "recent_or_local_events": list(chosen.values())[-2 * limit:],
            "correction_policy": "Corrections mark original events superseded; inspect their details before reuse."}
