"""bounded automatic recall from structured state and original sources"""

import re
import unicodedata
from collections import deque


def recall_settings(config):
    result = {"evidence_limit": 6, "location_limit": 3, "image_candidate_limit": 4, "aliases": []}
    value = config.get("automatic_recall", {})
    if not isinstance(value, dict) or set(value) - set(result):
        raise ValueError("Invalid automatic_recall settings")
    result.update(value)
    for key, lower, upper in (("evidence_limit", 0, 12), ("location_limit", 1, 5), ("image_candidate_limit", 0, 8)):
        if type(result[key]) is not int or not lower <= result[key] <= upper:
            raise ValueError(f"automatic_recall.{key} must be an integer between {lower} and {upper}")
    aliases = result["aliases"]
    if not isinstance(aliases, list) or len(aliases) > 32 or any(
            not isinstance(group, list) or not 2 <= len(group) <= 8 or any(
                not isinstance(name, str) or not name.strip() or len(name) > 80 for name in group) for group in aliases):
        raise ValueError("automatic_recall.aliases needs at most thirty two groups of two to eight names")
    return result


STOP_WORDS = set("a an the is was are were to from in at on of and or for with it this that find reach inspect get go".split())


def terms(value):
    value = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)
    value = "".join(char for char in unicodedata.normalize("NFKD", value) if not unicodedata.combining(char)).lower()
    value = re.sub(r"([a-z])([0-9])|([0-9])([a-z])", r"\1\3 \2\4", value)
    return set(re.findall(r"[a-z0-9]+", value)) - STOP_WORDS


def automatic_recall(facts, events, state, recent, contradictions, objective, observation_id, limit, settings):
    events = [event for event in events if event["evidence_observation_id"] <= observation_id]
    usable = [fact for fact in facts if fact["status"] != "resolved"
              and fact.get("scene_validity") not in {"stale", "needs_refresh"}
              and fact.get("is_latest", True)
              and fact.get("memory_validity") not in {"disputed", "invalidated"}
              and fact["evidence_observation_id"] <= observation_id]
    aliases = [(set().union(*(terms(name) for name in group)), [terms(name) for name in group])
               for group in settings["aliases"]]

    def expanded(value):
        original = terms(value)
        result = set(original)
        for combined, names in aliases:
            if any(name and name <= original for name in names):
                result.update(combined)
        return result

    locations = []
    location_keys = set()
    location_total = 0

    def location(value, source, ids, basis):
        nonlocal location_total
        key = frozenset(terms(value))
        if key and key not in location_keys:
            location_keys.add(key)
            location_total += 1
            if len(locations) < settings["location_limit"]:
                locations.append({"claim": value[:120], "source": source, "evidence_observation_ids": ids,
                                  "basis": basis, "hypothesis_only": True})

    valid_state = state.get("memory_validity") not in {"disputed", "invalidated"}
    if valid_state and state.get("location"):
        claim = state["location"]
        location(claim["claim"], "working_state", claim["evidence_observation_ids"], claim["basis"])
    for turn in reversed(recent):
        if not turn.get("contradiction_keys") and turn["observation_id"] <= observation_id:
            location(turn["decision"].get("location", ""), f"turn:{turn['id']}", [turn["observation_id"]], "inferred")
    for fact in sorted(usable, key=lambda item: item["evidence_observation_id"], reverse=True):
        if fact["category"] in {"world", "working"}:
            location(fact["location"], f"fact:{fact['fact_id']}", [fact["evidence_observation_id"]], fact["basis"])
    goal = state.get("subgoal") if valid_state else None
    goal_text = " ".join(goal["definition"].values()) if goal and goal["status"] == "active" and goal.get("definition") else ""
    uncertainty = " ".join(item["claim"] for item in state.get("uncertainties", [])) if valid_state else ""
    party = " ".join(fact["key"] for fact in usable if fact["category"] == "party")
    corrected = {event["corrects_event_key"] for event in events if event["type"] == "correction"}
    nearby = expanded(" ".join(item["claim"] for item in locations))
    identities = expanded(party)
    relevant_events = [event for event in events if event["key"] not in corrected
                       and event["evidence_observation_id"] <= observation_id
                       and event["type"] in {"faint", "capture", "encounter"}
                       and (expanded(event["area"]) & nearby or expanded(event["subject"]) & identities)]
    relevant_events.sort(key=lambda event: event["evidence_observation_id"], reverse=True)
    ledger_needs = " ".join(event["subject"] + " " + event["area"] for event in relevant_events[:12])
    signals = [("subgoal", expanded(goal_text)), ("party", expanded(party)),
               ("location", expanded(" ".join(item["claim"] for item in locations))),
               ("uncertainty", expanded(uncertainty + " " + (state.get("next_check", "") if valid_state else ""))),
               ("ledger", expanded(ledger_needs)), ("objective", expanded(objective))]
    all_terms = set().union(*(value for _, value in signals))
    signals.append(("lesson", all_terms))
    fact_terms = {fact["fact_id"]: expanded(" ".join(fact[key] for key in ("key", "claim", "location"))) for fact in usable}
    queues = []
    reasons = {}
    for name, target in signals:
        ranked = []
        for fact in usable:
            if (name == "party" and fact["category"] != "party") or (name == "lesson" and fact["category"] != "lesson"):
                continue
            overlap = len(target & fact_terms[fact["fact_id"]])
            if overlap:
                reasons.setdefault(fact["fact_id"], []).append(name)
                ranked.append((overlap, fact["evidence_observation_id"], fact["fact_id"], fact))
        queues.append(deque(item[-1] for item in sorted(ranked, key=lambda item: item[:3], reverse=True)))
    queues.append(deque(sorted(usable, key=lambda fact: (fact["evidence_observation_id"], fact["fact_id"]), reverse=True)))
    selected = {}
    while len(selected) < limit and any(queues):
        for queue in queues:
            while queue and queue[0]["fact_id"] in selected:
                queue.popleft()
            if queue and len(selected) < limit:
                fact = queue.popleft()
                selected[fact["fact_id"]] = dict(fact, age_observations=observation_id - fact["evidence_observation_id"],
                                               recall_reasons=reasons.get(fact["fact_id"], ["recent"]))
    evidence = []
    for dispute in contradictions:
        if dispute["operation"] == "open" and dispute["evidence_observation_id"] <= observation_id:
            evidence.append(dict(dispute, record_type="contradiction"))
    ranked_events = []
    for event in events:
        if event["evidence_observation_id"] > observation_id:
            continue
        overlap = len(all_terms & expanded(" ".join(str(event.get(key, "")) for key in ("key", "subject", "area", "details"))))
        if overlap:
            ranked_events.append((overlap, event["evidence_observation_id"], event["event_id"], event))
    correction_keys = {}
    for event in events:
        if event["corrects_event_key"]:
            correction_keys.setdefault(event["corrects_event_key"], []).append(event["key"])
    event_evidence = [dict(item[-1], record_type="event", superseded=item[-1]["key"] in corrected,
                           corrected_by=correction_keys.get(item[-1]["key"], []))
                      for item in sorted(ranked_events, key=lambda item: item[:3], reverse=True)]
    evidence_total = len(evidence) + len(event_evidence)
    pools = [deque(evidence), deque(event_evidence)]
    evidence = []
    while len(evidence) < settings["evidence_limit"] and any(pools):
        for pool in pools:
            if pool and len(evidence) < settings["evidence_limit"]:
                evidence.append(pool.popleft())
    evidence = [dict(item, age_observations=observation_id - item["evidence_observation_id"])
                for item in evidence[:settings["evidence_limit"]]]
    image_ids = list(dict.fromkeys(item["evidence_observation_id"] for item in [*evidence, *selected.values()]
                                  if item["evidence_observation_id"] < observation_id))
    return list(selected.values()), {
        "method": "structured lexical signals with configured aliases", "historical_only": True,
        "location_candidates": locations, "evidence": evidence,
        "image_candidate_observation_ids": image_ids[:settings["image_candidate_limit"]],
        "excluded_fact_count": len(facts) - len(usable),
        "omitted": {"facts": len(usable) - len(selected), "evidence": evidence_total - len(evidence),
                    "location_candidates": location_total - len(locations),
                    "image_candidates": max(0, len(image_ids) - settings["image_candidate_limit"])}}
