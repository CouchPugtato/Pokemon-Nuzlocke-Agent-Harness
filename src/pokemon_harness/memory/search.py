"""explicit read only search over durable memory and contradictions"""

from jsonschema import Draft202012Validator

from ..agent.schemas import MEMORY_SEARCH_SCHEMA
from .retrieval import words


def search_records(facts, events, request, observation_id, contradictions=(), turns=()):
    Draft202012Validator(MEMORY_SEARCH_SCHEMA).validate(request)
    if request["category"] and request["scope"] != "facts":
        raise ValueError("A memory category filter requires scope=facts")
    terms = words(request["query"])
    location = request["location"].strip().casefold()
    corrections = {}
    for event in events:
        reference = event["corrects_event_key"]
        if reference:
            corrections.setdefault(reference, []).append(event["key"])
    candidates = []
    if request["scope"] in {"facts", "all"}:
        for fact in facts:
            if request["category"] and fact["category"] != request["category"]:
                continue
            if not request["include_history"] and not fact["is_latest"]:
                continue
            candidates.append(dict(fact, record_type="fact"))
    if request["scope"] in {"events", "all"}:
        candidates.extend(dict(event, record_type="event", corrected_by=corrections.get(event["key"], []))
                          for event in events)
    if request["scope"] in {"contradictions", "all"}:
        candidates.extend(dict(record, record_type="contradiction") for record in contradictions)
    if request["scope"] in {"turns", "all"}:
        candidates.extend(turns)

    matches = []
    for record in candidates:
        area = record.get("location", record.get("area", ""))
        if location and location not in area.casefold():
            continue
        searchable = " ".join(str(record.get(key, "")) for key in
                              ("key", "claim", "category", "location", "type", "subject", "area", "details", "reason",
                               "fact_ids", "event_keys", "replacement_facts", "correction_event_keys", "strategy_kind", "strategy_target"))
        overlap = len(terms & words(searchable))
        if terms and not overlap:
            continue
        # exact keys when the earlier record identity is known
        exact_key = bool(request["query"].strip()) and record["key"].casefold() == request["query"].strip().casefold()
        record["age_observations"] = observation_id - record["evidence_observation_id"]
        rank = (exact_key, overlap, record["evidence_observation_id"], record.get("fact_id", record.get("event_id", 0)))
        matches.append((rank, record))
    matches.sort(key=lambda item: item[0], reverse=True)
    start, count = request["offset"], request["limit"]
    return {"request": dict(request), "searched_at_observation_id": observation_id,
            "total_matches": len(matches), "has_more": start + count < len(matches),
            "results": [record for _, record in matches[start:start + count]]}
