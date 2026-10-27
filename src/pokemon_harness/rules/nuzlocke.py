def unresolved_events(events):
    corrected = {event["corrects_event_key"] for event in events if event["type"] == "correction"}
    return [event["key"] for event in events
            if event["key"] not in corrected and event["confidence"] != "high"
            and event["type"] in {"encounter", "capture", "faint", "correction"}]
