"""configuration paths relative to the config file"""

import json
from pathlib import Path
from .telemetry.storage import storage_settings
from .memory.progress import progress_settings
from .agent.context_budget import budget_settings
from .memory.conversation import compaction_settings


def load_config(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Configuration must be an object")
    required = {"run_dir", "objective", "model", "emulator"}
    if not required.issubset(value):
        raise ValueError(f"Configuration needs: {', '.join(sorted(required))}")
    if not isinstance(value["objective"], str) or not value["objective"].strip():
        raise ValueError("objective must be nonempty text")
    if not isinstance(value["model"], dict) or not isinstance(value["emulator"], dict):
        raise ValueError("model and emulator must be objects")
    value["model"].setdefault("context_length", 32768)
    value["model"]["context_budget"] = budget_settings(value["model"])
    if type(value["model"]["context_length"]) is not int or value["model"]["context_length"] < 1024:
        raise ValueError("model.context_length must be an integer of at least 1024")
    value["run_dir"] = str((path.parent / value["run_dir"]).resolve())
    value["storage"] = storage_settings(value)
    value["progress"] = progress_settings(value)
    value["compaction"] = compaction_settings(value)
    value.setdefault("max_action_frames", 60)
    value.setdefault("max_sequence_length", 5)
    value.setdefault("sequence_delay_ms", 250)
    value.setdefault("recent_turns", 6)
    value.setdefault("memory_limit", 24)
    value.setdefault("ledger_limit", 24)
    value.setdefault("memory_search_rounds", 3)
    value.setdefault("outcome_review_rounds", 3)
    value.setdefault("rules", ["First eligible encounter per area may be caught.",
                                "Fainted Pokemon are permanently retired.",
                                "Nickname every captured Pokemon."])
    for key, lower, upper in [("max_action_frames", 1, 120), ("recent_turns", 1, 20),
                              ("max_sequence_length", 1, 20), ("sequence_delay_ms", 0, 60000),
                              ("memory_limit", 1, 100), ("ledger_limit", 1, 100),
                              ("memory_search_rounds", 0, 5), ("outcome_review_rounds", 0, 5)]:
        if type(value[key]) is not int or not lower <= value[key] <= upper:
            raise ValueError(f"{key} must be an integer between {lower} and {upper}")
    if not isinstance(value["rules"], list) or not all(isinstance(x, str) for x in value["rules"]):
        raise ValueError("rules must be a list of text instructions")
    if value["emulator"].get("backend") in {"mgba", "vbam"}:
        minimum = value["emulator"].get("min_hold_frames", 6)
        if type(minimum) is not int or not 1 <= minimum <= value["max_action_frames"]:
            raise ValueError("emulator.min_hold_frames must be between 1 and max_action_frames")
    return value
