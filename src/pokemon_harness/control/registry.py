"""Discover active harness sessions without guessing from saved-run timestamps."""

import json
import os
from pathlib import Path

from ..telemetry.recorder import RunLock, write_json


def registry_directory():
    override = os.environ.get("POKEMON_HARNESS_SESSION_DIR")
    if override:
        return Path(override).resolve()
    root = Path(os.environ["LOCALAPPDATA"]) if os.environ.get("LOCALAPPDATA") else Path.home() / ".local" / "state"
    return root / "pokemon-harness" / "sessions"


def read_record(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(value, dict):
            return value
    except (OSError, ValueError):
        pass
    return None


def run_is_locked(directory):
    path = Path(directory) / ".lock"
    if not path.is_file():
        return False
    try:
        with RunLock(path):
            pass
    except RuntimeError:
        return True
    return False


def register_session(record):
    root = registry_directory()
    root.mkdir(parents=True, exist_ok=True)
    write_json(Path(record["run_dir"]) / ".session.json", record)
    write_json(root / (record["session_id"] + ".json"), record)


def unregister_session(record):
    path = Path(record["run_dir"]) / ".session.json"
    current = read_record(path)
    if current and current.get("session_id") == record["session_id"]:
        path.unlink(missing_ok=True)
    (registry_directory() / (record["session_id"] + ".json")).unlink(missing_ok=True)


def active_sessions():
    result = []
    for path in registry_directory().glob("*.json"):
        record = read_record(path)
        if not record or not all(isinstance(record.get(key), str) and record[key]
                                 for key in ("session_id", "run_dir", "started_at")):
            continue
        directory = Path(record["run_dir"])
        current = read_record(directory / ".session.json")
        if current and current.get("session_id") == record["session_id"] and run_is_locked(directory):
            result.append(record)
        # Stale entries are ignored, never used to break locks or reclaim ownership.
    return sorted(result, key=lambda record: record["started_at"])


def select_session(run_dir=None, session_id=None, runs_dir=None):
    if run_dir is not None and session_id is not None:
        raise ValueError("Choose a run directory or --session, not both")
    sessions = active_sessions()
    if run_dir is not None:
        target = str(Path(run_dir).resolve())
        sessions = [record for record in sessions if record["run_dir"] == target]
    if session_id is not None:
        sessions = [record for record in sessions if record["session_id"] == session_id]
    if runs_dir is not None:
        root = Path(runs_dir).resolve()
        sessions = [record for record in sessions if Path(record["run_dir"]).is_relative_to(root)]
    if not sessions:
        raise ValueError("No matching active session. Use pokemon-harness status")
    if len(sessions) > 1:
        choices = ", ".join(record["session_id"] for record in sessions)
        raise ValueError(f"Multiple active sessions: {choices}. Specify --session ID or a run directory")
    return sessions[0]
