"""explicit bounded subgoal repetition allowances without recovery overrides"""

import json
from datetime import datetime, timedelta, timezone


class Repetition:
    def __init__(self, connection):
        self.connection = connection
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS repetition_allowances(
                subgoal_id TEXT PRIMARY KEY, working_revision INTEGER NOT NULL REFERENCES working_revisions(revision),
                strategy_kind TEXT NOT NULL, buttons TEXT NOT NULL, max_moves INTEGER NOT NULL,
                expires_at TEXT NOT NULL, reason TEXT NOT NULL, termination_condition TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS repetition_uses(
                turn_id INTEGER PRIMARY KEY REFERENCES turns(id), subgoal_id TEXT NOT NULL REFERENCES repetition_allowances(subgoal_id));
        """)

    def grant(self, state, strategy, buttons, moves, seconds, reason, until):
        goal = state.get("subgoal")
        if not goal or goal["status"] != "active" or state.get("memory_validity") == "invalidated":
            raise ValueError("Repetition allowance requires an active supported subgoal")
        if strategy not in {"move", "advance_dialogue", "wait"} or len(buttons) > 1 or any(
                button not in {"UP", "DOWN", "LEFT", "RIGHT", "A", "B"} for button in buttons):
            raise ValueError("Repetition allowance needs one narrow movement dialogue or wait input")
        if (strategy == "wait") != (not buttons) or (strategy == "move" and buttons[0] not in {"UP", "DOWN", "LEFT", "RIGHT"}) \
                or (strategy == "advance_dialogue" and buttons[0] not in {"A", "B"}):
            raise ValueError("Repetition buttons do not match the allowed strategy")
        if type(moves) is not int or not 1 <= moves <= 32 or type(seconds) is not int or not 1 <= seconds <= 300:
            raise ValueError("Repetition allowance needs one to thirty two moves and one to three hundred seconds")
        if not reason.strip() or not until.strip() or max(len(reason), len(until)) > 600:
            raise ValueError("Repetition allowance requires a bounded reason and termination condition")
        with self.connection:
            self.connection.execute("INSERT INTO repetition_allowances VALUES(?,?,?,?,?,?,?,?)",
                (goal["id"], state["revision"], strategy, json.dumps(sorted(buttons)), moves,
                 (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat(), reason, until))

    def context(self, state):
        goal = state.get("subgoal")
        if not goal:
            return None
        row = self.connection.execute("SELECT * FROM repetition_allowances WHERE subgoal_id=?", (goal["id"],)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["buttons"] = json.loads(result["buttons"])
        used = self.connection.execute("SELECT COUNT(*) FROM repetition_uses WHERE subgoal_id=?", (goal["id"],)).fetchone()[0]
        result["moves_remaining"] = max(0, row["max_moves"] - used)
        result["active"] = bool(goal["status"] == "active" and state["revision"] == row["working_revision"]
            and state.get("memory_validity") != "invalidated" and used < row["max_moves"]
            and datetime.now(timezone.utc) < datetime.fromisoformat(row["expires_at"]))
        return result

    def reserve(self, turn_id, decision, state):
        allowance = self.context(state)
        if not allowance or not allowance["active"]:
            return
        action = decision["action"]
        if (decision["scene"] not in {"overworld", "dialogue", "menu"} or decision.get("additional_actions")
                or decision["strategy"]["kind"] != allowance["strategy_kind"]
                or sorted(action["buttons"]) != allowance["buttons"]):
            return
        self.connection.execute("INSERT INTO repetition_uses VALUES(?,?)", (turn_id, allowance["subgoal_id"]))
