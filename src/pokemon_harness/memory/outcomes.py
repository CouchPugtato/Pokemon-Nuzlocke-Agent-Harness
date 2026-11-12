"""evidence linked reviews of attempted controller plans"""

import json
from datetime import datetime, timedelta, timezone

from jsonschema import Draft202012Validator

from ..agent.schemas import OUTCOME_REVIEW


ATTEMPTED = {"completed", "executing", "unknown"}
PENDING = """
    FROM turns t WHERE EXISTS(
        SELECT 1 FROM action_steps s WHERE s.turn_id=t.id
        AND s.status IN ('completed','executing','unknown'))
    AND NOT EXISTS(SELECT 1 FROM outcome_reviews r WHERE r.target_turn_id=t.id)
"""


class Outcomes:
    def __init__(self, connection):
        self.connection = connection
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS action_intents(
                turn_id INTEGER PRIMARY KEY REFERENCES turns(id),
                working_revision INTEGER NOT NULL, subgoal TEXT);
            CREATE TABLE IF NOT EXISTS outcome_reviews(
                id INTEGER PRIMARY KEY, target_turn_id INTEGER NOT NULL REFERENCES turns(id),
                revision INTEGER NOT NULL, observation_id INTEGER NOT NULL REFERENCES observations(id),
                payload TEXT NOT NULL, created_at TEXT NOT NULL,
                UNIQUE(target_turn_id, revision));
            CREATE INDEX IF NOT EXISTS attempted_steps ON action_steps(status, turn_id);
        """)

    def intent(self, turn_id, state):
        self.connection.execute("INSERT INTO action_intents VALUES(?,?,?)",
                                (turn_id, state["revision"], json.dumps(state["subgoal"])))

    def pending_count(self):
        return self.connection.execute("SELECT COUNT(*) " + PENDING).fetchone()[0]

    def latest(self, turn_id):
        row = self.connection.execute(
            "SELECT * FROM outcome_reviews WHERE target_turn_id=? ORDER BY revision DESC LIMIT 1",
            (turn_id,)).fetchone()
        return self._decode(row) if row else None

    @staticmethod
    def _decode(row):
        return dict(json.loads(row["payload"]), revision=row["revision"],
                    reviewed_observation_id=row["observation_id"], recorded_at=row["created_at"])

    def history(self, limit=20, offset=0):
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("Outcome history requires limit 1..100 and nonnegative offset")
        return [self._decode(row) for row in self.connection.execute(
            "SELECT * FROM outcome_reviews ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset))]

    def plan_context(self, turn_id, observation_id=None):
        turn = self.connection.execute("SELECT * FROM turns WHERE id=?", (turn_id,)).fetchone()
        if turn is None:
            raise ValueError("Outcome review references a missing turn")
        decision = json.loads(turn["decision"])
        intent = self.connection.execute("SELECT * FROM action_intents WHERE turn_id=?", (turn_id,)).fetchone()
        steps = []
        before = turn["observation_id"]
        for row in self.connection.execute(
                "SELECT * FROM action_steps WHERE turn_id=? ORDER BY step_index", (turn_id,)):
            execution = json.loads(row["execution"]) if row["execution"] else {}
            steps.append({"step_index": row["step_index"], "input_status": row["status"],
                          "action": execution.get("effective_action", json.loads(row["action"])),
                          "before_observation_id": before, "after_observation_id": row["observation_id"],
                          "error": row["error"]})
            if row["observation_id"] is not None:
                before = row["observation_id"]
        latest = self.latest(turn_id)
        current = self.connection.execute("SELECT * FROM observations WHERE id=?", (observation_id,)).fetchone() if observation_id else self.connection.execute(
            "SELECT * FROM observations ORDER BY id DESC LIMIT 1").fetchone()
        attempted = [step for step in steps if step["input_status"] in ATTEMPTED]
        if attempted:
            final = attempted[-1]
            captured = self.connection.execute("SELECT * FROM observations WHERE id=?", (final["after_observation_id"],)).fetchone()
            intervened = self.connection.execute(
                "SELECT 1 FROM turns WHERE id>? AND observation_id<? LIMIT 1",
                (turn_id, current["id"] if current else 0)).fetchone()
            interrupted = self.connection.execute(
                "SELECT 1 FROM notes WHERE kind IN ('session_start','control_handoff','resume_after_handoff','user_stop','recovery') "
                "AND created_at>=? AND created_at<=? LIMIT 1",
                (turn["created_at"], current["created_at"] if current else turn["created_at"])).fetchone()
            if (turn["status"] == "completed" and final["input_status"] == "completed" and captured and current
                    and captured["id"] <= current["id"] and not intervened and not interrupted
                    and timedelta(0) <= datetime.fromisoformat(current["created_at"]) - datetime.fromisoformat(captured["created_at"]) <= timedelta(seconds=600)):
                final["review_after_observation_ids"] = sorted({captured["id"], current["id"]})
        conflicts = [row[0] for row in self.connection.execute(
            "SELECT key FROM working_invalidations WHERE revision=?", (intent["working_revision"],))] if intent else []
        subgoal = json.loads(intent["subgoal"]) if intent else None
        if conflicts and subgoal:
            subgoal = {"id": subgoal["id"], "status": subgoal["status"], "definition": None}
        return {"turn_id": turn_id, "input_status": turn["status"],
                "expected_revision": latest["revision"] if latest else 0,
                "expected_result_prediction": decision.get("expected_result", ""),
                "subgoal_at_input": subgoal,
                "subgoal_memory_validity": "invalidated" if conflicts else "supported",
                "contradiction_keys": conflicts,
                "working_revision_at_input": intent["working_revision"] if intent else None,
                "steps": steps}

    def context(self):
        row = self.connection.execute("SELECT t.id " + PENDING + " ORDER BY t.id LIMIT 1").fetchone()
        return {"pending_count": self.pending_count(),
                "next_plan": self.plan_context(row[0]) if row else None}

    def apply(self, reviews, observation_id):
        """appending reviews within the caller transaction"""
        if len(reviews) > 1:
            raise ValueError("Review at most one plan per decision")
        for review in reviews:
            Draft202012Validator(OUTCOME_REVIEW).validate(review)
            plan = self.plan_context(review["turn_id"], observation_id)
            if review["expected_revision"] != plan["expected_revision"]:
                raise ValueError("Outcome review refers to a stale revision")
            attempted = {step["step_index"]: step for step in plan["steps"] if step["input_status"] in ATTEMPTED}
            indices = [step["step_index"] for step in review["steps"]]
            if len(set(indices)) != len(indices) or set(indices) != set(attempted):
                raise ValueError("Review every attempted step exactly once and omit cancelled or unexecuted steps")
            for step in review["steps"]:
                source = attempted[step["step_index"]]
                before, after = step["before_observation_id"], step["after_observation_id"]
                if before != source["before_observation_id"]:
                    raise ValueError("Outcome review has the wrong before observation")
                if not before < after <= observation_id or not self.connection.execute(
                        "SELECT 1 FROM observations WHERE id=?", (after,)).fetchone():
                    raise ValueError("Outcome review requires available later observation evidence")
                if source["after_observation_id"] is not None and after not in source.get(
                        "review_after_observation_ids", [source["after_observation_id"]]):
                    raise ValueError("Outcome review must cite the recorded step result or an uninterrupted delayed result")
                if source["after_observation_id"] is None and step["assessment"] != "uncertain":
                    raise ValueError("Missing step capture requires an uncertain outcome")
                if source["after_observation_id"] is None and after != observation_id:
                    raise ValueError("Missing step capture requires the current observation")
                if not step["explanation"].strip():
                    raise ValueError("Outcome review requires an evidence explanation")
            self.connection.execute(
                "INSERT INTO outcome_reviews(target_turn_id,revision,observation_id,payload,created_at) VALUES(?,?,?,?,?)",
                (review["turn_id"], plan["expected_revision"] + 1, observation_id,
                 json.dumps(review, ensure_ascii=False), datetime.now(timezone.utc).isoformat()))
