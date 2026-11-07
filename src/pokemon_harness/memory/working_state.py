"""versioned working state and subgoal transitions"""

import json
from datetime import datetime, timezone

from jsonschema import Draft202012Validator

from ..agent.schemas import WORKING_STATE_UPDATE


class WorkingState:
    def __init__(self, connection):
        self.connection = connection
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS working_revisions(
                revision INTEGER PRIMARY KEY, schema_version INTEGER NOT NULL,
                observation_id INTEGER NOT NULL REFERENCES observations(id),
                turn_id INTEGER NOT NULL UNIQUE REFERENCES turns(id),
                payload TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS subgoal_changes(
                revision INTEGER PRIMARY KEY REFERENCES working_revisions(revision),
                subgoal_id TEXT NOT NULL, operation TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS subgoal_identity ON subgoal_changes(subgoal_id);
        """)

    def current(self):
        row = self.connection.execute(
            "SELECT * FROM working_revisions ORDER BY revision DESC LIMIT 1").fetchone()
        if row is None:
            return {"schema_version": 1, "revision": 0, "location": None,
                    "uncertainties": [], "next_check": "", "subgoal": None,
                    "evidence_observation_id": None, "turn_id": None, "recorded_at": None}
        return self._decode(row)

    def history(self, limit=20, offset=0):
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("Working state history requires limit 1..100 and nonnegative offset")
        rows = self.connection.execute(
            "SELECT * FROM working_revisions ORDER BY revision DESC LIMIT ? OFFSET ?",
            (limit, offset)).fetchall()
        return [self._decode(row) for row in rows]

    @staticmethod
    def _decode(row):
        if row["schema_version"] != 1:
            raise ValueError("Unsupported working state schema version")
        return dict(json.loads(row["payload"]), schema_version=row["schema_version"],
                    revision=row["revision"], evidence_observation_id=row["observation_id"],
                    turn_id=row["turn_id"], recorded_at=row["created_at"])

    def _references(self, ids, observation_id):
        for evidence_id in ids:
            if evidence_id > observation_id or not self.connection.execute(
                    "SELECT 1 FROM observations WHERE id=?", (evidence_id,)).fetchone():
                raise ValueError(f"Working state references unavailable observation {evidence_id}")

    def apply(self, update, observation_id, turn_id):
        """appending within the caller transaction before input"""
        if update is None:
            return
        Draft202012Validator(WORKING_STATE_UPDATE).validate(update)
        state = self.current()
        if update["expected_revision"] != state["revision"]:
            raise ValueError("Working state update refers to a stale revision")
        for claim in [update["location"], *update["uncertainties"]]:
            if claim is None:
                continue
            ids = claim["evidence_observation_ids"]
            self._references(ids, observation_id)
            if claim["basis"] != "unknown" and not ids:
                raise ValueError("Observed and inferred working claims require evidence references")
            if claim["basis"] == "observed" and observation_id not in ids:
                raise ValueError("Observed working claims require the current observation")
        change = update["subgoal_change"]
        subgoal = state["subgoal"]
        if change is not None:
            self._references(change["evidence_observation_ids"], observation_id)
            operation = change["operation"]
            definition = change["definition"]
            if (operation in {"start", "revise"}) != (definition is not None):
                raise ValueError("Only start and revise require a subgoal definition")
            if operation == "start":
                if subgoal and subgoal["status"] == "active":
                    raise ValueError("Complete or abandon the active subgoal before starting another")
                if self.connection.execute(
                        "SELECT 1 FROM subgoal_changes WHERE subgoal_id=?", (change["id"],)).fetchone():
                    raise ValueError("Subgoal identities cannot be reused")
                subgoal = {"id": change["id"], "status": "active", "definition": definition}
            else:
                if not subgoal or subgoal["id"] != change["id"] or subgoal["status"] != "active":
                    raise ValueError("Subgoal transition requires the matching active subgoal")
                if operation == "complete" and observation_id not in change["evidence_observation_ids"]:
                    raise ValueError("Subgoal completion requires current visual evidence")
                if operation == "revise":
                    subgoal["definition"] = definition
                else:
                    subgoal["status"] = "completed" if operation == "complete" else "abandoned"
            subgoal.update(reason=change["reason"],
                           evidence_observation_ids=change["evidence_observation_ids"],
                           updated_observation_id=observation_id)
        payload = {"location": update["location"], "uncertainties": update["uncertainties"],
                   "next_check": update["next_check"], "subgoal": subgoal}
        revision = state["revision"] + 1
        self.connection.execute(
            "INSERT INTO working_revisions VALUES(?,1,?,?,?,?)",
            (revision, observation_id, turn_id, json.dumps(payload, ensure_ascii=False),
             datetime.now(timezone.utc).isoformat()))
        if change is not None:
            self.connection.execute("INSERT INTO subgoal_changes VALUES(?,?,?,?)",
                                    (revision, change["id"], change["operation"], json.dumps(change)))
