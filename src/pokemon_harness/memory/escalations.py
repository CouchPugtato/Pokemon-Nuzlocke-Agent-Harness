"""durable review cases and append only resolutions"""

import json
from datetime import datetime, timezone

from ..rules.nuzlocke import unresolved_events


def timestamp():
    return datetime.now(timezone.utc).isoformat()


class Escalations:
    def __init__(self, connection):
        self.connection = connection
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS escalation_cases(
                id INTEGER PRIMARY KEY, case_key TEXT NOT NULL UNIQUE,
                reason TEXT NOT NULL, details TEXT NOT NULL,
                evidence_ids TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS case_resolutions(
                id INTEGER PRIMARY KEY, case_id INTEGER NOT NULL REFERENCES escalation_cases(id),
                outcome TEXT NOT NULL, explanation TEXT NOT NULL, evidence_ids TEXT NOT NULL,
                correction_key TEXT NOT NULL, allow_continue INTEGER NOT NULL,
                created_at TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS resolutions_case ON case_resolutions(case_id,id);
        """)

    def _evidence(self, observation_ids):
        values = list(dict.fromkeys(observation_ids))
        for value in values:
            if type(value) is not int or not self.connection.execute(
                    "SELECT 1 FROM observations WHERE id=?", (value,)).fetchone():
                raise ValueError(f"Observation {value!r} does not belong to this run")
        return values

    def open(self, key, reason, evidence_ids=(), details=None):
        with self.connection:
            return self.open_in_transaction(key, reason, evidence_ids, details)

    def open_in_transaction(self, key, reason, evidence_ids=(), details=None):
        if not isinstance(key, str) or not key.strip() or len(key) > 240:
            raise ValueError("Case key must contain between 1 and 240 characters")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 2000:
            raise ValueError("Case reason must contain between 1 and 2000 characters")
        evidence = self._evidence(evidence_ids)
        self.connection.execute(
            "INSERT OR IGNORE INTO escalation_cases(case_key,reason,details,evidence_ids,created_at) "
            "VALUES(?,?,?,?,?)", (key, reason, json.dumps(details or {}), json.dumps(evidence), timestamp()))
        return self.connection.execute("SELECT id FROM escalation_cases WHERE case_key=?", (key,)).fetchone()[0]

    def cases(self):
        result = []
        for row in self.connection.execute("SELECT * FROM escalation_cases ORDER BY id"):
            case = dict(row)
            case["details"] = json.loads(case["details"])
            case["evidence_ids"] = json.loads(case["evidence_ids"])
            case["evidence"] = [dict(self.connection.execute(
                "SELECT id,image_path,created_at,timing FROM observations WHERE id=?", (value,)).fetchone())
                for value in case["evidence_ids"]]
            resolutions = []
            for resolution in self.connection.execute(
                    "SELECT * FROM case_resolutions WHERE case_id=? ORDER BY id", (case["id"],)):
                item = dict(resolution)
                item["evidence_ids"] = json.loads(item["evidence_ids"])
                item["allow_continue"] = bool(item["allow_continue"])
                resolutions.append(item)
            case["resolutions"] = resolutions
            last = resolutions[-1] if resolutions else None
            case["blocking"] = not last or last["outcome"] == "invalid" or not (
                last["outcome"] == "evidence" or last["allow_continue"])
            result.append(case)
        return result

    def validity(self):
        outcomes = {row[0] for row in self.connection.execute("SELECT outcome FROM case_resolutions")}
        if "invalid" in outcomes:
            return "invalid"
        if outcomes & {"user-assertion", "unverifiable"}:
            return "unverifiable"
        return "supported"

    def resolve(self, case_id, outcome, explanation, evidence_ids=(), correction_key="", allow_continue=False):
        with self.connection:
            return self.resolve_in_transaction(case_id, outcome, explanation, evidence_ids, correction_key, allow_continue)

    def resolve_in_transaction(self, case_id, outcome, explanation, evidence_ids=(), correction_key="", allow_continue=False):
        if outcome not in {"evidence", "user-assertion", "unverifiable", "invalid"}:
            raise ValueError("Unknown resolution outcome")
        if not isinstance(explanation, str) or not explanation.strip() or len(explanation) > 2000:
            raise ValueError("Resolution reason must contain between 1 and 2000 characters")
        case = next((case for case in self.cases() if case["id"] == case_id), None)
        if not case:
            raise ValueError(f"No case {case_id} in this run")
        evidence = self._evidence(evidence_ids)
        if outcome == "invalid" and allow_continue:
            raise ValueError("An invalid run cannot be authorized to continue")
        if outcome == "evidence" and case["details"].get("kind") == "stalled_progress":
            self._progress_evidence(case, evidence, correction_key, allow_continue)
        elif outcome == "evidence":
            row = self.connection.execute("SELECT * FROM events WHERE key=?", (correction_key,)).fetchone()
            event = json.loads(row["payload"]) if row else {}
            corrected = {json.loads(value[0])["corrects_event_key"] for value in
                         self.connection.execute("SELECT payload FROM events")}
            if not (row and evidence and row["observation_id"] in evidence
                    and case["details"].get("event_key")
                    and event.get("type") == "correction" and event.get("confidence") == "high"
                    and event.get("corrects_event_key") == case["details"]["event_key"]
                    and correction_key not in corrected):
                raise ValueError("Evidence resolution requires an active high-confidence ledger correction "
                                 "for this case and its observation ID")
        elif correction_key:
            raise ValueError("Only evidence resolutions may reference a ledger correction")
        # preserving assertions without upgrading them to observed facts
        cursor = self.connection.execute(
            "INSERT INTO case_resolutions(case_id,outcome,explanation,evidence_ids,correction_key,"
            "allow_continue,created_at) VALUES(?,?,?,?,?,?,?)",
            (case_id, outcome, explanation, json.dumps(evidence), correction_key, int(allow_continue), timestamp()))
        return cursor.lastrowid

    def _progress_evidence(self, case, evidence, correction_key, allow_continue):
        episode_id = case["details"].get("episode_id")
        episode = self.connection.execute("SELECT * FROM progress_episodes WHERE id=?", (episode_id,)).fetchone()
        if (correction_key or allow_continue or not episode or episode["case_id"] != case["id"]
                or episode["status"] != "stopped" or not hasattr(self, "progress")):
            raise ValueError("Progress evidence requires the stopped episode and no ledger correction or override")
        state = self.progress.refresh()
        row = self.connection.execute("""
            SELECT r.* FROM outcome_reviews r JOIN recovery_attempts a ON a.turn_id=r.target_turn_id
            WHERE a.episode_id=? ORDER BY r.target_turn_id DESC,r.revision DESC LIMIT 1
        """, (episode_id,)).fetchone()
        latest = self.connection.execute(
            "SELECT MAX(turn_id) FROM action_steps WHERE status IN ('completed','executing','unknown')").fetchone()[0]
        if (not row or row["target_turn_id"] <= episode["start_turn_id"] or state["no_progress_attempts"]
                or row["target_turn_id"] != latest
                or self.progress.outcomes.pending_count() or state["episode_id"] != episode_id
                or any(signal["kind"] in {"exact_visual", "similar_visual"} for signal in state["signals"])):
            raise ValueError("Progress evidence requires reviewed advancement from this episode with no remaining visual cycle or backlog")
        review = json.loads(row["payload"])
        for step in review["steps"]:
            ids = {step["before_observation_id"], step["after_observation_id"], row["observation_id"]}
            hashes = self.connection.execute("SELECT image_hash FROM observations WHERE id IN (?,?)",
                (step["before_observation_id"], step["after_observation_id"])).fetchall()
            if step["assessment"] == "progress_observed" and ids.issubset(evidence) and len({value[0] for value in hashes}) == 2:
                return
        raise ValueError("Progress evidence must include the latest recovery review and its before and after screenshots")

    def sync_events(self, events):
        uncertain = set(unresolved_events(events))
        for event in events:
            if event["key"] in uncertain:
                self.open("event:" + event["key"], "Uncertain consequential Nuzlocke event",
                          [event["evidence_observation_id"]],
                          {"event_key": event["key"], "type": event["type"], "subject": event["subject"],
                           "area": event["area"], "missing_fact": "A supported interpretation of the event"})
        corrected = {event["corrects_event_key"] for event in events}
        for case in self.cases():
            if not case["blocking"] or case["resolutions"]:
                continue
            key = case["details"].get("event_key")
            for event in events:
                if (key and event["type"] == "correction" and event["corrects_event_key"] == key
                        and event["confidence"] == "high" and event["key"] not in corrected):
                    self.resolve(case["id"], "evidence", "Recorded ledger correction linked to source observation",
                                 [event["evidence_observation_id"]], event["key"])
                    break

    def review(self, events):
        self.sync_events(events)
        cases = self.cases()
        return {"validity": self.validity(), "blocked_cases": [case for case in cases if case["blocking"]],
                "authorized_overrides": [case for case in cases if not case["blocking"] and
                                         case["resolutions"][-1]["outcome"] != "evidence"]}
