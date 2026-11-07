"""transactional memory and event history for a single run"""

import json
import sqlite3
from datetime import datetime, timezone
from .escalations import Escalations
from .working_state import WorkingState
from .outcomes import Outcomes
from .progress import Progress
from .contradictions import Contradictions
from .budgets import BudgetCalibration
from .conversation import Conversation, SOURCE_TABLES, compaction_settings
from ..control.loop_detection import image_features
from ..telemetry.cleanup import cleanup_operations


def now():
    return datetime.now(timezone.utc).isoformat()


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class Memory:
    def __init__(self, path):
        self.connection = sqlite3.connect(path)
        try:
            self._initialize_schema()
        except BaseException as exc:
            cleanup_operations([("database", self.connection.close)], exc)
            raise

    def _initialize_schema(self):
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS observations(
                id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, image_path TEXT NOT NULL,
                image_hash TEXT NOT NULL, timing TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS observation_capture(
                observation_id INTEGER PRIMARY KEY REFERENCES observations(id), payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS turns(
                id INTEGER PRIMARY KEY, observation_id INTEGER NOT NULL UNIQUE REFERENCES observations(id),
                decision TEXT NOT NULL, status TEXT NOT NULL, error TEXT,
                created_at TEXT NOT NULL, finished_at TEXT);
            CREATE TABLE IF NOT EXISTS facts(
                id INTEGER PRIMARY KEY, category TEXT NOT NULL, key TEXT NOT NULL, payload TEXT NOT NULL,
                observation_id INTEGER NOT NULL REFERENCES observations(id), created_at TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS facts_key ON facts(category, key, id);
            CREATE TABLE IF NOT EXISTS events(
                id INTEGER PRIMARY KEY, key TEXT NOT NULL UNIQUE, payload TEXT NOT NULL,
                observation_id INTEGER NOT NULL REFERENCES observations(id), created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS notes(
                id INTEGER PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS action_steps(
                turn_id INTEGER NOT NULL REFERENCES turns(id), step_index INTEGER NOT NULL,
                action TEXT NOT NULL, status TEXT NOT NULL, observation_id INTEGER REFERENCES observations(id),
                execution TEXT, error TEXT, updated_at TEXT NOT NULL,
                PRIMARY KEY(turn_id, step_index));
        """)
        self.escalations = Escalations(self.connection)
        self.working = WorkingState(self.connection)
        self.outcomes = Outcomes(self.connection)
        self.progress = Progress(self.connection, self.outcomes)
        self.contradictions = Contradictions(self.connection, self.escalations)
        self.budgets = BudgetCalibration(self.connection)
        self.conversation = Conversation(self.connection)

    def initialize(self, config):
        # keeping the objective and rules fixed across sessions
        identity = {"objective": config["objective"], "rules": config["rules"]}
        existing = self.connection.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if existing and json.loads(existing[0]) != identity:
            raise ValueError("Run objective/rules differ from saved run. Use a new run_dir.")
        with self.connection:
            self.connection.execute("INSERT OR IGNORE INTO metadata VALUES('identity', ?)", (encode(identity),))
            self.progress.configure(config)

    def observe(self, path, digest, timing, image=None, capture_metadata=None):
        features = image_features(image) if image is not None else None
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO observations(created_at,image_path,image_hash,timing) VALUES(?,?,?,?)",
                (now(), str(path), digest, timing))
            if features is not None:
                self.connection.execute("INSERT INTO observation_features VALUES(?,?)", (cursor.lastrowid, encode(features)))
            if capture_metadata is not None:
                self.connection.execute("INSERT INTO observation_capture VALUES(?,?)", (cursor.lastrowid, encode(capture_metadata)))
            self.conversation.append("observation", {"image_hash": digest, "timing": timing}, cursor.lastrowid)
        return cursor.lastrowid

    def capture_metadata(self, observation_id):
        row = self.connection.execute("SELECT payload FROM observation_capture WHERE observation_id=?", (observation_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def recover(self):
        """avoiding replay of uncertain actions"""
        rows = self.connection.execute("SELECT id,status FROM turns WHERE status IN ('planned','executing')").fetchall()
        with self.connection:
            for row in rows:
                self.connection.execute("UPDATE turns SET status='unknown', error=?, finished_at=? WHERE id=?",
                                        ("Process ended before completion was recorded", now(), row["id"]))
                self._cancel_steps(row["id"], "Process ended before sequence completion")
                self.conversation.turn_status(row["id"], "unknown", "Process ended before completion was recorded")
            self.conversation.recover()
        return [dict(row) for row in rows]

    def plan(self, observation_id, decision, request_id=None):
        """saving facts events and plans together before input"""
        with self.connection:
            if decision["action"]["kind"] in {"search_memory", "review_outcomes"}:
                raise ValueError("Search and review actions cannot change working state or create controller plans")
            self.outcomes.apply(decision.get("outcome_reviews", []), observation_id)
            if decision["action"]["kind"] in {"press", "wait"} and self.outcomes.pending_count():
                raise ValueError("Review pending action outcomes before another controller plan")
            dependencies = self.contradictions.dependencies(decision.get(
                "memory_dependencies", {"facts": [], "working_state": None, "working_state_events": None}))
            created = {}
            for event in decision["events"]:
                existing = self.connection.execute("SELECT payload FROM events WHERE key=?", (event["key"],)).fetchone()
                if existing:
                    if json.loads(existing[0]) != event:
                        raise ValueError(f"Event key {event['key']} already exists; append a correction")
                    continue
                reference = event["corrects_event_key"]
                if reference and not self.connection.execute("SELECT 1 FROM events WHERE key=?", (reference,)).fetchone():
                    raise ValueError(f"Correction references missing event: {reference}")
                self.connection.execute(
                    "INSERT INTO events(key,payload,observation_id,created_at) VALUES(?,?,?,?)",
                    (event["key"], encode(event), observation_id, now()))
            for fact in decision["memory_updates"]:
                identity = (fact["category"], fact["key"])
                if identity in created:
                    raise ValueError("Write each fact key at most once per decision")
                saved = self.connection.execute(
                    "INSERT INTO facts(category,key,payload,observation_id,created_at) VALUES(?,?,?,?,?)",
                    (fact["category"], fact["key"], encode(fact), observation_id, now()))
                created[identity] = saved.lastrowid
            self.contradictions.register_facts(created, dependencies["facts"], observation_id)
            self.contradictions.apply(decision.get("contradiction_updates", []), observation_id)
            self.contradictions.sync_corrections(decision["events"])
            cursor = self.connection.execute(
                "INSERT INTO turns(observation_id,decision,status,created_at) VALUES(?,?,'planned',?)",
                (observation_id, encode(decision), now()))
            previous = self.working.current()
            update = decision.get("working_state_update")
            self.working.apply(update, observation_id, cursor.lastrowid)
            self.contradictions.register_working(self.working.current(), previous, update,
                                                 dependencies["working_state"], dependencies["working_state_events"])
            progress = self.progress.refresh()
            if decision["action"]["kind"] in {"press", "wait"}:
                self.progress.reserve(cursor.lastrowid, decision, progress)
                self.outcomes.intent(cursor.lastrowid, self.working_context())
                for index, action in enumerate([decision["action"], *decision.get("additional_actions", [])], 1):
                    self.connection.execute(
                        "INSERT INTO action_steps(turn_id,step_index,action,status,updated_at) VALUES(?,?,?,'planned',?)",
                        (cursor.lastrowid, index, encode(action), now()))
            if request_id is not None:
                self.conversation.link(request_id, cursor.lastrowid, observation_id, decision)
        return cursor.lastrowid

    def review_outcomes(self, observation_id, reviews, request_id=None):
        with self.connection:
            self.outcomes.apply(reviews, observation_id)
            self.progress.refresh()
            if request_id is not None:
                self.conversation.complete_readonly(request_id, observation_id, "review_completed",
                                                    {"reviews": reviews})

    def begin_request(self, request, observation_id, report, model):
        with self.connection:
            return self.conversation.begin(request, observation_id, report, model)

    def record_response(self, request_id, decision, trace_path):
        with self.connection:
            self.conversation.respond(request_id, decision, trace_path)

    def validate_response(self, request_id):
        with self.connection:
            self.conversation.validate(request_id)

    def fail_request(self, request_id, status, reason):
        with self.connection:
            self.conversation.fail(request_id, status, reason)

    def record_search(self, request_id, observation_id, result):
        with self.connection:
            self.connection.execute("INSERT INTO notes(kind,payload,created_at) VALUES('memory_search',?,?)", (encode(result), now()))
            self.conversation.complete_readonly(request_id, observation_id, "search_completed", result)

    def save_checkpoint(self, compaction=None):
        with self.connection:
            previous = self.conversation.active_checkpoint()
            row = self.connection.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
            if row is None:
                raise ValueError("Initialize the run before checkpointing its conversation")
            progress = self.progress.refresh()
            cases = self.escalations.cases()
            snapshot = {"schema_version": 1, "identity": json.loads(row[0]), "working_state": self.working_context(),
                        "progress": progress, "outcome_review": self.outcomes.context(),
                        "review": {"validity": self.escalations.validity(), "case_count": len(cases),
                                   "blocking_case_ids": [case["id"] for case in cases if case["blocking"]]},
                        "source_watermarks": {table: self.connection.execute(
                            f"SELECT COALESCE(MAX(id),0) FROM {table}").fetchone()[0] for table in SOURCE_TABLES}}
            settings = compaction or (previous["payload"].get("compaction") if previous else None)
            if settings:
                settings = compaction_settings({"compaction": settings})
                snapshot["compaction"] = settings
                snapshot["continuity"] = self.conversation.continuity(settings["review_limit"])
            return self.conversation.checkpoint(snapshot)

    def monitor_progress(self):
        with self.connection:
            return self.progress.refresh()

    def working_context(self):
        return self.contradictions.working_context(self.working.current())

    def _cancel_steps(self, turn_id, error):
        self.connection.execute(
            "UPDATE action_steps SET status=CASE WHEN status='executing' THEN 'unknown' ELSE 'cancelled' END, "
            "error=?,updated_at=? WHERE turn_id=? AND status IN ('planned','executing')",
            (error, now(), turn_id))

    def step_status(self, turn_id, index, status, observation_id=None, execution=None):
        if status not in {"executing", "completed"}:
            raise ValueError("Unsupported step status")
        with self.connection:
            self.connection.execute(
                "UPDATE action_steps SET status=?,observation_id=?,execution=?,updated_at=? "
                "WHERE turn_id=? AND step_index=?",
                (status, observation_id, encode(execution) if execution is not None else None,
                 now(), turn_id, index))
            self.conversation.step(turn_id, index, status, observation_id, execution)

    def status(self, turn_id, status, error=None):
        if status not in {"executing", "completed", "unknown", "stopped", "blocked"}:
            raise ValueError("Unsupported turn status")
        with self.connection:
            self.connection.execute("UPDATE turns SET status=?,error=?,finished_at=? WHERE id=?",
                                    (status, error, now() if status != "executing" else None, turn_id))
            if status in {"unknown", "blocked", "stopped"}:
                self._cancel_steps(turn_id, error)
            self.conversation.turn_status(turn_id, status, error)

    def note(self, kind, payload):
        with self.connection:
            self.connection.execute("INSERT INTO notes(kind,payload,created_at) VALUES(?,?,?)",
                                    (kind, encode(payload), now()))
            if kind in {"user_stop", "control_handoff", "resume_after_handoff", "recovery"}:
                self.conversation.append(kind, payload)

    def previous_observation(self):
        row = self.connection.execute("SELECT * FROM observations ORDER BY id DESC LIMIT 1").fetchone()
        return dict(row) if row else None

    def recent_turns(self, limit=6, turn_id=None):
        rows = self.connection.execute("SELECT * FROM turns ORDER BY id DESC LIMIT ?", (limit,)).fetchall() \
            if turn_id is None else self.connection.execute("SELECT * FROM turns WHERE id=?", (turn_id,)).fetchall()
        reports = self.connection.execute(
            "SELECT payload FROM notes WHERE kind='action_result' ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        execution = {value["turn_id"]: value.get("execution")
                     for value in (json.loads(row[0]) for row in reversed(reports))}
        result = []
        for row in reversed(rows):
            value = dict(row)
            value["decision"] = json.loads(value["decision"])
            value["execution"] = execution.get(value["id"])
            value["outcome_review"] = self.outcomes.latest(value["id"])
            value["contradiction_keys"] = [row[0] for row in self.connection.execute("""
                SELECT DISTINCT i.key FROM facts f JOIN fact_invalidations i ON i.fact_id=f.id
                WHERE f.observation_id=? AND NOT EXISTS(SELECT 1 FROM fact_revalidations v
                    WHERE v.fact_id=i.fact_id AND v.key=i.key)
                UNION SELECT i.key FROM working_revisions w JOIN working_invalidations i ON i.revision=w.revision
                WHERE w.turn_id=?
                UNION SELECT i.key FROM action_intents a JOIN working_invalidations i ON i.revision=a.working_revision
                WHERE a.turn_id=?
            """, (value["observation_id"], value["id"], value["id"]))]
            steps = self.connection.execute(
                "SELECT * FROM action_steps WHERE turn_id=? ORDER BY step_index", (value["id"],)).fetchall()
            value["steps"] = [dict(step, action=json.loads(step["action"]),
                                   execution=json.loads(step["execution"]) if step["execution"] else None)
                              for step in steps]
            result.append(value)
        return result

    def facts(self, include_history=False):
        rows = self.connection.execute("""
            SELECT f.*, latest.id AS latest_id FROM facts f JOIN
            (SELECT category,key,MAX(id) AS id FROM facts GROUP BY category,key) latest
            ON f.category=latest.category AND f.key=latest.key
            WHERE ? OR f.id=latest.id
            ORDER BY f.id DESC
        """, (include_history,)).fetchall()
        return self.contradictions.annotate_facts([dict(json.loads(row["payload"]), evidence_observation_id=row["observation_id"],
                     recorded_at=row["created_at"], fact_id=row["id"],
                     is_latest=row["id"] == row["latest_id"]) for row in rows])

    def events(self):
        rows = self.connection.execute("SELECT * FROM events ORDER BY id").fetchall()
        return self.contradictions.annotate_events([dict(json.loads(row["payload"]), evidence_observation_id=row["observation_id"],
                     recorded_at=row["created_at"], event_id=row["id"]) for row in rows])

    def search(self, request, observation_id):
        from .search import search_records
        from .retrieval import recent_turn_context

        turns = []
        if request["scope"] in {"turns", "all"}:
            turns = [dict(row, key=f"turn:{row['turn_id']}", record_type="turn") for row in self.connection.execute("""
                SELECT id AS turn_id, observation_id AS evidence_observation_id, created_at AS recorded_at,
                    status, COALESCE(json_extract(decision,'$.location'),'') AS location,
                    json_extract(decision,'$.strategy.kind') AS strategy_kind,
                    json_extract(decision,'$.strategy.target') AS strategy_target
                FROM turns WHERE observation_id<=? ORDER BY id DESC
            """, (observation_id,))]
        result = search_records(self.facts(include_history=request["include_history"]),
                                self.events(), request, observation_id,
                                self.contradictions.records(request["include_history"]), turns)
        for record in result["results"]:
            if record["record_type"] == "turn":
                detail = recent_turn_context(self.recent_turns(1, record["turn_id"]))[0]
                record["detail"] = detail
                record["source_review_id"] = self.connection.execute(
                    "SELECT MAX(id) FROM outcome_reviews WHERE target_turn_id=?", (record["turn_id"],)).fetchone()[0]
                if detail["contradiction_keys"]:
                    record.update(strategy_kind=None, strategy_target=None, location="", memory_validity="invalidated")
        return result

    def hashes(self, limit=6):
        return [row[0] for row in self.connection.execute(
            "SELECT image_hash FROM observations ORDER BY id DESC LIMIT ?", (limit,))]

    def export(self):
        return {"facts": self.facts(), "events": self.events(), "recent_turns": self.recent_turns(20),
                "cases": self.escalations.cases(), "validity": self.escalations.validity(),
                "working_state": self.working_context(), "outcome_review": self.outcomes.context(),
                "recent_outcome_reviews": self.outcomes.history(),
                "progress": self.monitor_progress(), "recent_progress_events": self.progress.history(),
                "contradictions": self.contradictions.context(),
                "recent_contradiction_revisions": self.contradictions.records(True, 20),
                "conversation": self.conversation.context()}

    def close(self):
        self.connection.close()
