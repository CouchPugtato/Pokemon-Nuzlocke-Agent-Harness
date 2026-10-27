"""Transactional memory, event history, and action journal for a single run."""

import json
import sqlite3
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class Memory:
    def __init__(self, path):
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS observations(
                id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, image_path TEXT NOT NULL,
                image_hash TEXT NOT NULL, timing TEXT NOT NULL);
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
        """)

    def initialize(self, config):
        # Model may change between invocations, but run identity and rules may not.
        identity = {"objective": config["objective"], "rules": config["rules"]}
        existing = self.connection.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if existing and json.loads(existing[0]) != identity:
            raise ValueError("Run objective/rules differ from saved run. Use a new run_dir.")
        with self.connection:
            self.connection.execute("INSERT OR IGNORE INTO metadata VALUES('identity', ?)", (encode(identity),))

    def observe(self, path, digest, timing):
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO observations(created_at,image_path,image_hash,timing) VALUES(?,?,?,?)",
                (now(), str(path), digest, timing))
        return cursor.lastrowid

    def recover(self):
        """Never replay actions that may already have happened."""
        rows = self.connection.execute("SELECT id,status FROM turns WHERE status IN ('planned','executing')").fetchall()
        with self.connection:
            for row in rows:
                self.connection.execute("UPDATE turns SET status='unknown', error=?, finished_at=? WHERE id=?",
                                        ("Process ended before completion was recorded", now(), row["id"]))
        return [dict(row) for row in rows]

    def plan(self, observation_id, decision):
        """Facts, events, and planned action are committed together before input."""
        with self.connection:
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
                self.connection.execute(
                    "INSERT INTO facts(category,key,payload,observation_id,created_at) VALUES(?,?,?,?,?)",
                    (fact["category"], fact["key"], encode(fact), observation_id, now()))
            cursor = self.connection.execute(
                "INSERT INTO turns(observation_id,decision,status,created_at) VALUES(?,?,'planned',?)",
                (observation_id, encode(decision), now()))
        return cursor.lastrowid

    def status(self, turn_id, status, error=None):
        if status not in {"executing", "completed", "unknown", "stopped", "blocked"}:
            raise ValueError("Unsupported turn status")
        with self.connection:
            self.connection.execute("UPDATE turns SET status=?,error=?,finished_at=? WHERE id=?",
                                    (status, error, now() if status != "executing" else None, turn_id))

    def note(self, kind, payload):
        with self.connection:
            self.connection.execute("INSERT INTO notes(kind,payload,created_at) VALUES(?,?,?)",
                                    (kind, encode(payload), now()))

    def previous_observation(self):
        row = self.connection.execute("SELECT * FROM observations ORDER BY id DESC LIMIT 1").fetchone()
        return dict(row) if row else None

    def recent_turns(self, limit=6):
        rows = self.connection.execute("SELECT * FROM turns ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        result = []
        for row in reversed(rows):
            value = dict(row)
            value["decision"] = json.loads(value["decision"])
            result.append(value)
        return result

    def facts(self):
        rows = self.connection.execute("""
            SELECT f.* FROM facts f JOIN
            (SELECT category,key,MAX(id) AS id FROM facts GROUP BY category,key) latest ON f.id=latest.id
            ORDER BY f.id DESC
        """).fetchall()
        return [dict(json.loads(row["payload"]), evidence_observation_id=row["observation_id"],
                     recorded_at=row["created_at"]) for row in rows]

    def events(self):
        rows = self.connection.execute("SELECT * FROM events ORDER BY id").fetchall()
        return [dict(json.loads(row["payload"]), evidence_observation_id=row["observation_id"],
                     recorded_at=row["created_at"]) for row in rows]

    def hashes(self, limit=6):
        return [row[0] for row in self.connection.execute(
            "SELECT image_hash FROM observations ORDER BY id DESC LIMIT ?", (limit,))]

    def export(self):
        return {"facts": self.facts(), "events": self.events(), "recent_turns": self.recent_turns(20)}

    def close(self):
        self.connection.close()
