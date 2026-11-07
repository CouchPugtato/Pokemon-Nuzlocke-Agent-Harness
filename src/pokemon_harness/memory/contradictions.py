"""revision dependencies and append only contradiction records"""

import hashlib
import json
from datetime import datetime, timezone

from jsonschema import Draft202012Validator

from ..agent.schemas import CONTRADICTION_UPDATE, MEMORY_DEPENDENCIES


class Contradictions:
    def __init__(self, connection, escalations):
        self.connection = connection
        self.escalations = escalations
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS contradictions(
                key TEXT PRIMARY KEY, observation_id INTEGER NOT NULL REFERENCES observations(id));
            CREATE TABLE IF NOT EXISTS contradiction_revisions(
                id INTEGER PRIMARY KEY, key TEXT NOT NULL REFERENCES contradictions(key),
                revision INTEGER NOT NULL, observation_id INTEGER NOT NULL REFERENCES observations(id),
                operation TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL,
                UNIQUE(key,revision));
            CREATE TABLE IF NOT EXISTS contradiction_facts(
                key TEXT NOT NULL REFERENCES contradictions(key), fact_id INTEGER NOT NULL REFERENCES facts(id),
                PRIMARY KEY(key,fact_id));
            CREATE TABLE IF NOT EXISTS contradiction_events(
                key TEXT NOT NULL REFERENCES contradictions(key), event_key TEXT NOT NULL REFERENCES events(key),
                PRIMARY KEY(key,event_key));
            CREATE TABLE IF NOT EXISTS fact_dependencies(
                fact_id INTEGER NOT NULL REFERENCES facts(id), source_fact_id INTEGER NOT NULL REFERENCES facts(id),
                PRIMARY KEY(fact_id,source_fact_id));
            CREATE INDEX IF NOT EXISTS dependent_facts ON fact_dependencies(source_fact_id,fact_id);
            CREATE TABLE IF NOT EXISTS fact_event_dependencies(
                fact_id INTEGER NOT NULL REFERENCES facts(id), event_key TEXT NOT NULL REFERENCES events(key),
                PRIMARY KEY(fact_id,event_key));
            CREATE INDEX IF NOT EXISTS event_dependent_facts ON fact_event_dependencies(event_key,fact_id);
            CREATE TABLE IF NOT EXISTS working_dependencies(
                revision INTEGER NOT NULL REFERENCES working_revisions(revision),
                source_fact_id INTEGER NOT NULL REFERENCES facts(id), PRIMARY KEY(revision,source_fact_id));
            CREATE INDEX IF NOT EXISTS dependent_working ON working_dependencies(source_fact_id,revision);
            CREATE TABLE IF NOT EXISTS working_event_dependencies(
                revision INTEGER NOT NULL REFERENCES working_revisions(revision), event_key TEXT NOT NULL REFERENCES events(key),
                PRIMARY KEY(revision,event_key));
            CREATE INDEX IF NOT EXISTS event_dependent_working ON working_event_dependencies(event_key,revision);
            CREATE TABLE IF NOT EXISTS fact_invalidations(
                fact_id INTEGER NOT NULL REFERENCES facts(id), key TEXT NOT NULL REFERENCES contradictions(key),
                kind TEXT NOT NULL, PRIMARY KEY(fact_id,key));
            CREATE TABLE IF NOT EXISTS working_invalidations(
                revision INTEGER NOT NULL REFERENCES working_revisions(revision),
                key TEXT NOT NULL REFERENCES contradictions(key), PRIMARY KEY(revision,key));
            CREATE TABLE IF NOT EXISTS fact_revalidations(
                fact_id INTEGER NOT NULL REFERENCES facts(id), key TEXT NOT NULL REFERENCES contradictions(key),
                resolution_id INTEGER NOT NULL REFERENCES contradiction_revisions(id), PRIMARY KEY(fact_id,key));
        """)

    def _fact(self, fact_id):
        row = self.connection.execute("SELECT * FROM facts WHERE id=?", (fact_id,)).fetchone()
        if row is None:
            raise ValueError(f"Missing fact revision {fact_id}")
        return row

    @staticmethod
    def dependencies(value):
        Draft202012Validator(MEMORY_DEPENDENCIES).validate(value)
        return value

    def _invalid(self, fact_id):
        return self.connection.execute("""
            SELECT i.key,i.kind FROM fact_invalidations i WHERE i.fact_id=? AND NOT EXISTS(
                SELECT 1 FROM fact_revalidations r WHERE r.fact_id=i.fact_id AND r.key=i.key)
        """, (fact_id,)).fetchall()

    def _event_sources(self, event_keys, observation_id):
        invalid = set()
        corrected = self._corrected_events() if event_keys else set()
        for key in event_keys:
            row = self.connection.execute("SELECT observation_id FROM events WHERE key=?", (key,)).fetchone()
            if row is None or row[0] >= observation_id or key in corrected:
                raise ValueError("Event dependencies must reference earlier saved events")
            invalid.update(row[0] for row in self.connection.execute(
                "SELECT key FROM contradiction_events WHERE event_key=?", (key,)))
        return invalid

    def _corrected_events(self):
        return {json.loads(row[0])["corrects_event_key"] for row in self.connection.execute("SELECT payload FROM events")}

    def sync_corrections(self, events):
        for event in events:
            if event["type"] != "correction":
                continue
            key = "ledger-correction:" + hashlib.sha256(event["key"].encode()).hexdigest()[:48]
            if self.connection.execute("SELECT 1 FROM contradictions WHERE key=?", (key,)).fetchone():
                continue
            row = self.connection.execute("SELECT observation_id FROM events WHERE key=?", (event["key"],)).fetchone()
            observation = row[0]
            payload = {"operation": "open", "key": key, "expected_revision": 0,
                       "reason": "Ledger correction supersedes its original event",
                       "evidence_observation_ids": [observation], "fact_ids": [],
                       "event_keys": [event["corrects_event_key"]], "replacement_facts": [], "correction_event_keys": []}
            self.connection.execute("INSERT INTO contradictions VALUES(?,?)", (key, observation))
            self.connection.execute("INSERT INTO contradiction_events VALUES(?,?)", (key, event["corrects_event_key"]))
            self._propagate(key, [])
            for revision in range(1, 3 if event["confidence"] == "high" else 2):
                if revision == 2:
                    payload.update(operation="resolve", expected_revision=1, event_keys=[], correction_event_keys=[event["key"]])
                self.connection.execute("INSERT INTO contradiction_revisions"
                    "(key,revision,observation_id,operation,payload,created_at) VALUES(?,?,?,?,?,?)",
                    (key, revision, observation, payload["operation"], json.dumps(payload), datetime.now(timezone.utc).isoformat()))

    def register_facts(self, created, links, observation_id):
        seen = set()
        for link in links:
            reference = link["fact"]
            identity = (reference["category"], reference["key"])
            if identity in seen or identity not in created:
                raise ValueError("Dependency links must name each newly written fact at most once")
            seen.add(identity)
            fact_id = created[identity]
            payload = json.loads(self._fact(fact_id)["payload"])
            if payload["basis"] == "observed" and (link["depends_on_fact_ids"] or link["depends_on_event_keys"]):
                raise ValueError("Observed facts must be independently observed rather than derived from remembered facts")
            for source in link["depends_on_fact_ids"]:
                row = self._fact(source)
                if source >= fact_id or source in created.values() or row["observation_id"] > observation_id:
                    raise ValueError("Dependencies must reference earlier saved fact revisions")
                self.connection.execute("INSERT INTO fact_dependencies VALUES(?,?)", (fact_id, source))
                for invalid in self._invalid(source):
                    self.connection.execute("INSERT OR IGNORE INTO fact_invalidations VALUES(?,?,'invalidated')",
                                            (fact_id, invalid["key"]))
            for conflict in self._event_sources(link["depends_on_event_keys"], observation_id):
                self.connection.execute("INSERT OR IGNORE INTO fact_invalidations VALUES(?,?,'invalidated')", (fact_id, conflict))
            for event_key in link["depends_on_event_keys"]:
                self.connection.execute("INSERT INTO fact_event_dependencies VALUES(?,?)", (fact_id, event_key))
        for (category, key), fact_id in created.items():
            roots = self.connection.execute("""
                SELECT DISTINCT c.key FROM contradiction_facts c JOIN facts f ON f.id=c.fact_id
                WHERE f.category=? AND f.key=? AND (SELECT operation FROM contradiction_revisions r
                WHERE r.key=c.key ORDER BY revision DESC LIMIT 1)='open'
            """, (category, key)).fetchall()
            for root in roots:
                self.connection.execute("INSERT OR IGNORE INTO fact_invalidations VALUES(?,?,'disputed')",
                                        (fact_id, root["key"]))

    def register_working(self, state, previous, update, sources, event_sources):
        if update is None:
            if sources is not None or event_sources is not None:
                raise ValueError("Working dependencies require a working state update")
            return
        old = [row[0] for row in self.connection.execute(
            "SELECT source_fact_id FROM working_dependencies WHERE revision=?", (previous["revision"],))]
        stale = self.connection.execute("SELECT 1 FROM working_invalidations WHERE revision=?",
                                         (previous["revision"],)).fetchone()
        inherited = sources is None
        event_inherited = event_sources is None
        if inherited:
            sources = old
        if event_inherited:
            event_sources = [row[0] for row in self.connection.execute(
                "SELECT event_key FROM working_event_dependencies WHERE revision=?", (previous["revision"],))]
        if stale and previous["subgoal"] and update["subgoal_change"] is None and not (inherited and event_inherited):
            raise ValueError("Refresh or retire the stale subgoal before replacing its working dependencies")
        for fact_id in sources:
            row = self._fact(fact_id)
            if row["observation_id"] > state["evidence_observation_id"]:
                raise ValueError("Working dependencies reference future facts")
            self.connection.execute("INSERT INTO working_dependencies VALUES(?,?)", (state["revision"], fact_id))
            for invalid in self._invalid(fact_id):
                self.connection.execute("INSERT OR IGNORE INTO working_invalidations VALUES(?,?)",
                                        (state["revision"], invalid["key"]))
        for key in self._event_sources(event_sources, state["evidence_observation_id"]):
            self.connection.execute("INSERT OR IGNORE INTO working_invalidations VALUES(?,?)", (state["revision"], key))
        for event_key in event_sources:
            self.connection.execute("INSERT INTO working_event_dependencies VALUES(?,?)", (state["revision"], event_key))
        if stale and update["subgoal_change"] is None and (previous["subgoal"] or (inherited and event_inherited)):
            self.connection.execute("INSERT OR IGNORE INTO working_invalidations SELECT ?,key "
                                    "FROM working_invalidations WHERE revision=?", (state["revision"], previous["revision"]))

    def _propagate(self, key, roots):
        for fact_id in roots:
            self.connection.execute("INSERT OR IGNORE INTO fact_invalidations VALUES(?,?,'disputed')", (fact_id, key))
        for fact_id in roots:
            self.connection.execute("""
                WITH RECURSIVE affected(id) AS (
                    SELECT id FROM facts WHERE category=(SELECT category FROM facts WHERE id=?)
                    AND key=(SELECT key FROM facts WHERE id=?) AND id>=?
                    UNION SELECT d.fact_id FROM fact_dependencies d JOIN affected a ON d.source_fact_id=a.id)
                INSERT OR IGNORE INTO fact_invalidations(fact_id,key,kind)
                SELECT id,?,CASE WHEN id=? THEN 'disputed' ELSE 'invalidated' END FROM affected
            """, (fact_id, fact_id, fact_id, key, fact_id))
        self.connection.execute("""
            WITH RECURSIVE affected(id) AS (
                SELECT d.fact_id FROM fact_event_dependencies d JOIN contradiction_events c ON c.event_key=d.event_key
                WHERE c.key=? UNION SELECT d.fact_id FROM fact_dependencies d JOIN affected a ON d.source_fact_id=a.id)
            INSERT OR IGNORE INTO fact_invalidations SELECT id,?,'invalidated' FROM affected
        """, (key, key))
        self.connection.execute("""
            INSERT OR IGNORE INTO working_invalidations
            SELECT DISTINCT d.revision,i.key FROM working_dependencies d JOIN fact_invalidations i
            ON i.fact_id=d.source_fact_id WHERE i.key=?
        """, (key,))
        self.connection.execute("""
            INSERT OR IGNORE INTO working_invalidations
            SELECT DISTINCT d.revision,c.key FROM working_event_dependencies d JOIN contradiction_events c
            ON c.event_key=d.event_key WHERE c.key=?
        """, (key,))

    def apply(self, updates, observation_id):
        seen = set()
        for update in updates:
            Draft202012Validator(CONTRADICTION_UPDATE).validate(update)
            key = update["key"]
            if (not key.strip() or (key.startswith("ledger-correction:") and update["operation"] == "open")
                    or not update["reason"].strip() or key in seen):
                raise ValueError("Contradiction keys and reasons must be nonempty and keys unique per decision")
            seen.add(key)
            evidence = update["evidence_observation_ids"]
            if observation_id not in evidence:
                raise ValueError("Contradictions require the current observation as evidence")
            for value in evidence:
                if value > observation_id or not self.connection.execute(
                        "SELECT 1 FROM observations WHERE id=?", (value,)).fetchone():
                    raise ValueError("Contradiction references unavailable observation evidence")
            current = self.connection.execute("SELECT * FROM contradiction_revisions WHERE key=? "
                                               "ORDER BY revision DESC LIMIT 1", (key,)).fetchone()
            if update["expected_revision"] != (current["revision"] if current else 0):
                raise ValueError("Contradiction update refers to a stale revision")
            replacements = []
            if update["operation"] == "open":
                if current or not (update["fact_ids"] or update["event_keys"]):
                    raise ValueError("Open a new contradiction with at least one existing fact or event")
                if update["replacement_facts"] or update["correction_event_keys"]:
                    raise ValueError("Opening contradictions cannot claim a resolution")
                for fact_id in update["fact_ids"]:
                    if self._fact(fact_id)["observation_id"] >= observation_id:
                        raise ValueError("Dispute an earlier fact rather than a fact just observed")
                for event_key in update["event_keys"]:
                    row = self.connection.execute("SELECT observation_id FROM events WHERE key=?", (event_key,)).fetchone()
                    if row is None or row[0] >= observation_id or event_key in self._corrected_events():
                        raise ValueError("Dispute an earlier event that belongs to this run")
                self.connection.execute("INSERT INTO contradictions VALUES(?,?)", (key, observation_id))
                for fact_id in update["fact_ids"]:
                    self.connection.execute("INSERT INTO contradiction_facts VALUES(?,?)", (key, fact_id))
                for event_key in update["event_keys"]:
                    self.connection.execute("INSERT INTO contradiction_events VALUES(?,?)", (key, event_key))
                    digest = hashlib.sha256((key + '\0' + event_key).encode()).hexdigest()
                    self.escalations.open_in_transaction("contradiction:" + digest,
                        "Disputed historical event requires review", evidence,
                        {"contradiction_key": key, "event_key": event_key,
                         "missing_fact": update["reason"]})
                self._propagate(key, update["fact_ids"])
            else:
                if not current or current["operation"] != "open" or update["fact_ids"] or update["event_keys"]:
                    raise ValueError("Resolve an open contradiction without changing its original targets")
                if observation_id <= current["observation_id"]:
                    raise ValueError("Resolution requires a later fresh observation")
                roots = self.connection.execute("SELECT f.* FROM facts f JOIN contradiction_facts c "
                                                "ON c.fact_id=f.id WHERE c.key=?", (key,)).fetchall()
                required = {(row["category"], row["key"]) for row in roots}
                supplied = {(ref["category"], ref["key"]) for ref in update["replacement_facts"]}
                if required != supplied or len(supplied) != len(update["replacement_facts"]):
                    raise ValueError("Resolve every disputed fact key with one fresh replacement")
                for category, fact_key in supplied:
                    row = self.connection.execute("SELECT * FROM facts WHERE category=? AND key=? "
                                                  "ORDER BY id DESC LIMIT 1", (category, fact_key)).fetchone()
                    payload = json.loads(row["payload"])
                    if (row["observation_id"] != observation_id or payload["basis"] != "observed"
                            or payload["confidence"] != "high" or payload["status"] != "active"
                            or any(invalid["key"] != key for invalid in self._invalid(row["id"]))):
                        raise ValueError("Resolution needs fresh high confidence observed replacements without other disputes")
                    replacements.append(row["id"])
                targets = {row[0] for row in self.connection.execute(
                    "SELECT event_key FROM contradiction_events WHERE key=?", (key,))}
                covered = set()
                for correction_key in update["correction_event_keys"]:
                    row = self.connection.execute("SELECT * FROM events WHERE key=?", (correction_key,)).fetchone()
                    event = json.loads(row["payload"]) if row else {}
                    if (not row or row["observation_id"] != observation_id or event.get("type") != "correction"
                            or event.get("confidence") != "high" or event.get("corrects_event_key") not in targets
                            or correction_key in self._corrected_events()):
                        raise ValueError("Event disputes require fresh high confidence corrections for their original targets")
                    covered.add(event["corrects_event_key"])
                if covered != targets:
                    raise ValueError("Resolve every disputed event through an explicit ledger correction")
            cursor = self.connection.execute("INSERT INTO contradiction_revisions"
                "(key,revision,observation_id,operation,payload,created_at) VALUES(?,?,?,?,?,?)",
                (key, (current["revision"] if current else 0) + 1, observation_id, update["operation"],
                 json.dumps(update), datetime.now(timezone.utc).isoformat()))
            for fact_id in replacements:
                self.connection.execute("INSERT INTO fact_revalidations VALUES(?,?,?)", (fact_id, key, cursor.lastrowid))

    def records(self, include_history=False, limit=None, offset=0):
        sql = "SELECT r.* FROM contradiction_revisions r WHERE ? OR r.revision=" \
              "(SELECT MAX(revision) FROM contradiction_revisions latest WHERE latest.key=r.key) ORDER BY r.id DESC"
        params = [include_history]
        if limit is not None:
            if not 1 <= limit <= 100 or offset < 0:
                raise ValueError("Contradiction history requires limit 1..100 and nonnegative offset")
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        return [dict(json.loads(row["payload"]), revision=row["revision"],
                     contradiction_revision_id=row["id"], evidence_observation_id=row["observation_id"],
                     recorded_at=row["created_at"]) for row in self.connection.execute(sql, params)]

    def context(self):
        count = self.connection.execute("""
            SELECT COUNT(*) FROM contradiction_revisions r WHERE operation='open' AND revision=
            (SELECT MAX(revision) FROM contradiction_revisions latest WHERE latest.key=r.key)
        """).fetchone()[0]
        rows = self.connection.execute("""
            SELECT r.* FROM contradiction_revisions r WHERE operation='open' AND revision=
            (SELECT MAX(revision) FROM contradiction_revisions latest WHERE latest.key=r.key)
            ORDER BY id DESC LIMIT 10
        """).fetchall()
        return {"open_count": count, "open": [dict(json.loads(row["payload"]), revision=row["revision"])
                                               for row in rows]}

    def history(self, limit=20, offset=0):
        return self.records(True, limit, offset)

    def annotate_facts(self, facts):
        invalid, sources, events = {}, {}, {}
        for start in range(0, len(facts), 500):
            ids = [fact["fact_id"] for fact in facts[start:start + 500]]
            placeholders = ",".join("?" for _ in ids)
            for row in self.connection.execute(f"""
                SELECT i.* FROM fact_invalidations i WHERE fact_id IN ({placeholders}) AND NOT EXISTS(
                    SELECT 1 FROM fact_revalidations r WHERE r.fact_id=i.fact_id AND r.key=i.key)
            """, ids):
                invalid.setdefault(row["fact_id"], []).append(row)
            for row in self.connection.execute(
                    f"SELECT * FROM fact_dependencies WHERE fact_id IN ({placeholders}) ORDER BY source_fact_id", ids):
                sources.setdefault(row["fact_id"], []).append(row["source_fact_id"])
            for row in self.connection.execute(
                    f"SELECT * FROM fact_event_dependencies WHERE fact_id IN ({placeholders}) ORDER BY event_key", ids):
                events.setdefault(row["fact_id"], []).append(row["event_key"])
        for fact in facts:
            disputed = invalid.get(fact["fact_id"], [])
            fact["memory_validity"] = ("disputed" if any(row["kind"] == "disputed" for row in disputed)
                                       else "invalidated" if disputed else "supported" if fact["basis"] == "observed" else "inferred")
            fact["contradiction_keys"] = [row["key"] for row in disputed]
            fact["depends_on_fact_ids"] = sources.get(fact["fact_id"], [])
            fact["depends_on_event_keys"] = events.get(fact["fact_id"], [])
        return facts

    def working_context(self, state):
        keys = [row[0] for row in self.connection.execute(
            "SELECT key FROM working_invalidations WHERE revision=? ORDER BY key", (state["revision"],))]
        if keys:
            state["memory_validity"] = "invalidated"
            state["contradiction_keys"] = keys
            state["location"] = None
            state["uncertainties"] = []
            state["next_check"] = "Reinspect invalidated dependencies and refresh or retire the stale subgoal"
            if state["subgoal"]:
                state["subgoal"] = {"id": state["subgoal"]["id"], "status": state["subgoal"]["status"], "definition": None}
        return state

    def annotate_events(self, events):
        conflicts = {}
        for start in range(0, len(events), 500):
            keys = [event["key"] for event in events[start:start + 500]]
            placeholders = ",".join("?" for _ in keys)
            for row in self.connection.execute(
                    f"SELECT * FROM contradiction_events WHERE event_key IN ({placeholders}) ORDER BY key", keys):
                conflicts.setdefault(row["event_key"], []).append(row["key"])
        for event in events:
            keys = conflicts.get(event["key"], [])
            event["memory_validity"] = "disputed" if keys else "supported"
            event["contradiction_keys"] = keys
        return events
