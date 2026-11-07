"""durable requests execution messages and atomic checkpoint boundaries"""

import base64
import hashlib
import json
from datetime import datetime, timezone


PENDING = ("requested", "responded", "validated", "committed", "executing")
SOURCE_TABLES = ("facts", "events", "turns", "outcome_reviews", "contradiction_revisions",
                 "progress_events", "escalation_cases", "case_resolutions")


def timestamp():
    return datetime.now(timezone.utc).isoformat()


class Conversation:
    def __init__(self, connection):
        self.connection = connection
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS conversation_prompts(key TEXT PRIMARY KEY, content TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS conversation_requests(
                id INTEGER PRIMARY KEY, observation_id INTEGER NOT NULL REFERENCES observations(id),
                status TEXT NOT NULL, messages TEXT NOT NULL, budget TEXT NOT NULL, model TEXT NOT NULL,
                request_hash TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS conversation_images(
                request_id INTEGER NOT NULL REFERENCES conversation_requests(id),
                message_index INTEGER NOT NULL, block_index INTEGER NOT NULL,
                observation_id INTEGER NOT NULL REFERENCES observations(id),
                PRIMARY KEY(request_id,message_index,block_index));
            CREATE TABLE IF NOT EXISTS conversation_request_prompts(
                request_id INTEGER NOT NULL REFERENCES conversation_requests(id),
                message_index INTEGER NOT NULL, prompt_key TEXT NOT NULL REFERENCES conversation_prompts(key),
                PRIMARY KEY(request_id,message_index));
            CREATE TABLE IF NOT EXISTS conversation_turns(
                turn_id INTEGER PRIMARY KEY REFERENCES turns(id),
                request_id INTEGER NOT NULL UNIQUE REFERENCES conversation_requests(id));
            CREATE TABLE IF NOT EXISTS conversation_events(
                id INTEGER PRIMARY KEY, request_id INTEGER REFERENCES conversation_requests(id),
                turn_id INTEGER REFERENCES turns(id), observation_id INTEGER REFERENCES observations(id),
                kind TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS conversation_request_events ON conversation_events(request_id,id);
            CREATE INDEX IF NOT EXISTS observation_image_lookup ON observations(image_hash,id);
            CREATE TABLE IF NOT EXISTS conversation_checkpoints(
                id INTEGER PRIMARY KEY, schema_version INTEGER NOT NULL,
                boundary_id INTEGER REFERENCES conversation_events(id),
                observation_id INTEGER REFERENCES observations(id),
                working_revision INTEGER REFERENCES working_revisions(revision),
                payload TEXT NOT NULL, payload_hash TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS conversation_checkpoint_tail(
                checkpoint_id INTEGER NOT NULL REFERENCES conversation_checkpoints(id),
                event_id INTEGER NOT NULL REFERENCES conversation_events(id), PRIMARY KEY(checkpoint_id,event_id));
            CREATE TABLE IF NOT EXISTS conversation_active(
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                checkpoint_id INTEGER NOT NULL REFERENCES conversation_checkpoints(id));
            INSERT OR IGNORE INTO metadata(key,value)
                SELECT 'conversation_legacy_turn_id',COALESCE(MAX(id),0) FROM turns;
        """)

    def append(self, kind, payload, observation_id=None, request_id=None, turn_id=None):
        cursor = self.connection.execute("""
            INSERT INTO conversation_events(request_id,turn_id,observation_id,kind,payload,created_at)
            VALUES(?,?,?,?,?,?)
        """, (request_id, turn_id, observation_id, kind, json.dumps(payload, ensure_ascii=False), timestamp()))
        return cursor.lastrowid

    def begin(self, request, observation_id, budget, model):
        archived, references, prompts = [], [], []
        for index, message in enumerate(request):
            saved = dict(message)
            if message["role"] == "system":
                content = message["content"]
                key = hashlib.sha256(content.encode()).hexdigest()
                self.connection.execute("INSERT OR IGNORE INTO conversation_prompts VALUES(?,?)", (key, content))
                saved = {"role": "system", "content_reference": key}
                prompts.append((index, key))
            elif isinstance(message["content"], list):
                saved["content"] = []
                for block_index, block in enumerate(message["content"]):
                    if block["type"] != "image_url":
                        saved["content"].append(dict(block))
                        continue
                    url = block["image_url"]["url"]
                    prefix = "data:image/png;base64,"
                    if not url.startswith(prefix):
                        raise ValueError("Conversation images must be archived run observations")
                    digest = hashlib.sha256(base64.b64decode(url[len(prefix):], validate=True)).hexdigest()
                    row = self.connection.execute("SELECT id FROM observations WHERE image_hash=? AND id<=? "
                                                   "ORDER BY id DESC LIMIT 1", (digest, observation_id)).fetchone()
                    if row is None:
                        raise ValueError("Conversation image has no matching observation in this run")
                    saved["content"].append({"type": "observation_image", "observation_id": row[0], "image_hash": digest})
                    references.append((index, block_index, row[0]))
            archived.append(saved)
        fingerprint = hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        cursor = self.connection.execute("""
            INSERT INTO conversation_requests(observation_id,status,messages,budget,model,request_hash,created_at)
            VALUES(?,'requested',?,?,?,?,?)
        """, (observation_id, json.dumps(archived, ensure_ascii=False), json.dumps(budget), model, fingerprint, timestamp()))
        request_id = cursor.lastrowid
        for index, key in prompts:
            self.connection.execute("INSERT INTO conversation_request_prompts VALUES(?,?,?)", (request_id, index, key))
        for index, block, source in references:
            self.connection.execute("INSERT INTO conversation_images VALUES(?,?,?,?)", (request_id, index, block, source))
        self.append("request_started", {"request_hash": fingerprint}, observation_id, request_id)
        return request_id

    def _request(self, request_id):
        row = self.connection.execute("SELECT * FROM conversation_requests WHERE id=?", (request_id,)).fetchone()
        if row is None:
            raise ValueError("Conversation request does not belong to this run")
        return row

    def respond(self, request_id, decision, trace_path):
        row = self._request(request_id)
        if row["status"] != "requested":
            raise ValueError("Conversation request is no longer accepting a response")
        self.connection.execute("UPDATE conversation_requests SET status='responded' WHERE id=?", (request_id,))
        self.append("assistant_decision", {"decision": decision, "trace_path": trace_path}, row["observation_id"], request_id)

    def validate(self, request_id):
        row = self._request(request_id)
        if row["status"] != "responded":
            raise ValueError("Only a received response can be validated")
        self.connection.execute("UPDATE conversation_requests SET status='validated' WHERE id=?", (request_id,))
        self.append("decision_validated", {}, row["observation_id"], request_id)

    def _proposal(self, request_id, observation_id):
        row = self._request(request_id)
        if row["status"] != "validated" or row["observation_id"] != observation_id:
            raise ValueError("Only the validated decision for this observation can be committed")
        response = self.connection.execute("SELECT payload FROM conversation_events WHERE request_id=? "
                                           "AND kind='assistant_decision' ORDER BY id DESC LIMIT 1", (request_id,)).fetchone()
        return json.loads(response[0])["decision"]

    def link(self, request_id, turn_id, observation_id, decision):
        if self._proposal(request_id, observation_id) != decision:
            raise ValueError("Committed decision differs from the journaled response")
        self.connection.execute("INSERT INTO conversation_turns VALUES(?,?)", (turn_id, request_id))
        self.connection.execute("UPDATE conversation_requests SET status='committed' WHERE id=?", (request_id,))
        self.append("plan_committed", {}, observation_id, request_id, turn_id)

    def turn_status(self, turn_id, status, error=None):
        row = self.connection.execute("SELECT request_id FROM conversation_turns WHERE turn_id=?", (turn_id,)).fetchone()
        request_id = row[0] if row else None
        if request_id is not None:
            self.connection.execute("UPDATE conversation_requests SET status=? WHERE id=?", (status, request_id))
        self.append("turn_status", {"status": status, "error": error}, request_id=request_id, turn_id=turn_id)

    def step(self, turn_id, index, status, observation_id, execution):
        action = self.connection.execute("SELECT action FROM action_steps WHERE turn_id=? AND step_index=?",
                                         (turn_id, index)).fetchone()
        if action is None:
            raise ValueError("Conversation execution references an unavailable action step")
        row = self.connection.execute("SELECT request_id FROM conversation_turns WHERE turn_id=?", (turn_id,)).fetchone()
        self.append("input_step", {"step_index": index, "status": status, "execution": execution,
                                   "action": (execution or {}).get("effective_action", json.loads(action[0]))},
                    observation_id, row[0] if row else None, turn_id)

    def complete_readonly(self, request_id, observation_id, kind, payload):
        proposal = self._proposal(request_id, observation_id)
        if kind not in {"search_completed", "review_completed"}:
            raise ValueError("Invalid read only conversation result")
        expected = "search_memory" if kind == "search_completed" else "review_outcomes"
        if proposal["action"]["kind"] != expected:
            raise ValueError("Conversation result does not match the validated request kind")
        if kind == "search_completed" and proposal["memory_search"] != payload["request"]:
            raise ValueError("Search result differs from the validated conversation query")
        if kind == "review_completed" and proposal["outcome_reviews"] != payload["reviews"]:
            raise ValueError("Outcome reviews differ from the validated conversation decision")
        self.connection.execute("UPDATE conversation_requests SET status=? WHERE id=?", (kind, request_id))
        self.append(kind, payload, observation_id, request_id)

    def fail(self, request_id, status, reason):
        row = self._request(request_id)
        if row["status"] not in {"requested", "responded", "validated"}:
            return
        if status not in {"failed", "interrupted"}:
            raise ValueError("Invalid unfinished conversation disposition")
        self.connection.execute("UPDATE conversation_requests SET status=? WHERE id=?", (status, request_id))
        self.append(status, {"reason": reason}, row["observation_id"], request_id)

    def recover(self):
        rows = self.connection.execute("SELECT id FROM conversation_requests WHERE status IN "
                                       "('requested','responded','validated') ORDER BY id").fetchall()
        for row in rows:
            self.fail(row[0], "interrupted", "Harness ended before this decision was committed")
        return [row[0] for row in rows]

    def checkpoint(self, snapshot):
        if self.connection.execute("SELECT 1 FROM conversation_requests WHERE status IN "
                                   "('requested','responded','validated','committed','executing') LIMIT 1").fetchone():
            raise ValueError("Cannot checkpoint an unfinished conversation request")
        if self.connection.execute("SELECT 1 FROM turns WHERE status IN ('planned','executing') LIMIT 1").fetchone():
            raise ValueError("Cannot checkpoint a pending controller plan")
        boundary = self.connection.execute("SELECT MAX(id) FROM conversation_events").fetchone()[0]
        observation = self.connection.execute("SELECT MAX(id) FROM observations").fetchone()[0]
        revision = snapshot["working_state"]["revision"] or None
        payload = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        cursor = self.connection.execute("INSERT INTO conversation_checkpoints"
            "(schema_version,boundary_id,observation_id,working_revision,payload,payload_hash,created_at) VALUES(1,?,?,?,?,?,?)",
            (boundary, observation, revision, payload, digest, timestamp()))
        checkpoint_id = cursor.lastrowid
        self.connection.execute("INSERT INTO conversation_checkpoint_tail SELECT ?,id FROM conversation_events "
                                "ORDER BY id DESC LIMIT 12", (checkpoint_id,))
        self.connection.execute("INSERT OR REPLACE INTO conversation_active VALUES(1,?)", (checkpoint_id,))
        return checkpoint_id

    def active_checkpoint(self):
        row = self.connection.execute("SELECT c.* FROM conversation_checkpoints c JOIN conversation_active a "
                                       "ON a.checkpoint_id=c.id WHERE a.singleton=1").fetchone()
        if row is None:
            return None
        if row["schema_version"] != 1:
            raise ValueError("Unsupported conversation checkpoint schema")
        if hashlib.sha256(row["payload"].encode()).hexdigest() != row["payload_hash"]:
            raise ValueError("Conversation checkpoint fingerprint does not match its payload")
        payload = json.loads(row["payload"])
        identity = self.connection.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if payload.get("schema_version") != 1 or not identity or payload.get("identity") != json.loads(identity[0]):
            raise ValueError("Conversation checkpoint identity is inconsistent with this run")
        sources = payload.get("source_watermarks", {})
        if set(sources) != set(SOURCE_TABLES):
            raise ValueError("Conversation checkpoint source tables are invalid")
        for table, source_id in sources.items():
            if type(source_id) is not int or source_id < 0 or (source_id and not self.connection.execute(
                    f"SELECT 1 FROM {table} WHERE id=?", (source_id,)).fetchone()):
                raise ValueError("Conversation checkpoint references unavailable source records")
        if payload.get("working_state", {}).get("revision") != (row["working_revision"] or 0):
            raise ValueError("Conversation checkpoint working revision is inconsistent")
        return dict(row, payload=payload)

    def context(self, limit=12):
        if not 1 <= limit <= 50:
            raise ValueError("Conversation recall limit must be between one and fifty")
        checkpoint = self.active_checkpoint()
        boundary = (checkpoint["boundary_id"] or 0) if checkpoint else 0
        ids = [row[0] for row in self.connection.execute(
            "SELECT id FROM conversation_events WHERE id>? ORDER BY id DESC LIMIT ?", (boundary, limit))]
        if checkpoint:
            ids.extend(row[0] for row in self.connection.execute(
                "SELECT event_id FROM conversation_checkpoint_tail WHERE checkpoint_id=?", (checkpoint["id"],)))
        recent = [self._project(self.connection.execute("SELECT * FROM conversation_events WHERE id=?", (event_id,)).fetchone())
                  for event_id in sorted(set(ids))[-limit:]]
        interrupted = self.connection.execute("SELECT COUNT(*) FROM conversation_requests WHERE status='interrupted'").fetchone()[0]
        return {"checkpoint": {key: checkpoint[key] for key in ("id", "boundary_id", "observation_id", "working_revision", "created_at")}
                if checkpoint else None, "recent": recent, "interrupted_request_count": interrupted,
                "legacy_turn_watermark": int(self.connection.execute(
                    "SELECT value FROM metadata WHERE key='conversation_legacy_turn_id'").fetchone()[0]),
                "pending_request_count": self.connection.execute("SELECT COUNT(*) FROM conversation_requests WHERE status IN "
                    "('requested','responded','validated','committed','executing')").fetchone()[0]}

    @staticmethod
    def _project(row):
        payload = json.loads(row["payload"])
        kind = row["kind"]
        if kind == "assistant_decision":
            value = payload["decision"]
            payload = {"proposed_action": value.get("action"), "additional_actions": value.get("additional_actions", []),
                       "proposal_only": True} if isinstance(value, dict) else {"invalid_response": True}
        elif kind == "search_completed":
            payload = {"request": payload["request"], "total_matches": payload["total_matches"],
                       "result_ids": [{key: record[key] for key in ("record_type", "fact_id", "event_id", "contradiction_revision_id")
                                      if key in record} for record in payload["results"]]}
        elif kind == "input_step":
            execution = payload.get("execution") or {}
            payload = {"step_index": payload["step_index"], "status": payload["status"],
                       "effective_action": execution.get("effective_action", payload.get("action"))}
        elif kind == "review_completed":
            payload = {"reviewed_turn_ids": [review["turn_id"] for review in payload["reviews"]]}
        elif kind in {"failed", "interrupted"}:
            payload = {"disposition": kind, "fresh_observation_required": True}
        elif kind == "turn_status":
            payload = {key: value[:200] if isinstance(value, str) else value for key, value in payload.items()}
        return {"id": row["id"], "role": "assistant" if kind == "assistant_decision" else "user",
                "kind": kind, "request_id": row["request_id"], "turn_id": row["turn_id"],
                "observation_id": row["observation_id"], "content": payload}

    def history(self, limit=20, offset=0):
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("Conversation history requires limit 1..100 and nonnegative offset")
        return [dict(row, payload=json.loads(row["payload"])) for row in self.connection.execute(
            "SELECT * FROM conversation_events ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset))]

    def request(self, request_id):
        row = self._request(request_id)
        prompts = {source["key"]: source["content"] for source in self.connection.execute(
            "SELECT p.* FROM conversation_prompts p JOIN conversation_request_prompts r ON r.prompt_key=p.key WHERE r.request_id=?",
            (request_id,))}
        return dict(row, messages=json.loads(row["messages"]), budget=json.loads(row["budget"]), prompts=prompts)
