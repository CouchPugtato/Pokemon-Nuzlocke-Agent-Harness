"""durable scene reassessments and stale scene claim markers"""

import copy
import hashlib
import json
from datetime import datetime, timezone

from jsonschema import Draft202012Validator

from ..agent.reassessment import schema


def goal_token(state):
    goal = state.get("subgoal")
    value = json.dumps({key: goal.get(key) for key in ("id", "updated_observation_id", "definition")} if goal else None,
                       sort_keys=True)
    return hashlib.sha256(value.encode()).hexdigest()


class Scenes:
    def __init__(self, connection):
        self.connection = connection
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS scene_reassessments(
                id INTEGER PRIMARY KEY, observation_id INTEGER NOT NULL UNIQUE REFERENCES observations(id),
                turn_watermark INTEGER NOT NULL, episode_id INTEGER, goal_token TEXT NOT NULL,
                context TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS stale_scene_facts(
                fact_id INTEGER PRIMARY KEY REFERENCES facts(id),
                reassessment_id INTEGER NOT NULL REFERENCES scene_reassessments(id));
            CREATE TABLE IF NOT EXISTS stale_scene_goals(
                goal_token TEXT PRIMARY KEY,
                reassessment_id INTEGER NOT NULL REFERENCES scene_reassessments(id));
            INSERT OR IGNORE INTO stale_scene_goals
                SELECT goal_token,id FROM scene_reassessments
                WHERE json_extract(payload,'$.subgoal_assessment')='contradicted' ORDER BY id;
        """)

    def latest(self):
        row = self.connection.execute("SELECT * FROM scene_reassessments ORDER BY id DESC LIMIT 1").fetchone()
        if not row:
            return None
        return dict(row, context=json.loads(row["context"]), payload=json.loads(row["payload"]))

    def watermark(self):
        return self.connection.execute("SELECT COALESCE(MAX(turn_id),0) FROM action_steps "
            "WHERE status IN ('completed','executing','unknown')").fetchone()[0]

    def controller_context(self, observation_id):
        latest = self.latest()
        if not latest:
            return None
        value = {key: latest["payload"][key] for key in ("observation_id", "visible_objects", "visible_text",
            "possible_exits", "scene", "subgoal_assessment")}
        value.update(reassessment_id=latest["id"], age_observations=observation_id - latest["observation_id"])
        return value

    def reason(self, observation_id, progress, policy):
        latest = self.latest()
        if not latest:
            return "initial_scene_check"
        if self.connection.execute("SELECT 1 FROM notes WHERE kind IN ('session_start','control_handoff','resume_after_handoff') "
                                   "AND id>? LIMIT 1", (latest["context"].get("note_watermark", 0),)).fetchone():
            return "resumed_scene_check"
        distance = self.watermark() - latest["turn_watermark"]
        if progress.get("requires_recovery") and progress.get("episode_id") != latest["episode_id"]:
            return "new_recovery_episode"
        if distance < policy["min_turn_interval"]:
            return None
        if progress.get("no_progress_attempts", 0) >= policy["no_progress_attempts"]:
            return "ineffective_inputs"
        rows = self.connection.execute("SELECT observation_id,payload FROM observation_features WHERE observation_id IN (?,?)",
                                       (latest["observation_id"], observation_id)).fetchall()
        features = {row[0]: json.loads(row[1])["scene"] for row in rows}
        if len(features) == 2:
            before, after = features[latest["observation_id"]], features[observation_id]
            if sum(abs(a - b) for a, b in zip(before, after)) / len(before) >= policy["scene_change_threshold"]:
                return "changed_scene"
        if distance >= policy["max_turn_interval"]:
            return "periodic_scene_check"
        return None

    @staticmethod
    def eligible(fact):
        return (fact["status"] == "active" and fact.get("scene_validity") != "stale" and
                (fact["category"] == "working" or fact["category"] == "world" and
                 fact["key"].startswith(("scene:", "dialogue:", "location:current"))))

    @classmethod
    def candidates(cls, facts):
        return [{key: fact[key] for key in ("fact_id", "claim", "key", "evidence_observation_id")}
                for fact in facts if cls.eligible(fact)][:12]

    def save(self, context, report, state, progress):
        Draft202012Validator(schema(context)).validate(report)
        if self.connection.execute("SELECT MAX(id) FROM observations").fetchone()[0] != context["observation_id"]:
            raise ValueError("Scene reassessment is stale")
        if goal_token(state) != context["goal_token"] or state["revision"] != context["working_revision"]:
            raise ValueError("Scene reassessment goal changed during generation")
        saved_context = dict(context, note_watermark=self.connection.execute("SELECT COALESCE(MAX(id),0) FROM notes").fetchone()[0],
            turn_history_watermark=self.connection.execute("SELECT COALESCE(MAX(id),0) FROM turns").fetchone()[0])
        cursor = self.connection.execute("INSERT INTO scene_reassessments(observation_id,turn_watermark,episode_id,goal_token,context,payload,created_at) "
            "VALUES(?,?,?,?,?,?,?)", (context["observation_id"], self.watermark(), progress.get("episode_id"),
            goal_token(state), json.dumps(saved_context), json.dumps(report), datetime.now(timezone.utc).isoformat()))
        for fact_id in report["stale_scene_fact_ids"]:
            row = self.connection.execute("SELECT category,key,payload FROM facts WHERE id=?", (fact_id,)).fetchone()
            if not row or not self.candidates([dict(json.loads(row["payload"]), fact_id=fact_id, evidence_observation_id=0)]):
                raise ValueError("Only scene-specific facts may be marked stale")
            self.connection.execute("INSERT OR IGNORE INTO stale_scene_facts VALUES(?,?)", (fact_id, cursor.lastrowid))
        if report["subgoal_assessment"] == "contradicted":
            self.connection.execute("INSERT OR IGNORE INTO stale_scene_goals VALUES(?,?)", (goal_token(state), cursor.lastrowid))
        return cursor.lastrowid

    def annotate(self, facts):
        stale = {row[0]: row[1] for row in self.connection.execute("SELECT fact_id,reassessment_id FROM stale_scene_facts")}
        latest = self.latest()
        refresh = {fact["fact_id"] for fact in facts
                   if latest and self.eligible(fact) and fact["evidence_observation_id"] < latest["observation_id"]}
        return [dict(fact, scene_validity="stale", scene_reassessment_id=stale[fact["fact_id"]])
                if fact["fact_id"] in stale else dict(fact, scene_validity="needs_refresh", scene_reassessment_id=latest["id"])
                if fact["fact_id"] in refresh else fact for fact in facts]

    def working_context(self, state):
        latest = self.latest()
        token = goal_token(state)
        marker = self.connection.execute("SELECT reassessment_id FROM stale_scene_goals WHERE goal_token=?", (token,)).fetchone()
        uncertain = latest and latest["goal_token"] == token and latest["payload"]["subgoal_assessment"] == "uncertain"
        if not marker and not uncertain:
            return state
        state = copy.deepcopy(state)
        state["location"] = None
        if state.get("subgoal"):
            state["subgoal"]["definition"] = None
            state["subgoal"]["scene_validity"] = "stale" if marker else "needs_refresh"
            state["subgoal"]["reason"] = "Previous scene interpretation requires explicit reassessment before reuse"
        state["next_check"] = "Inspect the visible scene and choose a fresh check without assuming the old subgoal applies"
        state["scene_reassessment_id"] = marker[0] if marker else latest["id"]
        return state
