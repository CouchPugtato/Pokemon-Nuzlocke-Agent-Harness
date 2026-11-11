"""durable progress monitoring and bounded recovery episodes"""

import json
from collections import Counter
from datetime import datetime, timezone

from ..control.loop_detection import cycle_signals, input_signature


DEFAULTS = {"window": 32, "no_progress_attempts": 12, "strategy_attempts": 3,
            "cycle_repetitions": 3, "max_cycle_period": 8, "similarity_threshold": 3,
            "max_recovery_attempts": 2}


def progress_settings(config):
    supplied = config.get("progress", {})
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
        raise ValueError("progress must contain only supported progress settings")
    policy = dict(DEFAULTS, **supplied)
    bounds = {"window": (8, 128), "no_progress_attempts": (1, 128), "strategy_attempts": (1, 32),
              "cycle_repetitions": (2, 8), "max_cycle_period": (1, 8),
              "similarity_threshold": (0, 16), "max_recovery_attempts": (1, 5)}
    for key, (low, high) in bounds.items():
        if type(policy[key]) is not int or not low <= policy[key] <= high:
            raise ValueError(f"progress.{key} must be an integer between {low} and {high}")
    if max(policy["no_progress_attempts"], policy["strategy_attempts"],
           policy["cycle_repetitions"] * policy["max_cycle_period"]) > policy["window"]:
        raise ValueError("progress.window must cover attempt limits and configured cycles")
    return policy


class Progress:
    def __init__(self, connection, outcomes):
        self.connection = connection
        self.outcomes = outcomes
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS observation_features(
                observation_id INTEGER PRIMARY KEY REFERENCES observations(id), payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS progress_episodes(
                id INTEGER PRIMARY KEY, start_turn_id INTEGER NOT NULL REFERENCES turns(id),
                status TEXT NOT NULL, signals TEXT NOT NULL, case_id INTEGER REFERENCES escalation_cases(id));
            CREATE TABLE IF NOT EXISTS recovery_attempts(
                turn_id INTEGER PRIMARY KEY REFERENCES turns(id), episode_id INTEGER NOT NULL REFERENCES progress_episodes(id),
                signature TEXT NOT NULL, strategy TEXT NOT NULL, explanation TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS recovery_episode ON recovery_attempts(episode_id);
            CREATE TABLE IF NOT EXISTS progress_events(
                id INTEGER PRIMARY KEY, episode_id INTEGER NOT NULL REFERENCES progress_episodes(id),
                kind TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL);
        """)

    def configure(self, config):
        policy = progress_settings(config)
        row = self.connection.execute("SELECT value FROM metadata WHERE key='progress_policy'").fetchone()
        if row and json.loads(row[0]) != policy:
            raise ValueError("Progress policy differs from saved run; use its saved settings or a new run")
        self.connection.execute("INSERT OR IGNORE INTO metadata VALUES('progress_policy',?)", (json.dumps(policy),))

    def policy(self):
        row = self.connection.execute("SELECT value FROM metadata WHERE key='progress_policy'").fetchone()
        return json.loads(row[0]) if row else dict(DEFAULTS)

    def _event(self, episode_id, kind, payload):
        self.connection.execute("INSERT INTO progress_events(episode_id,kind,payload,created_at) VALUES(?,?,?,?)",
                                (episode_id, kind, json.dumps(payload), datetime.now(timezone.utc).isoformat()))

    def _samples(self, window):
        turns = self.connection.execute("""
            SELECT t.id FROM turns t WHERE EXISTS(
                SELECT 1 FROM action_steps s WHERE s.turn_id=t.id AND s.status IN ('completed','executing','unknown'))
            ORDER BY t.id DESC LIMIT ?
        """, (window,)).fetchall()
        samples, frames = [], []
        for turn in reversed(turns):
            plan = self.outcomes.plan_context(turn[0])
            review = self.outcomes.latest(turn[0])
            progress = False
            if review and plan["subgoal_memory_validity"] != "invalidated":
                for step in review["steps"]:
                    if step["assessment"] != "progress_observed":
                        continue
                    before = self.connection.execute("SELECT image_hash FROM observations WHERE id=?",
                                                     (step["before_observation_id"],)).fetchone()
                    after = self.connection.execute("SELECT image_hash FROM observations WHERE id=?",
                                                    (step["after_observation_id"],)).fetchone()
                    progress |= bool(before and after and before[0] != after[0])
            actions = []
            reviewed = {step["step_index"]: step for step in review["steps"]} if review else {}
            for step in plan["steps"]:
                if step["input_status"] not in {"completed", "executing", "unknown"}:
                    continue
                actions.append(step["action"])
                after_id = reviewed.get(step["step_index"], step)["after_observation_id"]
                row = self.connection.execute("""
                    SELECT o.image_hash,f.payload FROM observations o LEFT JOIN observation_features f
                    ON f.observation_id=o.id WHERE o.id=?
                """, (after_id,)).fetchone()
                frames.append({"observation_id": after_id,
                               "turn_id": turn[0],
                               "image_hash": row[0] if row else None,
                               "features": json.loads(row[1]) if row and row[1] else None,
                               "signature": input_signature([step["action"]])})
            samples.append({"turn_id": turn[0], "progress": progress, "reviewed": review is not None,
                            "signature": input_signature(actions)})
        return samples, frames[-window:]

    def refresh(self):
        """reconciling derived counters and durable episodes in the caller transaction"""
        policy = self.policy()
        samples, frames = self._samples(policy["window"])
        signals = cycle_signals(frames, policy["cycle_repetitions"], policy["max_cycle_period"],
                                policy["similarity_threshold"])
        failures = []
        for sample in reversed(samples):
            if sample["progress"]:
                break
            failures.append(sample)
        repeats = max(Counter(sample["signature"] for sample in failures).values(), default=0)
        repeating_inputs = any(signal["kind"] == "input_cycle" for signal in signals)
        triggers = [signal for signal in signals if signal["kind"] == "exact_visual" or
                    (signal["kind"] == "similar_visual" and repeating_inputs) or
                    len(failures) >= policy["strategy_attempts"]]
        if len(failures) >= policy["no_progress_attempts"]:
            triggers.append({"kind": "no_progress_budget"})
        if repeats >= policy["strategy_attempts"]:
            triggers.append({"kind": "repeated_inputs"})
        allowance = self.repetition.context(self.working_context()) if hasattr(self, "repetition") else None
        suppressed = []
        if allowance and allowance["active"]:
            covered = {row[0] for row in self.connection.execute(
                "SELECT turn_id FROM repetition_uses WHERE subgoal_id=?", (allowance["subgoal_id"],))}
            for signal in triggers:
                if signal["kind"] not in {"input_cycle", "similar_visual", "repeated_inputs"}:
                    continue
                count = signal.get("period", 1) * signal.get("repetitions", policy["strategy_attempts"])
                tail = frames[-count:]
                if len(tail) >= count and all(frame["turn_id"] in covered for frame in tail):
                    suppressed.append(signal)
            triggers = [signal for signal in triggers if signal not in suppressed]
        episode = self.connection.execute("SELECT * FROM progress_episodes ORDER BY id DESC LIMIT 1").fetchone()
        latest_turn = samples[-1]["turn_id"] if samples else 0
        last_progress = max((sample["turn_id"] for sample in samples if sample["progress"]), default=0)
        if episode and episode["status"] == "stopped":
            resolution = self.connection.execute(
                "SELECT outcome,allow_continue FROM case_resolutions WHERE case_id=? ORDER BY id DESC LIMIT 1",
                (episode["case_id"],)).fetchone()
            if resolution and resolution[0] != "invalid" and resolution[1]:
                self.connection.execute("UPDATE progress_episodes SET status='overridden' WHERE id=?", (episode["id"],))
                self._event(episode["id"], "user_override", {"case_id": episode["case_id"]})
                episode = None
                triggers = triggers or [{"kind": "explicit_user_continuation"}]
        if episode and episode["status"] == "active" and last_progress > episode["start_turn_id"] and not triggers:
            self.connection.execute("UPDATE progress_episodes SET status='recovered' WHERE id=?", (episode["id"],))
            self._event(episode["id"], "recovered", {"evidence_turn_id": last_progress})
            episode = self.connection.execute("SELECT * FROM progress_episodes WHERE id=?", (episode["id"],)).fetchone()
        if triggers and latest_turn and (not episode or episode["status"] == "recovered"):
            cursor = self.connection.execute(
                "INSERT INTO progress_episodes(start_turn_id,status,signals) VALUES(?,'active',?)",
                (latest_turn, json.dumps(triggers)))
            self._event(cursor.lastrowid, "suspected_loop", {"signals": triggers})
            self._event(cursor.lastrowid, "reassess", {"turn_id": latest_turn})
            episode = self.connection.execute("SELECT * FROM progress_episodes WHERE id=?", (cursor.lastrowid,)).fetchone()
        used, signatures = 0, []
        pending = self.outcomes.pending_count()
        if episode:
            attempts = self.connection.execute(
                "SELECT signature FROM recovery_attempts WHERE episode_id=?", (episode["id"],)).fetchall()
            used = len(attempts)
            signatures = [json.loads(row[0]) for row in attempts]
            if episode["status"] == "active" and used >= policy["max_recovery_attempts"] and pending == 0:
                evidence = list(dict.fromkeys(frame["observation_id"] for frame in frames if frame["observation_id"] is not None))
                key = f"progress:{episode['id']}"
                self.connection.execute("""
                    INSERT OR IGNORE INTO escalation_cases(case_key,reason,details,evidence_ids,created_at)
                    VALUES(?,?,?,?,?)
                """, (key, "Progress recovery budget exhausted", json.dumps({"kind": "stalled_progress",
                      "episode_id": episode["id"], "recovery_attempts": used,
                      "missing_fact": "Evidence of safe advancement beyond the repeated state"}),
                      json.dumps(evidence), datetime.now(timezone.utc).isoformat()))
                case = self.connection.execute("SELECT id FROM escalation_cases WHERE case_key=?", (key,)).fetchone()[0]
                self.connection.execute("UPDATE progress_episodes SET status='stopped',case_id=? WHERE id=?",
                                        (case, episode["id"]))
                self._event(episode["id"], "stopped", {"case_id": case})
                episode = self.connection.execute("SELECT * FROM progress_episodes WHERE id=?", (episode["id"],)).fetchone()
        mode = "normal"
        if episode:
            mode = {"active": "reassess", "recovered": "recovered", "stopped": "stopped"}[episode["status"]]
            if episode["status"] == "active" and used and pending:
                mode = "test_alternative"
        return {"mode": mode, "episode_id": episode["id"] if episode else None,
                "repetition_allowance": allowance, "suppressed_signals": suppressed,
                "case_id": episode["case_id"] if episode else None, "policy": policy,
                "no_progress_attempts": len(failures), "repeated_input_attempts": repeats,
                "recovery_attempts_used": used, "recovery_attempts_remaining": max(0, policy["max_recovery_attempts"] - used),
                "recovery_signatures": signatures, "signals": signals,
                "requires_recovery": mode in {"reassess", "test_alternative"},
                "last_input_signature": frames[-1]["signature"] if frames else None,
                "representative_observation_ids": list(dict.fromkeys(
                    evidence for signal in signals for evidence in signal["evidence_observation_ids"]))[:3]}

    def reserve(self, turn_id, decision, context):
        if not context["requires_recovery"]:
            return
        proposal = decision.get("recovery_plan")
        if not proposal or proposal["episode_id"] != context["episode_id"] or not proposal["explanation"].strip():
            raise ValueError("Reassessment requires an explicit recovery plan for the current episode")
        if context["recovery_attempts_remaining"] == 0:
            raise ValueError("Review the final recovery outcome before further input")
        if decision.get("additional_actions"):
            raise ValueError("Recovery tests must contain exactly one controller move")
        strategy = decision["strategy"]
        if strategy["kind"] not in {"inspect_menu", "move", "interact", "advance_dialogue", "wait"}:
            raise ValueError("Recovery tests cannot retry battle capture or flee strategies")
        if decision["scene"] in {"battle", "unknown", "transition"} and decision["action"]["kind"] != "wait":
            raise ValueError("Only wait recovery is permitted in battle or an uncertain scene")
        signature = input_signature([decision["action"]])
        serialized = json.dumps(signature)
        forbidden = [*context["recovery_signatures"], context["last_input_signature"]]
        if any(value is not None and serialized == json.dumps(value) for value in forbidden):
            raise ValueError("Recovery must test a different input rather than rename or retime the same action")
        self.connection.execute("INSERT INTO recovery_attempts VALUES(?,?,?,?,?)",
                                (turn_id, context["episode_id"], serialized, json.dumps(strategy), proposal["explanation"]))
        self._event(context["episode_id"], "test_alternative", {"turn_id": turn_id, "strategy": strategy})

    def history(self, limit=20, offset=0):
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("Progress history requires limit 1..100 and nonnegative offset")
        return [dict(row, payload=json.loads(row["payload"])) for row in self.connection.execute(
            "SELECT * FROM progress_events ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset))]
