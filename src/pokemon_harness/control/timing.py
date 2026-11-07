"""session generation freshness and durable active runtime limits"""

import json
import queue
import threading
import time
from datetime import datetime, timezone


class TimingLimit(RuntimeError):
    pass


def timing_settings(config):
    result = {"session_seconds": 3600, "run_seconds": 86400, "generation_seconds": 600,
              "decision_age_seconds": 600, "scene_age_seconds": {}}
    supplied = config.get("timing_budgets", {})
    if not isinstance(supplied, dict) or set(supplied) - set(result):
        raise ValueError("Invalid timing_budgets settings")
    result.update(supplied)
    for key in ("session_seconds", "run_seconds", "generation_seconds", "decision_age_seconds"):
        if type(result[key]) is not int or not 1 <= result[key] <= 604800:
            raise ValueError(f"timing_budgets.{key} must be an integer between one and six hundred four thousand eight hundred")
    scenes = result["scene_age_seconds"]
    if not isinstance(scenes, dict) or set(scenes) - {"overworld", "menu", "dialogue", "battle", "transition", "unknown"} or any(
            type(value) is not int or not 1 <= value <= result["decision_age_seconds"] for value in scenes.values()):
        raise ValueError("Invalid timing_budgets.scene_age_seconds")
    return result


def bounded_decision(agent, request, seconds, check=lambda: None):
    result = queue.Queue(maxsize=1)
    deadline = time.monotonic() + seconds

    def worker():
        try:
            result.put((True, agent.decide(request)))
        except BaseException as exc:
            result.put((False, exc))

    def guard():
        check()
        if time.monotonic() >= deadline:
            raise TimingLimit("Generation time budget exhausted; response abandoned before further input")

    guard()
    threading.Thread(target=worker, daemon=True).start()
    while True:
        guard()
        try:
            success, value = result.get(timeout=min(0.1, max(0.001, deadline - time.monotonic())))
        except queue.Empty:
            continue
        guard()
        if success:
            return value
        raise value


class RuntimeBudget:
    def __init__(self, memory, config, clock=time.monotonic):
        self.connection = memory.connection
        self.policy = timing_settings(config)
        self.clock = clock
        self.session_id = None
        self.connection.execute("""CREATE TABLE IF NOT EXISTS runtime_sessions(
            id INTEGER PRIMARY KEY, started_at TEXT NOT NULL, elapsed_seconds REAL NOT NULL, closed INTEGER NOT NULL)""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS runtime_extensions(
            id INTEGER PRIMARY KEY, seconds INTEGER NOT NULL, reason TEXT NOT NULL, created_at TEXT NOT NULL)""")

    def extend(self, seconds, reason):
        if type(seconds) is not int or not 1 <= seconds <= 604800 or not isinstance(reason, str) or not reason.strip() or len(reason) > 600:
            raise ValueError("Runtime extension needs one to six hundred four thousand eight hundred seconds and a bounded reason")
        with self.connection:
            self.connection.execute("INSERT INTO runtime_extensions VALUES(NULL,?,?,?)",
                                    (seconds, reason, datetime.now(timezone.utc).isoformat()))

    def start(self):
        now = datetime.now(timezone.utc)
        with self.connection:
            row = self.connection.execute("SELECT value FROM metadata WHERE key='timing_policy'").fetchone()
            if row and json.loads(row[0]) != self.policy:
                raise ValueError("Timing policy differs from saved run; use its saved settings or a new run")
            self.connection.execute("INSERT OR IGNORE INTO metadata VALUES('timing_policy',?)", (json.dumps(self.policy),))
            for row in self.connection.execute("SELECT id,started_at FROM runtime_sessions WHERE closed=0").fetchall():
                elapsed = min(self.policy["session_seconds"], max(0, (now - datetime.fromisoformat(row[1])).total_seconds()))
                self.connection.execute("UPDATE runtime_sessions SET closed=1,elapsed_seconds=? WHERE id=?", (elapsed, row[0]))
            self.used = self.connection.execute("SELECT COALESCE(SUM(elapsed_seconds),0) FROM runtime_sessions").fetchone()[0]
            self.run_limit = self.policy["run_seconds"] + self.connection.execute("SELECT COALESCE(SUM(seconds),0) FROM runtime_extensions").fetchone()[0]
        if self.used >= self.run_limit:
            raise TimingLimit("Cumulative active run time budget exhausted")
        self.started = self.clock()
        with self.connection:
            self.session_id = self.connection.execute("INSERT INTO runtime_sessions VALUES(NULL,?,0,0)", (now.isoformat(),)).lastrowid

    def check(self):
        elapsed = self.clock() - self.started
        if elapsed >= self.policy["session_seconds"] or elapsed + self.used >= self.run_limit:
            raise TimingLimit("Session or cumulative active run time budget exhausted; no further input permitted")

    def fresh(self, age, scene):
        self.check()
        maximum = self.policy["scene_age_seconds"].get(scene, self.policy["decision_age_seconds"])
        if age is None or age > maximum:
            raise TimingLimit("Decision screenshot is too old for controller input; stop and resume with fresh evidence")

    def finish(self):
        if self.session_id is not None:
            with self.connection:
                self.connection.execute("UPDATE runtime_sessions SET elapsed_seconds=?,closed=1 WHERE id=?",
                                        (max(0, self.clock() - self.started), self.session_id))
            self.session_id = None
