"""Cooperative harness stops, including while a model request is pending."""

import os
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..telemetry.recorder import RunLock
from ..telemetry.recorder import write_json
from .registry import read_record, register_session, unregister_session
from .handoff import ControlLost


class StopRequested(KeyboardInterrupt):
    reason = "user_stop"


class HandoffRequested(StopRequested):
    reason = "manual_handoff"


class RunControl:
    def __init__(self, run_dir, hotkey=True):
        self.path = Path(run_dir) / ".stop"
        self.key_down = None
        if hotkey and os.name == "nt":
            import ctypes
            self.key_down = lambda: bool(ctypes.windll.user32.GetAsyncKeyState(0x77) & 0x8000)
        self.was_down = False
        self.stopped = threading.Event()
        self.finished = threading.Event()
        self.watcher = None
        self.record = None
        self.guard = None
        self.guard_error = None

    def start(self):
        # Called only after acquiring the run lock, before accepting new requests.
        self.path.unlink(missing_ok=True)
        self.stopped.clear()
        self.finished.clear()
        self.guard_error = None
        self.was_down = self.key_down() if self.key_down else False
        self.record = {"session_id": uuid.uuid4().hex, "run_dir": str(self.path.parent.resolve()),
                       "pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()}
        register_session(self.record)
        print(f"Session: {self.record['session_id']}", flush=True)
        self.watcher = threading.Thread(target=self._watch_hotkey, daemon=True)
        self.watcher.start()

    def _watch_hotkey(self):
        while not self.finished.wait(0.03):
            down = self.key_down() if self.key_down else False
            if down and not self.was_down:
                self.stopped.set()
            self.was_down = down
            self._check_guard()

    def set_guard(self, guard):
        self.guard = guard

    def _check_guard(self):
        guard = self.guard
        if guard and self.guard_error is None:
            try:
                guard()
            except Exception as exc:
                self.guard_error = str(exc) or type(exc).__name__

    def finish_monitoring(self):
        self.finished.set()
        if self.watcher:
            self.watcher.join(timeout=1)

    def close(self):
        self.finish_monitoring()
        if self.record:
            unregister_session(self.record)

    def check(self):
        request = read_record(self.path)
        matching_request = self.record and request and request.get("session_id") == self.record["session_id"]
        if matching_request and request.get("kind") == "handoff":
            raise HandoffRequested("User requested manual control")
        if self.stopped.is_set() or matching_request:
            raise StopRequested("User requested a harness stop")
        self._check_guard()
        if self.guard_error is not None:
            raise ControlLost(self.guard_error)

    def wait(self, seconds):
        deadline = time.monotonic() + seconds
        while True:
            self.check()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.1, remaining))

    def decide(self, agent, messages):
        self.check()
        result = queue.Queue(maxsize=1)

        def request():
            try:
                result.put((True, agent.decide(messages)))
            except BaseException as exc:
                result.put((False, exc))

        # This thread handles HTTP only, never game input or memory. If stopped,
        # discard its eventual response; do not wait for the server to finish.
        threading.Thread(target=request, daemon=True).start()
        while True:
            self.check()
            try:
                success, value = result.get(timeout=0.1)
            except queue.Empty:
                continue
            self.check()
            if success:
                return value
            raise value


def request_stop(run_dir, session_id=None, kind="stop"):
    if kind not in {"stop", "handoff"}:
        raise ValueError("Unknown control request kind")
    directory = Path(run_dir)
    if not (directory / ".lock").is_file():
        raise ValueError("This run is not running")
    try:
        with RunLock(directory / ".lock"):
            pass
    except RuntimeError:
        record = read_record(directory / ".session.json")
        if not record or (session_id is not None and record.get("session_id") != session_id):
            raise ValueError("The requested session is no longer active")
        write_json(directory / ".stop", {"session_id": record["session_id"], "kind": kind})
        return record["session_id"]
    raise ValueError("This run is not running")
