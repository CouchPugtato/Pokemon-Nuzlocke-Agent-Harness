"""Cooperative harness stops, including while a model request is pending."""

import os
import queue
import threading
import time
from pathlib import Path

from ..telemetry.recorder import RunLock


class StopRequested(KeyboardInterrupt):
    pass


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

    def start(self):
        # Called only after acquiring the run lock, before accepting new requests.
        self.path.unlink(missing_ok=True)
        self.stopped.clear()
        self.finished.clear()
        self.was_down = self.key_down() if self.key_down else False
        if self.key_down:
            self.watcher = threading.Thread(target=self._watch_hotkey, daemon=True)
            self.watcher.start()

    def _watch_hotkey(self):
        while not self.finished.wait(0.03):
            down = self.key_down()
            if down and not self.was_down:
                self.stopped.set()
            self.was_down = down

    def close(self):
        self.finished.set()
        if self.watcher:
            self.watcher.join(timeout=1)

    def check(self):
        if self.stopped.is_set() or self.path.exists():
            raise StopRequested("User requested a harness stop")

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


def request_stop(run_dir):
    directory = Path(run_dir)
    if not (directory / ".lock").is_file():
        raise ValueError("This run is not running")
    try:
        with RunLock(directory / ".lock"):
            pass
    except RuntimeError:
        (directory / ".stop").touch()
        return
    raise ValueError("This run is not running")
