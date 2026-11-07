"""bounded controller checks without model inference"""

import uuid
import sys
from pathlib import Path

from ..agent.schemas import BUTTONS
from ..telemetry.recorder import RunLock, save_capture, write_json
from ..telemetry.storage import StorageGuard
from ..telemetry.cleanup import cleanup_operations


def input_test(config, emulator, button, hold_frames=12, settle_frames=45):
    storage = None
    try:
        storage = StorageGuard(config)
        storage.check()
        return _input_test(config, emulator, button, hold_frames, settle_frames, storage)
    finally:
        operations = []
        if storage:
            operations.append(("emergency reserve", storage.release_reserve))
        operations.append(("emulator", emulator.close))
        cleanup_operations(operations, sys.exc_info()[1])


def _input_test(config, emulator, button, hold_frames, settle_frames, storage):
    maximum = config["max_action_frames"]
    if button not in BUTTONS:
        raise ValueError("Unknown GBA button")
    if type(hold_frames) is not int or not 1 <= hold_frames <= maximum:
        raise ValueError(f"hold_frames must be between 1 and {maximum}")
    if type(settle_frames) is not int or not 0 <= settle_frames <= maximum:
        raise ValueError(f"settle_frames must be between 0 and {maximum}")
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    with RunLock(run_dir / ".lock"):
        directory = run_dir / "input-tests" / uuid.uuid4().hex
        directory.mkdir(parents=True)
        action = {"kind": "press", "buttons": [button], "hold_frames": hold_frames,
                  "settle_frames": settle_frames}
        report = {"action": action, "status": "not_sent"}
        try:
            storage.start()
            before, before_hash = save_capture(directory, emulator.capture())
            storage.check()
            report.update(before=str(before), before_hash=before_hash, status="executing")
            write_json(directory / "report.json", report)
            report["execution"] = emulator.execute(action)
            storage.check()
            after, after_hash = save_capture(directory, emulator.capture())
            report.update(after=str(after), after_hash=after_hash, status="submitted",
                          screenshots_identical=before_hash == after_hash,
                          interpretation="Inspect both screenshots: animation changes alone do not prove input registered.")
        except BaseException as exc:
            cleanup_operations([("emergency reserve", storage.release_reserve)], exc)
            report.update(status="unknown" if report["status"] == "executing" else "not_sent",
                          error=str(exc) or type(exc).__name__)
            raise
        finally:
            cleanup_operations([("emergency reserve", storage.release_reserve), ("emulator", emulator.close),
                                ("diagnostic report", lambda: write_json(directory / "report.json", report))],
                               sys.exc_info()[1])
    print(f"Input test saved to {directory}. Inspect before/after images to verify the game response.")
    return report
