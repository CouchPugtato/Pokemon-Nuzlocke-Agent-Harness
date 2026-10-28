"""One bounded controller test, independently of inference and agent memory."""

import uuid
from pathlib import Path

from ..agent.schemas import BUTTONS
from ..telemetry.recorder import RunLock, save_capture, write_json


def input_test(config, emulator, button, hold_frames=12, settle_frames=45):
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
            before, before_hash = save_capture(directory, emulator.capture())
            report.update(before=str(before), before_hash=before_hash, status="executing")
            write_json(directory / "report.json", report)
            report["execution"] = emulator.execute(action)
            after, after_hash = save_capture(directory, emulator.capture())
            report.update(after=str(after), after_hash=after_hash, status="submitted",
                          screenshots_identical=before_hash == after_hash,
                          interpretation="Inspect both screenshots: animation changes alone do not prove input registered.")
        except BaseException as exc:
            report.update(status="unknown" if report["status"] == "executing" else "not_sent",
                          error=str(exc) or type(exc).__name__)
            raise
        finally:
            try:
                emulator.close()
            finally:
                write_json(directory / "report.json", report)
    print(f"Input test saved to {directory}. Inspect before/after images to verify the game response.")
    return report
