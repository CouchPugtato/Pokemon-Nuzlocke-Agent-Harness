from pathlib import Path

from .agent.prompts import messages
from .agent.schemas import validate_decision
from .control.loop_detection import loop_warning
from .memory.database import Memory
from .memory.retrieval import ledger_context, retrieve
from .rules.nuzlocke import unresolved_events
from .telemetry.recorder import RunLock, save_capture, write_json


def run(config, emulator, agent, turns=20):
    if type(turns) is not int or turns < 1:
        raise ValueError("turns must be a positive integer")
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    with RunLock(run_dir / ".lock"):
        memory = Memory(run_dir / "memory.sqlite")
        try:
            memory.initialize(config)
            write_json(run_dir / "config.json", config)
            recovered = memory.recover()
            if recovered:
                memory.note("recovery", recovered)
            return _loop(config, emulator, agent, turns, memory, run_dir, recovered)
        finally:
            try:
                emulator.close()
            finally:
                write_json(run_dir / "memory-export.json", memory.export())
                memory.close()


def _loop(config, emulator, agent, turns, memory, run_dir, recovered):
    previous = memory.previous_observation()
    previous_image = (run_dir / previous["image_path"]).read_bytes() if previous else None
    current_image, observation_id, timing = _capture(memory, run_dir, emulator)
    completed = 0
    for _ in range(turns):
        recent = memory.recent_turns(config["recent_turns"])
        last = recent[-1]["decision"] if recent else {}
        location = last.get("location", "")
        context = {
            "observation_id": observation_id,
            "objective": config["objective"], "rules": config["rules"],
            "action_limit_frames": config["max_action_frames"],
            "timing": timing,
            "memory": retrieve(memory.facts(), config["objective"] + " " + location,
                               observation_id, config["memory_limit"]),
            "ledger": ledger_context(memory.events(), location, config["ledger_limit"]),
            "recent_turns": recent,
            "recovery": recovered,
            "warning": loop_warning(memory.hashes()),
            "instruction": "Interpret the current screenshot, update evidenced facts, then choose one short action.",
        }
        traces = run_dir / "turns"
        traces.mkdir(exist_ok=True)
        prefix = traces / f"observation-{observation_id:06d}"
        write_json(prefix.with_suffix(".context.json"), context)
        try:
            decision, trace = agent.decide(messages(context, current_image, previous_image))
            write_json(prefix.with_suffix(".model.json"), trace)
            validate_decision(decision, observation_id, config["max_action_frames"])
            turn_id = memory.plan(observation_id, decision)
        except Exception as exc:
            memory.note("decision_error", {"observation_id": observation_id, "error": str(exc)})
            raise

        uncertain = unresolved_events(memory.events())
        if uncertain:
            memory.status(turn_id, "blocked", "Unresolved Nuzlocke events: " + ", ".join(uncertain))
            print(f"Stopped before input: unresolved events {', '.join(uncertain)}")
            return {"completed_actions": completed, "reason": "unresolved_events"}
        if decision["action"]["kind"] == "stop":
            memory.status(turn_id, "stopped")
            print(f"Agent stopped: {decision['expected_result']}")
            return {"completed_actions": completed, "reason": "agent_stop"}

        memory.status(turn_id, "executing")
        try:
            emulator.execute(decision["action"])
            # Capture after EVERY action, including the last allowed action.
            result_image, result_id, result_timing = _capture(memory, run_dir, emulator)
            memory.note("action_result", {"turn_id": turn_id, "observation_id": result_id})
            memory.status(turn_id, "completed")
        except BaseException as exc:
            memory.status(turn_id, "unknown", str(exc) or type(exc).__name__)
            raise
        completed += 1
        print(f"Turn {turn_id}: {decision['scene']} / {decision['action']} -> observation {result_id}")
        previous_image, current_image, observation_id = current_image, result_image, result_id
        timing = result_timing
        recovered = []
    return {"completed_actions": completed, "reason": "turn_limit"}


def _capture(memory, run_dir, emulator):
    capture = emulator.capture()
    path, digest = save_capture(run_dir, capture)
    observation_id = memory.observe(path, digest, capture.timing)
    return capture.image, observation_id, capture.timing
