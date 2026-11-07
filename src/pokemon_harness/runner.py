from pathlib import Path
import time
import sys
import json

from .agent.prompts import messages
from .agent.lmstudio import InvalidModelResponse
from .agent.schemas import MEMORY_CATEGORIES, validate_decision
from .control.loop_detection import loop_warning
from .control.session import StopRequested
from .control.handoff import ControlLost
from .control.registry import read_record
from .memory.database import Memory
from .memory.retrieval import ledger_context, recent_turn_context, retrieve
from .telemetry.recorder import RunLock, save_capture, write_json
from .telemetry.storage import StorageGuard
from .telemetry.cleanup import cleanup_operations
from .telemetry.backup import require_live_run


def run(config, emulator, agent, turns=20, control=None):
    cleanup_started = False
    try:
        if type(turns) is not int or turns < 1:
            raise ValueError("turns must be a positive integer")
        require_live_run(config["run_dir"])
        storage = StorageGuard(config)
        storage.check()
        run_dir = Path(config["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        with RunLock(run_dir / ".lock"):
            cleanup_started = True
            return _run_locked(config, emulator, agent, turns, control, run_dir, storage)
    finally:
        if not cleanup_started:
            cleanup_operations([("emulator", emulator.close)], sys.exc_info()[1])


def _run_locked(config, emulator, agent, turns, control, run_dir, storage):
    memory = None
    try:
        require_live_run(run_dir)
        storage.start()
        memory = Memory(run_dir / "memory.sqlite")
        memory.initialize(config)
        if control:
            control.start()
        write_json(run_dir / "config.json", config)
        recovered = memory.recover()
        if recovered:
            memory.note("recovery", recovered)
        review = memory.escalations.review(memory.events())
        if review["blocked_cases"] or review["validity"] == "invalid":
            print("Run needs review before input. Use pokemon-harness cases.")
            return {"reason": "review_required", "case_ids": [case["id"] for case in review["blocked_cases"]],
                    "validity": review["validity"]}
        try:
            return _loop(config, emulator, agent, turns, memory, run_dir, recovered, control, storage)
        except (ControlLost, StopRequested, KeyboardInterrupt) as exc:
            if control is None:
                raise
            storage.release_reserve()
            reason = "control_lost" if isinstance(exc, ControlLost) else getattr(exc, "reason", "user_stop")
            if reason in {"control_lost", "manual_handoff"}:
                handoff = {"reason": reason, "message": str(exc),
                           "session_id": control.record["session_id"],
                           "last_observation_id": (memory.previous_observation() or {}).get("id")}
                memory.note("control_handoff", handoff)
                write_json(run_dir / ".handoff.json", handoff)
            else:
                memory.note("user_stop", {"message": "Harness stopped; emulator continues running"})
            print("Stopping harness; saving memory and releasing inputs. Emulator keeps running.")
            return {"reason": reason}
    finally:
        operations = [("emergency reserve", storage.release_reserve)]
        if control:
            operations.append(("control monitor", control.finish_monitoring))
        operations.append(("emulator", emulator.close))
        if memory is not None:
            def export():
                value = memory.export()
                storage.check(len(json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")))
                write_json(run_dir / "memory-export.json", value)
            operations.extend([("memory export", export), ("database", memory.close)])
        if control:
            operations.append(("session", control.close))
        cleanup_operations(operations, sys.exc_info()[1])


def _loop(config, emulator, agent, turns, memory, run_dir, recovered, control=None, storage=None):
    previous = memory.previous_observation()
    previous_image = (run_dir / previous["image_path"]).read_bytes() if previous else None
    current_image, observation_id, timing = _capture(memory, run_dir, emulator, storage)
    if control:
        control.set_guard(getattr(emulator, "check_control", None))
        control.check()
    intervention = read_record(run_dir / ".handoff.json")
    completed = 0
    sequence_images = []
    for _ in range(turns):
        if control:
            control.check()
        decision, current_image, previous_image, observation_id, timing = _choose_decision(
            config, emulator, agent, memory, run_dir, current_image, previous_image,
            observation_id, timing, recovered, sequence_images, control, intervention, storage)
        if control:
            control.check()
        try:
            if storage:
                storage.check()
            turn_id = memory.plan(observation_id, decision)
        except Exception as exc:
            if storage:
                cleanup_operations([("emergency reserve", storage.release_reserve)], exc)
            cleanup_operations([("decision error", lambda: memory.note(
                "decision_error", {"observation_id": observation_id, "error": str(exc)}))], exc)
            raise

        review = memory.escalations.review(memory.events())
        if review["blocked_cases"] or review["validity"] == "invalid":
            case_ids = [case["id"] for case in review["blocked_cases"]]
            memory.status(turn_id, "blocked", "Unresolved review cases: " + str(case_ids))
            print(f"Stopped before input: review cases {case_ids}. Use pokemon-harness cases.")
            return {"completed_actions": completed, "reason": "unresolved_events", "case_ids": case_ids,
                    "validity": review["validity"]}
        if decision["action"]["kind"] == "stop":
            memory.status(turn_id, "stopped")
            print(f"Agent stopped: {decision['expected_result']}")
            return {"completed_actions": completed, "reason": "agent_stop"}

        memory.status(turn_id, "executing")
        actions = [decision["action"], *decision["additional_actions"]]
        sequence_images = []
        try:
            for index, action in enumerate(actions, 1):
                if control:
                    control.check()
                if index > 1:
                    # extra delay after the previous move and capture
                    delay = config.get("sequence_delay_ms", 250) / 1000
                    if control:
                        control.wait(delay)
                    else:
                        time.sleep(delay)
                if storage:
                    storage.check()
                if control:
                    control.check()
                memory.step_status(turn_id, index, "executing")
                execution = emulator.execute(action)
                result_image, result_id, result_timing = _capture(memory, run_dir, emulator, storage)
                memory.note("action_result", {"turn_id": turn_id, "observation_id": result_id,
                                              "execution": execution, "step_index": index,
                                              "sequence_length": len(actions)})
                memory.step_status(turn_id, index, "completed", result_id, execution)
                sequence_images.append({"step_index": index, "observation_id": result_id,
                                        "image": result_image})
                completed += 1
                effective = execution.get("effective_action", action) if execution else action
                print(f"Turn {turn_id}, move {index}/{len(actions)}: input sent / {effective} "
                      f"-> observation {result_id} (game response not yet verified)", flush=True)
                previous_image, current_image, observation_id = current_image, result_image, result_id
                timing = result_timing
            memory.status(turn_id, "completed")
            if intervention:
                memory.note("resume_after_handoff", {"handoff": intervention,
                                                     "fresh_observation_id": decision["observation_id"]})
                (run_dir / ".handoff.json").unlink(missing_ok=True)
                intervention = None
        except StopRequested as exc:
            if storage:
                cleanup_operations([("emergency reserve", storage.release_reserve)], exc)
            cleanup_operations([("stopped plan", lambda: memory.status(
                turn_id, "stopped", "User stopped before the next move"))], exc)
            raise
        except BaseException as exc:
            if storage:
                cleanup_operations([("emergency reserve", storage.release_reserve)], exc)
            cleanup_operations([("uncertain plan", lambda: memory.status(
                turn_id, "unknown", str(exc) or type(exc).__name__))], exc)
            raise
        recovered = []
    return {"completed_actions": completed, "reason": "turn_limit"}


def _capture(memory, run_dir, emulator, storage=None):
    if storage:
        storage.check()
    capture = emulator.capture()
    if storage:
        storage.check(len(capture.image))
    path, digest = save_capture(run_dir, capture)
    observation_id = memory.observe(path, digest, capture.timing)
    return capture.image, observation_id, capture.timing


def _choose_decision(config, emulator, agent, memory, run_dir, current_image,
                     previous_image, observation_id, timing, recovered, sequence_images=None, control=None,
                     intervention=None, storage=None):
    """read only searches before controller input"""
    budget = config.get("memory_search_rounds", 3)
    search_results = []
    for search_round in range(budget + 1):
        if storage:
            storage.check()
        recent = memory.recent_turns(config["recent_turns"])
        location = recent[-1]["decision"].get("location", "") if recent else ""
        facts, events = memory.facts(), memory.events()
        context = {
            "observation_id": observation_id,
            "objective": config["objective"], "rules": config["rules"],
            "intervention": intervention,
            "run_review": {"validity": memory.escalations.validity(), "authorized_overrides": [
                {"case_id": case["id"], "details": case["details"],
                 "resolution": case["resolutions"][-1]} for case in memory.escalations.cases()
                if not case["blocking"] and case["resolutions"][-1]["allow_continue"]]},
            "action_limit_frames": config["max_action_frames"],
            "max_sequence_length": config.get("max_sequence_length", 5),
            "sequence_delay_ms": config.get("sequence_delay_ms", 250),
            "sequence_observations": [{key: value for key, value in frame.items() if key != "image"}
                                      for frame in (sequence_images or [])],
            "minimum_press_frames": config["emulator"].get("min_hold_frames", 6)
                if config["emulator"]["backend"] in {"mgba", "vbam"} else 1,
            "timing": timing,
            "memory": retrieve(facts, config["objective"] + " " + location,
                               observation_id, config["memory_limit"]),
            "ledger": ledger_context(events, location, config["ledger_limit"]),
            "memory_available": {
                "latest_fact_count": len(facts), "event_count": len(events),
                "categories": MEMORY_CATEGORIES, "history_available": True,
                "searches_remaining": budget - search_round, "max_results_per_search": 10,
            },
            "memory_search_results": [dict(bundle, results=[
                dict(record, age_observations=observation_id - record["evidence_observation_id"])
                for record in bundle["results"]]) for bundle in search_results],
            "working_state": memory.working.current(),
            "recent_turns": recent_turn_context(recent), "recovery": recovered,
            "warning": loop_warning(memory.hashes()),
            "instruction": "Interpret the screenshots. Search memory if needed, or update evidenced facts and choose a bounded sequence of moves.",
        }
        traces = run_dir / "turns"
        traces.mkdir(exist_ok=True)
        prefix = traces / f"observation-{observation_id:06d}"
        write_json(prefix.with_suffix(".context.json"), context)
        try:
            print(f"Observation {observation_id}: requesting model decision "
                  f"(timeout {config['model'].get('timeout_seconds', 180)}s)...", flush=True)
            request = messages(context, current_image, previous_image, sequence_images)
            decision, trace = control.decide(agent, request) if control else agent.decide(request)
            if storage:
                storage.check()
            write_json(prefix.with_suffix(".model.json"), trace)
            validate_decision(decision, observation_id, config["max_action_frames"],
                              config.get("max_sequence_length", 5))
            if decision["action"]["kind"] != "search_memory":
                return decision, current_image, previous_image, observation_id, timing
            if search_round == budget:
                raise ValueError("Memory search budget exhausted; no controller input was sent for this decision")
            result = memory.search(decision["memory_search"], observation_id)
            memory.note("memory_search", result)
            write_json(prefix.with_suffix(".search.json"), result)
            search_results.append(result)
            print(f"Memory search: {result['total_matches']} matches; returning {len(result['results'])} records", flush=True)
            previous_image = current_image
            current_image, observation_id, timing = _capture(memory, run_dir, emulator, storage)
        except Exception as exc:
            if storage:
                cleanup_operations([("emergency reserve", storage.release_reserve)], exc)
            if isinstance(exc, InvalidModelResponse):
                cleanup_operations([("failed model trace", lambda: write_json(
                    prefix.with_suffix(".model.json"), exc.trace))], exc)
            cleanup_operations([("decision error", lambda: memory.note(
                "decision_error", {"observation_id": observation_id, "error": str(exc)}))], exc)
            raise
