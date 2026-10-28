from pathlib import Path
import time

from .agent.prompts import messages
from .agent.schemas import MEMORY_CATEGORIES, validate_decision
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
    sequence_images = []
    for _ in range(turns):
        decision, current_image, previous_image, observation_id, timing = _choose_decision(
            config, emulator, agent, memory, run_dir, current_image, previous_image,
            observation_id, timing, recovered, sequence_images)
        try:
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
        actions = [decision["action"], *decision["additional_actions"]]
        sequence_images = []
        try:
            for index, action in enumerate(actions, 1):
                if index > 1:
                    # All buttons are already released. This extra delay is in
                    # addition to the previous move's settle/capture overhead.
                    time.sleep(config.get("sequence_delay_ms", 250) / 1000)
                memory.step_status(turn_id, index, "executing")
                execution = emulator.execute(action)
                result_image, result_id, result_timing = _capture(memory, run_dir, emulator)
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
        except BaseException as exc:
            memory.status(turn_id, "unknown", str(exc) or type(exc).__name__)
            raise
        recovered = []
    return {"completed_actions": completed, "reason": "turn_limit"}


def _capture(memory, run_dir, emulator):
    capture = emulator.capture()
    path, digest = save_capture(run_dir, capture)
    observation_id = memory.observe(path, digest, capture.timing)
    return capture.image, observation_id, capture.timing


def _choose_decision(config, emulator, agent, memory, run_dir, current_image,
                     previous_image, observation_id, timing, recovered, sequence_images=None):
    """Read-only searches do not execute inputs or consume controller turns."""
    budget = config.get("memory_search_rounds", 3)
    search_results = []
    for search_round in range(budget + 1):
        recent = memory.recent_turns(config["recent_turns"])
        location = recent[-1]["decision"].get("location", "") if recent else ""
        facts, events = memory.facts(), memory.events()
        context = {
            "observation_id": observation_id,
            "objective": config["objective"], "rules": config["rules"],
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
            "recent_turns": recent, "recovery": recovered,
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
            decision, trace = agent.decide(messages(context, current_image, previous_image, sequence_images))
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
            current_image, observation_id, timing = _capture(memory, run_dir, emulator)
        except Exception as exc:
            memory.note("decision_error", {"observation_id": observation_id, "error": str(exc)})
            raise
