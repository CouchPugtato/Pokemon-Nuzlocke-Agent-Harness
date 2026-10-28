from pathlib import Path

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
    for _ in range(turns):
        decision, current_image, previous_image, observation_id, timing = _choose_decision(
            config, emulator, agent, memory, run_dir, current_image, previous_image,
            observation_id, timing, recovered)
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
        try:
            execution = emulator.execute(decision["action"])
            # Capture after EVERY action, including the last allowed action.
            result_image, result_id, result_timing = _capture(memory, run_dir, emulator)
            memory.note("action_result", {"turn_id": turn_id, "observation_id": result_id,
                                          "execution": execution})
            memory.status(turn_id, "completed")
        except BaseException as exc:
            memory.status(turn_id, "unknown", str(exc) or type(exc).__name__)
            raise
        completed += 1
        effective = execution.get("effective_action", decision["action"]) if execution else decision["action"]
        print(f"Turn {turn_id}: input sent / {decision['scene']} / {effective} -> observation {result_id} "
              "(game response not yet verified)", flush=True)
        previous_image, current_image, observation_id = current_image, result_image, result_id
        timing = result_timing
        recovered = []
    return {"completed_actions": completed, "reason": "turn_limit"}


def _capture(memory, run_dir, emulator):
    capture = emulator.capture()
    path, digest = save_capture(run_dir, capture)
    observation_id = memory.observe(path, digest, capture.timing)
    return capture.image, observation_id, capture.timing


def _choose_decision(config, emulator, agent, memory, run_dir, current_image,
                     previous_image, observation_id, timing, recovered):
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
            "instruction": "Interpret the current screenshot. Search memory if needed, or update evidenced facts and choose one short action.",
        }
        traces = run_dir / "turns"
        traces.mkdir(exist_ok=True)
        prefix = traces / f"observation-{observation_id:06d}"
        write_json(prefix.with_suffix(".context.json"), context)
        try:
            print(f"Observation {observation_id}: requesting model decision "
                  f"(timeout {config['model'].get('timeout_seconds', 180)}s)...", flush=True)
            decision, trace = agent.decide(messages(context, current_image, previous_image))
            write_json(prefix.with_suffix(".model.json"), trace)
            validate_decision(decision, observation_id, config["max_action_frames"])
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
