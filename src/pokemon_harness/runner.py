from pathlib import Path
import time
import sys
import json
from jsonschema import Draft202012Validator

from .agent.lmstudio import InvalidModelResponse, LMStudio
from .agent.context_budget import ContextBudgetError, calibration_key, fit_request, validate_visual_reviews
from .memory.conversation import compaction_settings
from .memory.archive import archive_settings, retrieve_images
from .memory.recall import automatic_recall, recall_settings
from .agent.schemas import MEMORY_CATEGORIES, validate_decision, schema_for_context
from .agent.reassessment import settings as reassessment_settings
from .agent.prompts import messages
from .memory.scenes import goal_token
from .control.loop_detection import loop_warning
from .control.capture import capture_settings, settled_capture, capture_age
from .control.timing import RuntimeBudget, bounded_decision
from .control.session import StopRequested
from .control.handoff import ControlLost
from .control.registry import read_record
from .memory.database import Memory
from .memory.retrieval import ledger_context, recent_turn_context
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
    runtime = None
    try:
        require_live_run(run_dir)
        storage.start()
        memory = Memory(run_dir / "memory.sqlite")
        memory.initialize(config)
        runtime = RuntimeBudget(memory, config)
        runtime.start()
        memory.conversation.active_checkpoint()
        if control:
            control.start()
            control.budget_check = runtime.check
        write_json(run_dir / "config.json", config)
        recovered = memory.recover()
        if recovered:
            memory.note("recovery", recovered)
        memory.monitor_progress()
        review = memory.escalations.review(memory.events())
        if review["blocked_cases"] or review["validity"] == "invalid":
            print("Run needs review before input. Use pokemon-harness cases.")
            return {"reason": "review_required", "case_ids": [case["id"] for case in review["blocked_cases"]],
                    "validity": review["validity"]}
        try:
            return _loop(config, emulator, agent, turns, memory, run_dir, recovered, control, storage, runtime)
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
        if runtime:
            operations.append(("runtime accounting", runtime.finish))
        if memory is not None:
            def checkpoint():
                memory.recover()
                storage.check(128 * 1024)
                memory.save_checkpoint()

            def export():
                value = memory.export()
                storage.check(len(json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")))
                write_json(run_dir / "memory-export.json", value)
            operations.extend([("conversation checkpoint", checkpoint), ("memory export", export), ("database", memory.close)])
        if control:
            operations.append(("session", control.close))
        cleanup_operations(operations, sys.exc_info()[1])


def _loop(config, emulator, agent, turns, memory, run_dir, recovered, control=None, storage=None, runtime=None):
    memory.note("session_start", {"reason": "New control session requires fresh outcome attribution"})
    previous = memory.previous_observation()
    previous_image = (run_dir / previous["image_path"]).read_bytes() if previous else None
    current_image, observation_id, timing = _capture(memory, run_dir, emulator, storage, config, control)
    memory.monitor_progress()
    review = memory.escalations.review(memory.events())
    if review["blocked_cases"]:
        return {"completed_actions": 0, "reason": "review_required",
                "case_ids": [case["id"] for case in review["blocked_cases"]]}
    if control:
        control.set_guard(getattr(emulator, "check_control", None))
        control.check()
    intervention = read_record(run_dir / ".handoff.json")
    completed = 0
    sequence_images = []
    for _ in range(turns):
        if runtime:
            runtime.check()
        if control:
            control.check()
        decision, current_image, previous_image, observation_id, timing, request_id = _choose_decision(
            config, emulator, agent, memory, run_dir, current_image, previous_image,
            observation_id, timing, recovered, sequence_images, control, intervention, storage, runtime)
        if control:
            control.check()
        try:
            if runtime and decision["action"]["kind"] in {"press", "wait"}:
                runtime.fresh(capture_age(memory.capture_metadata(observation_id)), decision["scene"])
            if storage:
                storage.check()
            turn_id = memory.plan(observation_id, decision, request_id)
        except Exception as exc:
            if storage:
                cleanup_operations([("emergency reserve", storage.release_reserve)], exc)
            cleanup_operations([("uncommitted conversation", lambda: memory.fail_request(request_id, "failed", str(exc)))], exc)
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
        interrupted = False
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
                input_age = capture_age(memory.capture_metadata(observation_id))
                decision_age = capture_age(memory.capture_metadata(decision["observation_id"]))
                if runtime:
                    runtime.fresh(decision_age, decision["scene"])
                memory.step_status(turn_id, index, "executing")
                input_started = time.monotonic()
                execution = emulator.execute(action)
                execution = dict(execution or {}, observation_age_at_input_seconds=input_age,
                                 preceding_observation_id=observation_id, decision_observation_id=decision["observation_id"],
                                 decision_observation_age_at_input_seconds=decision_age,
                                 harness_elapsed_seconds=time.monotonic() - input_started)
                result_image, result_id, result_timing = _capture(memory, run_dir, emulator, storage, config, control,
                    decision["scene"], {"effective_action": execution.get("effective_action", action),
                                       "elapsed_seconds": execution["harness_elapsed_seconds"],
                                       "decision_observation_id": decision["observation_id"],
                                       "decision_observation_age_at_input_seconds": decision_age,
                                       "observation_age_at_input_seconds": input_age})
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
                progress = memory.monitor_progress()
                if progress["requires_recovery"] and index < len(actions):
                    memory.status(turn_id, "stopped", "Sequence interrupted for progress reassessment")
                    interrupted = True
                    break
            if not interrupted:
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


def _capture(memory, run_dir, emulator, storage=None, config=None, control=None, scene="unknown", after_input=None):
    if storage:
        storage.check()
    capture, metadata = settled_capture(emulator, capture_settings(config or {}), control, scene,
                                        after_input=after_input is not None)
    metadata["after_input"] = after_input
    if storage:
        storage.check(len(capture.image))
    path, digest = save_capture(run_dir, capture)
    observation_id = memory.observe(path, digest, capture.timing, capture.image, metadata)
    return capture.image, observation_id, capture.timing


def _reassess_scene(config, agent, memory, run_dir, current_image, observation_id, reason,
                    progress, control=None, runtime=None, storage=None):
    state = memory.working.current()
    context = {"phase": "scene_reassessment", "observation_id": observation_id, "reason": reason,
        "objective": config["objective"], "rules": config["rules"], "subgoal": state["subgoal"],
        "working_revision": state["revision"], "goal_token": goal_token(state),
        "scene_fact_candidates": memory.scenes.candidates(memory.facts())}
    model = agent.config if isinstance(agent, LMStudio) else config["model"]
    length = agent.loaded_context_length if isinstance(agent, LMStudio) else None
    context, request, budget = fit_request(context, model, current_image, context_length=length,
        factor=agent.budget_factor if isinstance(agent, LMStudio) else 1.0)
    prefix = run_dir / "turns" / f"reassessment-{observation_id:06d}"
    prefix.parent.mkdir(exist_ok=True)
    write_json(prefix.with_suffix(".context.json"), context)
    print(f"Observation {observation_id}: read-only scene reassessment ({reason})...", flush=True)

    def check():
        if control:
            control.check()
        if runtime:
            runtime.check()

    check()
    try:
        report, trace = bounded_decision(agent, request,
            runtime.policy["generation_seconds"] if runtime else config["model"].get("timeout_seconds", 600), check)
        check()
        if storage:
            storage.check()
        write_json(prefix.with_suffix(".model.json"), trace)
        with memory.connection:
            memory.scenes.save(context, report, memory.working.current(), progress)
        print("Scene reassessment saved; no controller input sent", flush=True)
    except InvalidModelResponse as exc:
        write_json(prefix.with_suffix(".model.json"), exc.trace)
        raise


def _choose_decision(config, emulator, agent, memory, run_dir, current_image,
                     previous_image, observation_id, timing, recovered, sequence_images=None, control=None,
                     intervention=None, storage=None, runtime=None):
    """read only searches before controller input"""
    budget = config.get("memory_search_rounds", 3)
    review_budget = config.get("outcome_review_rounds", 3)
    search_count = 0
    review_count = 0
    review_correction = None
    review_corrections = 0
    search_results = []
    archive_images = []
    archive_policy = archive_settings(config)
    settings = compaction_settings(config)
    compacted = False
    pressure = False
    reassessed = False
    scene_policy = reassessment_settings(config)
    for _ in range(budget + review_budget + 4):
        if runtime:
            runtime.check()
        if storage:
            storage.check()
        if not compacted and memory.conversation.compaction_due(settings, pressure):
            if control:
                control.check()
            if storage:
                storage.check(128 * 1024)
            checkpoint_id = memory.save_checkpoint(compaction=settings)
            compacted = True
            previous_image = current_image
            current_image, observation_id, timing = _capture(memory, run_dir, emulator, storage, config, control)
            sequence_images = []
            print(f"Context compacted at checkpoint {checkpoint_id}; captured a fresh observation", flush=True)
        if scene_policy["enabled"] and not reassessed:
            progress = memory.monitor_progress()
            reason = memory.scenes.reason(observation_id, progress, scene_policy)
            if reason:
                _reassess_scene(config, agent, memory, run_dir, current_image, observation_id, reason,
                                progress, control, runtime, storage)
                reassessed = True
                previous_image = current_image
                current_image, observation_id, timing = _capture(memory, run_dir, emulator, storage, config, control)
                continue
        recent = memory.recent_turns(config["recent_turns"])
        scene_report = memory.scenes.latest()
        if scene_report:
            for turn in recent:
                if turn["id"] <= scene_report["context"]["turn_history_watermark"]:
                    turn["decision"] = dict(turn["decision"], location="", scene="unknown",
                        observation="Historical interpretation predating the latest visual reassessment")
                    if turn.get("outcome_review"):
                        for step in turn["outcome_review"]["steps"]:
                            step.pop("explanation_excerpt", None)
        location = recent[-1]["decision"].get("location", "") if recent and not recent[-1].get("contradiction_keys") else ""
        checkpoint = memory.conversation.active_checkpoint()
        if checkpoint and "compaction" in checkpoint["payload"]:
            boundary_turn = checkpoint["payload"]["source_watermarks"]["turns"]
            recent = [turn for turn in recent[:-2] if turn["id"] > boundary_turn] + recent[-2:]
        facts, events = memory.facts(), memory.events()
        working_state = memory.working_context()
        recalled, recall_report = automatic_recall(facts, events, working_state, recent,
            memory.contradictions.records(), config["objective"], observation_id,
            config["memory_limit"], recall_settings(config))
        context = {
            "observation_id": observation_id,
            "scene_reassessment": memory.scenes.controller_context(observation_id),
            "grounded_scene_action": bool(scene_report and memory.scenes.watermark() == scene_report["turn_watermark"]),
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
            "timing_budgets": dict(runtime.policy, effective_run_seconds=runtime.run_limit) if runtime else None,
            "memory": recalled,
            "automatic_recall": recall_report,
            "ledger": ledger_context(events, location, config["ledger_limit"]),
            "memory_available": {
                "latest_fact_count": len(facts), "event_count": len(events),
                "categories": MEMORY_CATEGORIES, "history_available": True,
                "searches_remaining": budget - search_count, "max_results_per_search": 10,
                "search_scopes": ["all", "facts", "events", "contradictions", "turns"],
                "archive_retrieval": dict(archive_policy, retained_observation_ids=[
                    frame["observation_id"] for frame in archive_images]),
            },
            "memory_search_results": [dict(bundle, results=[
                dict(record, age_observations=observation_id - record["evidence_observation_id"])
                for record in bundle["results"]]) for bundle in search_results],
            "working_state": working_state,
            "conversation": memory.conversation.context(),
            "contradictions": memory.contradictions.context(),
            "progress": memory.monitor_progress(),
            "outcome_review": dict(memory.outcomes.context(), review_batches_remaining=review_budget - review_count),
            "review_correction": review_correction,
            "recent_turns": recent_turn_context(recent), "recovery": recovered,
            "warning": loop_warning(memory.hashes()),
            "instruction": "Interpret the screenshots. Search memory if needed, or update evidenced facts and choose a bounded sequence of moves.",
        }
        if scene_report:
            context["conversation"]["recent"] = [event for event in context["conversation"].get("recent", [])
                if (event.get("observation_id") or 0) > scene_report["observation_id"]]
            context["conversation"].get("continuity", {}).get("reviews", []).clear()
            if context["grounded_scene_action"]:
                context["conversation"]["recent"] = []
        traces = run_dir / "turns"
        traces.mkdir(exist_ok=True)
        prefix = traces / f"observation-{observation_id:06d}"
        write_json(prefix.with_suffix(".context.json"), context)
        model_settings = agent.config if isinstance(agent, LMStudio) else config["model"]
        key = calibration_key(model_settings)
        request_id = None
        try:
            print(f"Observation {observation_id}: requesting model decision "
                  f"(timeout {config['model'].get('timeout_seconds', 180)}s)...", flush=True)
            outcome_images = _outcome_images(memory, run_dir, context["outcome_review"]["next_plan"],
                                             context["progress"]["representative_observation_ids"])
            existing_ids = {frame["observation_id"] for frame in outcome_images}
            outcome_images.extend(frame for frame in archive_images if frame["observation_id"] not in existing_ids)
            context["requested_archive_observation_ids"] = [frame["observation_id"] for frame in archive_images]
            capture_metadata = memory.capture_metadata(observation_id)
            if capture_metadata:
                context["observation_capture"] = {key: capture_metadata[key] for key in (
                    "status", "samples", "elapsed_ms", "captured_at", "deadline_exceeded", "scene_hint")}
                context["observation_capture"]["age_seconds_at_request_build"] = capture_age(capture_metadata)
            factor = memory.budgets.factor(key)
            loaded_context = agent.loaded_context_length if isinstance(agent, LMStudio) else None
            context, request, report = fit_request(context, model_settings, current_image, previous_image,
                                                   sequence_images, outcome_images, loaded_context, factor)
            pressure = any(report["omitted"].get(key) for key in ("recent_turns", "conversation_events"))
            if not compacted and pressure and memory.conversation.compaction_due(settings, pressure=True):
                continue
            write_json(prefix.with_suffix(".context.json"), context)
            write_json(prefix.with_suffix(".budget.json"), report)
            if report["omitted"]:
                print(f"Context budget: omitted optional context {report['omitted']}", flush=True)
            if isinstance(agent, LMStudio):
                agent.budget_factor = factor
            if control:
                control.check()
            request_id = memory.begin_request(request, observation_id, report, model_settings["name"])
            def check_generation():
                if control:
                    control.check()
                if runtime:
                    runtime.check()
            if runtime:
                decision, trace = bounded_decision(agent, request, runtime.policy["generation_seconds"], check_generation)
            else:
                decision, trace = control.decide(agent, request) if control else agent.decide(request)
            if storage:
                storage.check()
            write_json(prefix.with_suffix(".model.json"), trace)
            memory.record_response(request_id, decision, prefix.with_suffix(".model.json").relative_to(run_dir).as_posix())
            memory.budgets.record(key, trace.get("context_budget", {}), observation_id)
            validate_decision(decision, observation_id, config["max_action_frames"],
                              config.get("max_sequence_length", 5))
            if context.get("grounded_scene_action") and not context["outcome_review"]["pending_count"]:
                Draft202012Validator(schema_for_context(context)).validate(decision)
            validate_visual_reviews(decision, context)
            kind = decision["action"]["kind"]
            pending = context["outcome_review"]
            if kind in {"press", "wait"} and pending["pending_count"] and (
                    not decision["outcome_reviews"] or review_correction is not None):
                reason = "Review pending action outcomes before another controller plan"
                if review_corrections or review_count >= review_budget:
                    raise ValueError(reason + "; bounded review correction exhausted")
                memory.fail_request(request_id, "failed", reason)
                review_corrections += 1
                review_correction = {"rejected_request_id": request_id, "pending_turn_id": pending["next_plan"]["turn_id"],
                                     "required_action": "review_outcomes", "reason": reason}
                memory.note("review_correction", dict(review_correction, observation_id=observation_id))
                print("Missing outcome review; discarding proposed input and requesting one read-only correction", flush=True)
                previous_image = current_image
                current_image, observation_id, timing = _capture(memory, run_dir, emulator, storage, config, control, decision["scene"])
                continue
            memory.validate_response(request_id)
            if kind == "review_outcomes":
                if review_count == review_budget:
                    raise ValueError("Outcome review budget exhausted; stop and resume to review remaining plans")
                if control:
                    control.check()
                memory.review_outcomes(observation_id, decision["outcome_reviews"], request_id)
                review_correction = None
                review_count += 1
                print("Outcome review saved; no controller input sent", flush=True)
                previous_image = current_image
                current_image, observation_id, timing = _capture(memory, run_dir, emulator, storage, config, control, decision["scene"])
                continue
            if kind != "search_memory":
                return decision, current_image, previous_image, observation_id, timing, request_id
            if search_count == budget:
                raise ValueError("Memory search budget exhausted; no controller input was sent for this decision")
            search_count += 1
            result = memory.search(decision["memory_search"], observation_id)
            archive_images, image_records = retrieve_images(memory, run_dir,
                decision["memory_search"].get("observation_ids", []), observation_id, archive_policy, archive_images)
            result["archive_images"] = image_records
            memory.record_search(request_id, observation_id, result)
            write_json(prefix.with_suffix(".search.json"), result)
            search_results.append(result)
            print(f"Memory search: {result['total_matches']} matches; returning {len(result['results'])} records", flush=True)
            previous_image = current_image
            current_image, observation_id, timing = _capture(memory, run_dir, emulator, storage, config, control, decision["scene"])
        except Exception as exc:
            if storage:
                cleanup_operations([("emergency reserve", storage.release_reserve)], exc)
            if isinstance(exc, InvalidModelResponse):
                cleanup_operations([("context calibration", lambda: memory.budgets.record(
                    key, exc.trace.get("context_budget", {}), observation_id))], exc)
                cleanup_operations([("failed model trace", lambda: write_json(
                    prefix.with_suffix(".model.json"), exc.trace))], exc)
            if isinstance(exc, ContextBudgetError):
                cleanup_operations([("context budget error", lambda: write_json(
                    prefix.with_suffix(".budget.json"), exc.report))], exc)
            if request_id is not None:
                cleanup_operations([("failed conversation request", lambda: memory.fail_request(
                    request_id, "interrupted" if isinstance(exc, ControlLost) else "failed", str(exc)))], exc)
            cleanup_operations([("decision error", lambda: memory.note(
                "decision_error", {"observation_id": observation_id, "error": str(exc)}))], exc)
            raise


def _outcome_images(memory, run_dir, plan, loop_ids=()):
    ids = {step[key] for step in (plan["steps"] if plan else [])
           if step["input_status"] in {"completed", "executing", "unknown"}
           for key in ("before_observation_id", "after_observation_id") if step[key] is not None}
    frames = []
    for observation_id in sorted(ids | set(loop_ids)):
        row = memory.connection.execute(
            "SELECT image_path FROM observations WHERE id=?", (observation_id,)).fetchone()
        frames.append({"observation_id": observation_id, "image": (run_dir / row[0]).read_bytes(),
                       "purpose": "outcome" if observation_id in ids else "loop"})
    return frames
