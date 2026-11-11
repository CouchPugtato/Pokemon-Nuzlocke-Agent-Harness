"""archived request replay and shadow inference without controller access"""

import copy
import hashlib
import json
import sqlite3
import shutil
import tempfile
import time
from contextlib import closing
from pathlib import Path

from .agent.prompts import image
from .agent.lmstudio import LMStudio
from .agent.schemas import validate_decision
from .agent.context_budget import validate_visual_reviews
from .control.timing import bounded_decision, TimingLimit
from .control.loop_detection import input_signature
from .memory.archive import retrieve_images
from .memory.database import Memory
from .telemetry.recorder import RunLock, write_json


def rehydrate(memory, source, archived):
    request = copy.deepcopy(archived["messages"])
    ids = list(dict.fromkeys(block["observation_id"] for message in request if isinstance(message.get("content"), list)
                            for block in message["content"] if block["type"] == "observation_image"))
    frames, records = retrieve_images(memory, source, ids, archived["observation_id"],
        {"max_images": 32, "max_bytes": 67108864, "max_image_pixels": 16777216})
    if any(record["read_status"] != "verified" for record in records):
        raise ValueError("Evaluation requires every original request image")
    frames = {frame["observation_id"]: frame["image"] for frame in frames}
    for message in request:
        if "content_reference" in message:
            message["content"] = archived["prompts"][message.pop("content_reference")]
        elif isinstance(message.get("content"), list):
            for index, block in enumerate(message["content"]):
                if block["type"] == "observation_image":
                    png = frames[block["observation_id"]]
                    if hashlib.sha256(png).hexdigest() != block["image_hash"]:
                        raise ValueError("Evaluation image reference fingerprint mismatch")
                    message["content"][index] = image(png)
    fingerprint = hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if fingerprint != archived["request_hash"]:
        raise ValueError("Rehydrated evaluation request differs from the original request")
    return request


def load_labels(path):
    if path is None:
        return {}
    path = Path(path)
    if path.stat().st_size > 1048576:
        raise ValueError("Evaluation label file is too large")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"cases"} or not isinstance(value["cases"], list) or len(value["cases"]) > 1000:
        raise ValueError("Evaluation labels need a bounded cases array")
    result = {}
    for case in value["cases"]:
        if not isinstance(case, dict) or set(case) - {"request_id", "allowed_action_kinds", "forbidden_claim_fragments", "note"}:
            raise ValueError("Invalid evaluation label fields")
        identity = case.get("request_id")
        if type(identity) is not int or identity < 1 or identity in result:
            raise ValueError("Evaluation label request ids must be positive and unique")
        kinds = case.get("allowed_action_kinds", [])
        fragments = case.get("forbidden_claim_fragments", [])
        if not isinstance(kinds, list) or len(kinds) > 5 or any(kind not in {"press", "wait", "stop", "search_memory", "review_outcomes"} for kind in kinds):
            raise ValueError("Invalid allowed evaluation action kinds")
        if not isinstance(fragments, list) or len(fragments) > 20 or any(
                not isinstance(fragment, str) or not fragment.strip() or len(fragment) > 200 for fragment in fragments):
            raise ValueError("Invalid forbidden evaluation claim fragments")
        if not kinds and not fragments:
            raise ValueError("Evaluation labels require at least one action or claim criterion")
        if not isinstance(case.get("note", ""), str) or len(case.get("note", "")) > 1000:
            raise ValueError("Invalid evaluation label note")
        result[identity] = case
    return result


def input_plan(decision):
    return [decision["action"], *decision["additional_actions"]]


def validate_context_gates(decision, context):
    controller = decision["action"]["kind"] in {"press", "wait"}
    if controller and context.get("review_correction"):
        raise ValueError("Proposal bypasses the archived read only review correction")
    pending = context.get("outcome_review", {})
    plan = pending.get("next_plan")
    reviews = decision["outcome_reviews"]
    if controller and pending.get("pending_count", 0) and (
            pending["pending_count"] > 1 or not reviews or not plan or reviews[0]["turn_id"] != plan["turn_id"]):
        raise ValueError("Proposal bypasses the archived pending outcome review gate")
    if plan:
        for review in reviews:
            if review["turn_id"] != plan["turn_id"] or review["expected_revision"] != plan["expected_revision"]:
                raise ValueError("Proposal reviews the wrong archived plan revision")
            attempted = {step["step_index"]: step for step in plan["steps"] if step["input_status"] in {"completed", "executing", "unknown"}}
            if len(review["steps"]) != len(attempted) or {step["step_index"] for step in review["steps"]} != set(attempted):
                raise ValueError("Proposal must review each archived attempted step exactly once")
            for step in review["steps"]:
                source = attempted[step["step_index"]]
                after = source["after_observation_id"] or context["observation_id"]
                allowed = source.get("review_after_observation_ids", [after])
                if step["before_observation_id"] != source["before_observation_id"] or step["after_observation_id"] not in allowed:
                    raise ValueError("Proposal review uses the wrong archived observation evidence")
                if source["after_observation_id"] is None and step["assessment"] != "uncertain":
                    raise ValueError("Proposal claims a result for an uncaptured input")
    progress = context.get("progress", {})
    if controller and progress.get("requires_recovery"):
        recovery = decision["recovery_plan"]
        if not recovery or recovery["episode_id"] != progress["episode_id"] or not recovery["explanation"].strip() \
                or decision["additional_actions"] or not progress["recovery_attempts_remaining"]:
            raise ValueError("Proposal bypasses the archived recovery budget")
        if decision["strategy"]["kind"] not in {"inspect_menu", "move", "interact", "advance_dialogue", "wait"} \
                or decision["scene"] in {"battle", "unknown", "transition"} and decision["action"]["kind"] != "wait":
            raise ValueError("Proposal uses an unsafe recovery strategy")
        signature = json.dumps(input_signature([decision["action"]]))
        if any(value is not None and signature == json.dumps(value) for value in
               [*progress["recovery_signatures"], progress["last_input_signature"]]):
            raise ValueError("Proposal repeats a forbidden archived recovery input")


def label_result(decision, label):
    if label is None:
        return None
    claims = " ".join(item["claim"] for item in decision["memory_updates"]) + " " + " ".join(
        item["details"] for item in decision["events"])
    hits = [fragment for fragment in label.get("forbidden_claim_fragments", []) if fragment.casefold() in claims.casefold()]
    kinds = label.get("allowed_action_kinds", [])
    return {"matched": not hits and (not kinds or decision["action"]["kind"] in kinds),
            "forbidden_fragment_hits": hits, "criteria": label,
            "method": "human supplied action and literal claim criteria without independent semantic verification"}


def evaluate(source, output, mode="replay", agent=None, limit=20, offset=0, repeat=1, labels=None,
             generation_seconds=600, total_seconds=3600):
    source, output = Path(source).resolve(), Path(output).resolve()
    if mode not in {"replay", "shadow"} or (mode == "shadow" and agent is None):
        raise ValueError("Shadow evaluation requires a model")
    if type(limit) is not int or not 1 <= limit <= 100 or type(offset) is not int or not 0 <= offset <= 1000000 \
            or type(repeat) is not int or not 1 <= repeat <= 5 or (mode == "replay" and repeat != 1):
        raise ValueError("Evaluation needs bounded request count offset and repeats")
    if type(generation_seconds) is not int or not 1 <= generation_seconds <= 3600 or type(total_seconds) is not int or not 1 <= total_seconds <= 86400:
        raise ValueError("Invalid evaluation timing limits")
    if output.is_relative_to(source) or source.is_relative_to(output) or output.exists():
        raise ValueError("Evaluation output must be a new directory separate from the source run")
    if not (source / "memory.sqlite").is_file():
        raise ValueError("Evaluation source needs an archived run database")
    criteria = load_labels(labels)
    started = time.monotonic()

    def check():
        if time.monotonic() - started >= total_seconds:
            report["termination"] = "total_time_budget_exhausted"
            write_json(output / "evaluation-report.json", report)
            raise TimeoutError("Evaluation total time budget exhausted")

    report = {"schema_version": 1, "mode": mode, "source_run": str(source), "controller_inputs_sent": 0,
              "semantic_success_verified": False, "results": [], "completed": False,
              "validation_scope": "Schema supplied visual evidence and archived review and recovery gates; database write legality and semantic success are not verified",
              "scope": "Original archived requests with independent proposals and no rollout or source memory updates"}
    output.mkdir(parents=True)
    with RunLock(source / ".lock"), tempfile.TemporaryDirectory() as temporary:
        database = Path(temporary) / "memory.sqlite"
        staging = Path(temporary) / "source.sqlite"
        shutil.copyfile(source / "memory.sqlite", staging)
        if (source / "memory.sqlite-wal").is_file():
            shutil.copyfile(source / "memory.sqlite-wal", Path(str(staging) + "-wal"))
        with closing(sqlite3.connect(staging)) as original, closing(sqlite3.connect(database)) as snapshot:
            original.backup(snapshot)
        with closing(Memory(database)) as memory:
            ids = [row[0] for row in memory.connection.execute("SELECT id FROM conversation_requests ORDER BY id LIMIT ? OFFSET ?", (limit, offset))]
            report["selected_request_ids"] = ids
            if not ids:
                raise ValueError("No archived requests selected for evaluation")
            report["unselected_label_request_ids"] = sorted(set(criteria) - set(ids))
            report["source_metrics"] = {
                "unknown_steps": memory.connection.execute("SELECT COUNT(*) FROM action_steps WHERE status='unknown'").fetchone()[0],
                "loop_episodes": memory.connection.execute("SELECT COUNT(*) FROM progress_episodes").fetchone()[0],
                "recovery_attempts": memory.connection.execute("SELECT COUNT(*) FROM recovery_attempts").fetchone()[0],
                "review_cases": len(memory.escalations.cases())}
            prepared = False
            for request_id in ids:
                for attempt in range(1, repeat + 1):
                    result = {"request_id": request_id, "attempt": attempt}
                    begin = time.monotonic()
                    try:
                        check()
                        archived = memory.conversation.request(request_id)
                        request = rehydrate(memory, source, archived)
                        context = json.loads(request[1]["content"][0]["text"])
                        row = memory.connection.execute("SELECT payload FROM conversation_events WHERE request_id=? AND kind='assistant_decision' ORDER BY id DESC LIMIT 1", (request_id,)).fetchone()
                        expected = json.loads(row[0])["decision"] if row else None
                        if mode == "shadow":
                            if isinstance(agent, LMStudio) and not prepared:
                                agent.prepare()
                                prepared = True
                                check()
                            generation_started = time.monotonic()
                            decision, trace = bounded_decision(agent, request, generation_seconds, check)
                            result["generation_elapsed_seconds"] = time.monotonic() - generation_started
                            if len(json.dumps(trace, ensure_ascii=False).encode()) > 1048576:
                                raise ValueError("Evaluation model trace exceeds its byte limit")
                            write_json(output / f"request-{request_id}-attempt-{attempt}.model.json", trace)
                            result["usage"] = trace.get("response", {}).get("usage", trace.get("usage"))
                            result["context_budget"] = trace.get("context_budget")
                        else:
                            if expected is None:
                                result.update(status="unscored", disposition=archived["status"])
                                continue
                            decision = expected
                        validate_decision(decision, archived["observation_id"], context.get("action_limit_frames", 60), context.get("max_sequence_length", 5))
                        validate_visual_reviews(decision, context)
                        validate_context_gates(decision, context)
                        result.update(status="valid", proposed_input_plan=input_plan(decision),
                                      labels=label_result(decision, criteria.get(request_id)))
                        if isinstance(expected, dict) and "action" in expected and "additional_actions" in expected:
                            result["same_as_archived_input_plan"] = input_plan(decision) == input_plan(expected)
                    except Exception as exc:
                        result.update(status="error", error_type=type(exc).__name__, error=str(exc)[:1000])
                        trace = getattr(exc, "trace", None)
                        if isinstance(trace, dict) and len(json.dumps(trace, ensure_ascii=False).encode()) <= 1048576:
                            write_json(output / f"request-{request_id}-attempt-{attempt}.model.json", trace)
                        if isinstance(exc, TimingLimit):
                            report["termination"] = "generation_time_budget_exhausted"
                            raise
                    except KeyboardInterrupt:
                        result.update(status="abandoned", error="User interrupted evaluation")
                        report["termination"] = "user_interruption"
                        raise
                    finally:
                        result["elapsed_seconds"] = time.monotonic() - begin
                        report["results"].append(result)
                        write_json(output / "evaluation-report.json", report)
                    check()
            report["completed"] = True
    report["elapsed_seconds"] = time.monotonic() - started
    report["summary"] = {"valid": sum(item["status"] == "valid" for item in report["results"]),
                         "errors": sum(item["status"] == "error" for item in report["results"]),
                         "unscored": sum(item["status"] == "unscored" for item in report["results"]),
                         "label_failures": sum(item.get("labels") is not None and not item["labels"]["matched"] for item in report["results"])}
    report["proposal_variants"] = {str(identity): len({json.dumps(item["proposed_input_plan"], sort_keys=True)
        for item in report["results"] if item["request_id"] == identity and item["status"] == "valid"}) for identity in ids}
    write_json(output / "evaluation-report.json", report)
    return report
