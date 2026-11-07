import argparse
import json
import sys
import uuid
from datetime import datetime
from contextlib import closing
from pathlib import Path

from .agent.lmstudio import LMStudio
from .config import load_config
from .agent.schemas import BUTTONS
from .control.diagnostics import input_test
from .control.session import RunControl, request_stop
from .control.registry import active_sessions, select_session
from .emulator.interface import ReplayEmulator
from .memory.database import Memory
from .runner import run
from .telemetry.recorder import RunLock, write_json
from .telemetry.storage import StorageGuard
from .telemetry.cleanup import closing_safely
from .telemetry.backup import create_backup, verify_backup, require_live_run
from .telemetry.restore import restore_backup


def emulator_for(config, config_path):
    settings = config["emulator"]
    if settings["backend"] == "mgba":
        from .emulator.mgba import MGBA
        return MGBA(settings)
    if settings["backend"] == "vbam":
        from .emulator.vbam import VBAM
        return VBAM(settings)
    if settings["backend"] == "replay":
        directory = (Path(config_path).resolve().parent / settings["screenshots_dir"]).resolve()
        return ReplayEmulator(sorted(directory.glob("*.png")))
    raise ValueError("emulator.backend must be mgba, vbam, or replay")


def saved_run(directory, runs_dir):
    if directory is None:
        candidates = [path.parent for path in Path(runs_dir).glob("*/config.json")
                      if (path.parent / "memory.sqlite").is_file()
                      and path.parent.name != "demo" and not path.parent.name.startswith(".")]
        if not candidates:
            raise ValueError("No saved runs found. Start one with pokemon-harness run")
        directory = max(candidates, key=lambda path: (path / "config.json").stat().st_mtime_ns)
    directory = Path(directory).resolve()
    require_live_run(directory)
    if not (directory / "memory.sqlite").is_file() or not (directory / "config.json").is_file():
        raise ValueError(f"No saved run at {directory}")
    return directory


def play_run(args):
    if args.turns < 1:
        raise ValueError("turns must be a positive integer")
    if args.command == "resume":
        directory = saved_run(args.run_dir, args.runs_dir)
        config_path = directory / "config.json"
        require_live_run(directory)
        config = load_config(config_path)
        config["run_dir"] = str(directory)
        StorageGuard(config).check()
        # checking durable cases before loading a model or opening the emulator
        with RunLock(directory / ".lock"), closing(Memory(directory / "memory.sqlite")) as memory:
            memory.initialize(config)
            memory.monitor_progress()
            review = memory.escalations.review(memory.events())
            if review["blocked_cases"] or review["validity"] == "invalid":
                raise ValueError("Run needs review or is invalid. Use pokemon-harness cases and resolve-case")
    else:
        config_path = args.config
        config = load_config(config_path)
        if args.goal == "first-gym":
            config["objective"] = ("Beat the first Gym Leader in Pokemon Emerald and obtain the Stone Badge "
                                   "while following the Nuzlocke rules. Stop once the badge is visibly "
                                   "confirmed, or if the entire usable party faints.")
        elif args.objective:
            config["objective"] = args.objective
        name = datetime.now().strftime("emerald-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        directory = (args.run_dir or (args.runs_dir / name)).resolve()
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("New run directory must be empty. Use pokemon-harness resume to continue a run")
        config["run_dir"] = str(directory)
        if config["emulator"]["backend"] == "replay":
            config["emulator"]["screenshots_dir"] = str(
                (config_path.resolve().parent / config["emulator"]["screenshots_dir"]).resolve())
    if args.context_length is not None:
        config["model"]["context_length"] = args.context_length
    if args.max_tokens is not None:
        config["model"]["max_tokens"] = args.max_tokens
    if args.thinking is not None:
        config["model"]["thinking"] = args.thinking == "on"
    StorageGuard(config).check()
    agent = LMStudio(config["model"])
    # holding emulator ownership before model loading
    # releasing ownership on startup and run failures
    with closing_safely(emulator_for(config, config_path), "emulator") as emulator:
        agent.prepare()
        thinking = ("on" if config["model"]["thinking"] else "off") if "thinking" in config["model"] else "model default"
        print(f"Generation: max_tokens={config['model'].get('max_tokens', 32768)}, thinking={thinking}", flush=True)
        print(f"Run: {directory}\nObjective: {config['objective']}", flush=True)
        print("Stop: F8 (all local harnesses), Ctrl+C, or pokemon-harness stop in another terminal. "
              "Emulator keeps running.", flush=True)
        result = run(config, emulator, agent, args.turns, control=RunControl(directory))
    if result.get("reason") == "user_stop":
        print("Harness stopped. Memory saved; inputs released. Use pokemon-harness resume.")
    elif result.get("reason") in {"manual_handoff", "control_lost"}:
        print("Control returned to you. Memory saved; inputs released. "
              "Resume will inspect a fresh screenshot and receive an intervention warning.")
    print(json.dumps(result))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Screenshot-only Pokemon agent harness")
    commands = parser.add_subparsers(dest="command", required=True)
    play = commands.add_parser("run", help="Start a fresh run using configs/emerald.json by default")
    play.add_argument("--config", type=Path, default=Path("configs/emerald.json"))
    play.add_argument("--context-length", type=int, help="LM Studio load-time context size in tokens")
    play.add_argument("--max-tokens", type=int, help="Generation output budget, including reasoning")
    play.add_argument("--thinking", choices=["on", "off"], help="Toggle thinking for compatible model templates")
    play.add_argument("--run-dir", type=Path, help="Optional empty directory for the new run")
    play.add_argument("--runs-dir", type=Path, default=Path("runs"))
    goal = play.add_mutually_exclusive_group()
    goal.add_argument("--goal", choices=["first-gym"], help="Use a predefined objective")
    goal.add_argument("--objective", help="Override the new run's objective")
    play.add_argument("--turns", type=int, default=500,
                      help="Maximum model plans; each can contain multiple controller moves")
    resume = commands.add_parser("resume", help="Resume the latest run, or a specified run directory")
    resume.add_argument("run_dir", nargs="?", type=Path)
    resume.add_argument("--runs-dir", type=Path, default=Path("runs"))
    resume.add_argument("--turns", type=int, default=500)
    resume.add_argument("--context-length", type=int, help="Override the saved model's load-time context size")
    resume.add_argument("--max-tokens", type=int, help="Override the saved generation output budget")
    resume.add_argument("--thinking", choices=["on", "off"], help="Toggle thinking for compatible model templates")
    status = commands.add_parser("status", help="List active harness sessions")
    status.add_argument("--json", action="store_true", help="Print machine-readable session records")
    stop = commands.add_parser("stop", aliases=["pause"], help="Stop the active harness without pausing the emulator")
    stop.add_argument("run_dir", nargs="?", type=Path)
    stop.add_argument("--session", help="Target an exact active session ID")
    stop.add_argument("--runs-dir", type=Path, help="Restrict selection to runs below this directory")
    handoff = commands.add_parser("handoff", help="Stop the active harness for manual play")
    handoff.add_argument("run_dir", nargs="?", type=Path)
    handoff.add_argument("--session", help="Target an exact active session ID")
    handoff.add_argument("--runs-dir", type=Path, help="Restrict selection to runs below this directory")
    capture = commands.add_parser("capture", help="Save a live screenshot to verify cropping and bindings")
    capture.add_argument("--config", required=True, type=Path)
    capture.add_argument("--output", type=Path, default=Path("capture.png"))
    controls = commands.add_parser("input-test", help="Test one controller button without LM Studio")
    controls.add_argument("--config", required=True, type=Path)
    controls.add_argument("--button", choices=BUTTONS, default="START")
    controls.add_argument("--hold-frames", type=int, default=12)
    controls.add_argument("--settle-frames", type=int, default=45)
    inspect = commands.add_parser("memory", help="Read durable facts and ledger as JSON")
    inspect.add_argument("--run-dir", required=True, type=Path)
    state = commands.add_parser("state", help="Inspect working state and subgoal history for a stopped run")
    state.add_argument("run_dir", nargs="?", type=Path)
    state.add_argument("--runs-dir", type=Path, default=Path("runs"))
    state.add_argument("--history", action="store_true")
    state.add_argument("--limit", type=int, default=20)
    state.add_argument("--offset", type=int, default=0)
    outcomes = commands.add_parser("outcomes", help="Inspect pending action reviews and history for a stopped run")
    outcomes.add_argument("run_dir", nargs="?", type=Path)
    outcomes.add_argument("--runs-dir", type=Path, default=Path("runs"))
    outcomes.add_argument("--history", action="store_true")
    outcomes.add_argument("--limit", type=int, default=20)
    outcomes.add_argument("--offset", type=int, default=0)
    progress = commands.add_parser("progress", help="Inspect progress and recovery history for a stopped run")
    progress.add_argument("run_dir", nargs="?", type=Path)
    progress.add_argument("--runs-dir", type=Path, default=Path("runs"))
    progress.add_argument("--history", action="store_true")
    progress.add_argument("--limit", type=int, default=20)
    progress.add_argument("--offset", type=int, default=0)
    conflicts = commands.add_parser("contradictions", help="Inspect disputed memories and history for a stopped run")
    conflicts.add_argument("run_dir", nargs="?", type=Path)
    conflicts.add_argument("--runs-dir", type=Path, default=Path("runs"))
    conflicts.add_argument("--history", action="store_true")
    conflicts.add_argument("--limit", type=int, default=20)
    conflicts.add_argument("--offset", type=int, default=0)
    conversation = commands.add_parser("conversation", help="Inspect the conversation journal for a stopped run")
    conversation.add_argument("run_dir", nargs="?", type=Path)
    conversation.add_argument("--runs-dir", type=Path, default=Path("runs"))
    conversation.add_argument("--history", action="store_true")
    conversation.add_argument("--limit", type=int, default=20)
    conversation.add_argument("--offset", type=int, default=0)
    conversation.add_argument("--request", type=int, help="Show an archived request with observation image references")
    conversation.add_argument("--checkpoint", action="store_true", help="Show the full latest checkpoint snapshot")
    checkpoint = commands.add_parser("checkpoint", help="Commit a conversation checkpoint without starting the emulator")
    checkpoint.add_argument("run_dir", nargs="?", type=Path)
    checkpoint.add_argument("--runs-dir", type=Path, default=Path("runs"))
    allowance = commands.add_parser("allow-repeat", help="Grant a stopped run a narrow subgoal repetition allowance")
    allowance.add_argument("run_dir", type=Path)
    allowance.add_argument("--subgoal", required=True)
    allowance.add_argument("--strategy", required=True, choices=["move", "advance_dialogue", "wait"])
    allowance.add_argument("--button", choices=["UP", "DOWN", "LEFT", "RIGHT", "A", "B"])
    allowance.add_argument("--moves", type=int, required=True)
    allowance.add_argument("--seconds", type=int, required=True)
    allowance.add_argument("--reason", required=True)
    allowance.add_argument("--until", required=True, help="Visible termination condition")
    extension = commands.add_parser("extend-runtime", help="Record an explicit cumulative runtime extension for a stopped run")
    extension.add_argument("run_dir", type=Path)
    extension.add_argument("--seconds", type=int, required=True)
    extension.add_argument("--reason", required=True)
    evaluation = commands.add_parser("evaluate", help="Replay archived requests or compare shadow proposals without game input")
    evaluation.add_argument("run_dir", type=Path)
    evaluation.add_argument("--mode", choices=["replay", "shadow"], default="replay")
    evaluation.add_argument("--output-dir", type=Path)
    evaluation.add_argument("--config", type=Path, help="Model configuration for shadow mode")
    evaluation.add_argument("--limit", type=int, default=20)
    evaluation.add_argument("--offset", type=int, default=0)
    evaluation.add_argument("--repeat", type=int, default=1)
    evaluation.add_argument("--labels", type=Path)
    evaluation.add_argument("--generation-seconds", type=int, default=600)
    evaluation.add_argument("--total-seconds", type=int, default=3600)
    cases = commands.add_parser("cases", help="Inspect durable review cases for the latest or specified run")
    cases.add_argument("run_dir", nargs="?", type=Path)
    cases.add_argument("--runs-dir", type=Path, default=Path("runs"))
    cases.add_argument("--case", type=int, help="Show one case including its resolution history")
    create_case = commands.add_parser("case-open", help="Record uncertainty that blocks further input")
    create_case.add_argument("run_dir", nargs="?", type=Path)
    create_case.add_argument("--runs-dir", type=Path, default=Path("runs"))
    create_case.add_argument("--reason", required=True)
    create_case.add_argument("--observation-id", action="append", type=int, default=[])
    resolve = commands.add_parser("resolve-case", help="Append an evidence resolution or explicit user decision")
    resolve.add_argument("case_id", type=int)
    resolve.add_argument("run_dir", nargs="?", type=Path)
    resolve.add_argument("--runs-dir", type=Path, default=Path("runs"))
    resolve.add_argument("--outcome", choices=["evidence", "user-assertion", "unverifiable", "invalid"], required=True)
    resolve.add_argument("--reason", required=True)
    resolve.add_argument("--observation-id", action="append", type=int, default=[])
    resolve.add_argument("--correction-key", default="")
    resolve.add_argument("--allow-continue", action="store_true",
                         help="Explicitly continue an unverifiable run without changing the event ledger")
    backup = commands.add_parser("backup", help="Snapshot a stopped run with its memory and evidence")
    backup.add_argument("run_dir", nargs="?", type=Path)
    backup.add_argument("--runs-dir", type=Path, default=Path("runs"))
    backup.add_argument("--backup-dir", type=Path, help="Backup storage directory separate from the run")
    verify = commands.add_parser("verify-backup", help="Check a backup manifest and its database evidence")
    verify.add_argument("backup_dir", type=Path)
    restore = commands.add_parser("restore", help="Restore harness state into a new run requiring game-state review")
    restore.add_argument("backup_dir", type=Path)
    restore.add_argument("--run-dir", type=Path, help="New destination directory that does not already exist")
    restore.add_argument("--runs-dir", type=Path, default=Path("runs"))
    demo = commands.add_parser("demo", help="Offline synthetic loop: no emulator or LM Studio required")
    demo.add_argument("--run-dir", type=Path, default=Path("runs/demo"))
    args = parser.parse_args(argv)
    try:
        if args.command == "extend-runtime":
            from .control.timing import RuntimeBudget
            directory = saved_run(args.run_dir, Path("runs"))
            config = load_config(directory / "config.json")
            with RunLock(directory / ".lock"), closing(Memory(directory / "memory.sqlite")) as memory:
                runtime = RuntimeBudget(memory, config)
                runtime.extend(args.seconds, args.reason)
                print(json.dumps({"added_seconds": args.seconds, "reason": args.reason}, indent=2))
            return 0
        if args.command == "evaluate":
            from .evaluation import evaluate
            directory = saved_run(args.run_dir, Path("runs"))
            output = args.output_dir or Path("evaluations") / (datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6])
            agent = None
            if args.mode == "shadow":
                config = load_config(args.config or directory / "config.json")
                agent = LMStudio(config["model"])
            report = evaluate(directory, output, args.mode, agent, args.limit, args.offset, args.repeat,
                              args.labels, args.generation_seconds, args.total_seconds)
            print(json.dumps({"report": str(output.resolve() / "evaluation-report.json"), "summary": report["summary"],
                              "controller_inputs_sent": report["controller_inputs_sent"]}, indent=2))
            return 0 if not report["summary"]["errors"] and not report["summary"]["label_failures"] else 1
        if args.command == "allow-repeat":
            directory = saved_run(args.run_dir, Path("runs"))
            with RunLock(directory / ".lock"), closing(Memory(directory / "memory.sqlite")) as memory:
                state = memory.working_context()
                if not state.get("subgoal") or state["subgoal"]["id"] != args.subgoal:
                    raise ValueError("Repetition allowance must name the current subgoal exactly")
                memory.repetition.grant(state, args.strategy, [args.button] if args.button else [],
                                        args.moves, args.seconds, args.reason, args.until)
                print(json.dumps(memory.repetition.context(state), indent=2))
            return 0
        if args.command in {"run", "resume"}:
            play_run(args)
        elif args.command == "backup":
            directory = saved_run(args.run_dir, args.runs_dir)
            target = create_backup(directory, args.backup_dir)
            print(json.dumps({"backup_dir": str(target), "verified": True,
                              "scope": "harness_state_only", "emulator_save_included": False}, indent=2))
        elif args.command == "verify-backup":
            print(json.dumps(verify_backup(args.backup_dir), indent=2))
        elif args.command == "restore":
            print(json.dumps(restore_backup(args.backup_dir, args.run_dir, args.runs_dir), indent=2))
        elif args.command == "checkpoint":
            directory = saved_run(args.run_dir, args.runs_dir)
            config = load_config(directory / "config.json")
            StorageGuard(config).check(128 * 1024)
            with RunLock(directory / ".lock"), closing(Memory(directory / "memory.sqlite")) as memory:
                memory.initialize(config)
                memory.conversation.active_checkpoint()
                memory.recover()
                checkpoint_id = memory.save_checkpoint()
                print(json.dumps({"run_dir": str(directory), "checkpoint_id": checkpoint_id,
                                  "boundary_id": memory.conversation.active_checkpoint()["boundary_id"]}, indent=2))
        elif args.command == "conversation":
            directory = saved_run(args.run_dir, args.runs_dir)
            with RunLock(directory / ".lock"), closing(Memory(directory / "memory.sqlite")) as memory:
                result = {"run_dir": str(directory), "conversation": memory.conversation.context()}
                if args.history:
                    result["history"] = memory.conversation.history(args.limit, args.offset)
                if args.request is not None:
                    result["request"] = memory.conversation.request(args.request)
                if args.checkpoint:
                    result["checkpoint"] = memory.conversation.active_checkpoint()
                print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command in {"state", "outcomes", "progress", "contradictions"}:
            directory = saved_run(args.run_dir, args.runs_dir)
            with RunLock(directory / ".lock"), closing(Memory(directory / "memory.sqlite")) as memory:
                store = {"state": memory.working, "outcomes": memory.outcomes, "progress": memory.progress,
                         "contradictions": memory.contradictions}[args.command]
                key = {"state": "working_state", "outcomes": "outcome_review", "progress": "progress",
                       "contradictions": "contradictions"}[args.command]
                value = (memory.working_context() if args.command == "state" else
                         memory.monitor_progress() if args.command == "progress" else store.context())
                result = {"run_dir": str(directory), key: value}
                if args.history:
                    result["history"] = store.history(args.limit, args.offset)
                print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command in {"cases", "case-open", "resolve-case"}:
            directory = saved_run(args.run_dir, args.runs_dir)
            require_live_run(directory)
            with RunLock(directory / ".lock"), closing(Memory(directory / "memory.sqlite")) as memory:
                memory.escalations.review(memory.events())
                if args.command == "case-open":
                    memory.escalations.open("manual:" + uuid.uuid4().hex, args.reason, args.observation_id,
                                            {"source": "user"})
                elif args.command == "resolve-case":
                    memory.escalations.resolve(args.case_id, args.outcome, args.reason, args.observation_id,
                                               args.correction_key, args.allow_continue)
                records = memory.escalations.cases()
                if args.command == "cases" and args.case is not None:
                    records = [case for case in records if case["id"] == args.case]
                    if not records:
                        raise ValueError(f"No case {args.case} in this run")
                print(json.dumps({"run_dir": str(directory), "validity": memory.escalations.validity(),
                                  "cases": records}, indent=2))
                write_json(directory / "memory-export.json", memory.export())
        elif args.command in {"stop", "pause", "handoff"}:
            session = select_session(args.run_dir, args.session, args.runs_dir)
            kind = "handoff" if args.command == "handoff" else "stop"
            request_stop(session["run_dir"], session["session_id"], kind=kind)
            print(f"Stop requested for session {session['session_id']} ({session['run_dir']}). "
                  "The current move may finish. Wait for the harness to confirm it stopped.")
        elif args.command == "status":
            sessions = active_sessions()
            if args.json:
                print(json.dumps(sessions, indent=2))
            elif not sessions:
                print("No active harness sessions")
            else:
                for session in sessions:
                    print(f"{session['session_id']}  PID {session['pid']}  {session['run_dir']}")
        elif args.command == "memory":
            require_live_run(args.run_dir)
            path = args.run_dir / "memory.sqlite"
            if not path.is_file():
                raise ValueError(f"No run database at {path}")
            memory = Memory(path)
            try:
                print(json.dumps(memory.export(), ensure_ascii=False, indent=2))
            finally:
                memory.close()
        elif args.command == "demo":
            from .demo import DemoAgent, create_images
            directory = args.run_dir.resolve()
            require_live_run(directory)
            directory.mkdir(parents=True, exist_ok=True)
            config_path = directory / "demo-config.json"
            write_json(config_path, {"run_dir": ".", "objective": "Exercise the synthetic vision-only loop",
                                     "model": {"name": "scripted-demo"}, "emulator": {"backend": "replay"}})
            config = load_config(config_path)
            emulator = ReplayEmulator(create_images(directory / "fixtures"))
            print(json.dumps(run(config, emulator, DemoAgent(), turns=3)))
        else:
            config = load_config(args.config)
            if args.command == "input-test":
                require_live_run(config["run_dir"])
                StorageGuard(config).check()
                with closing_safely(emulator_for(config, args.config), "emulator") as emulator:
                    input_test(config, emulator, args.button, args.hold_frames, args.settle_frames)
            else:
                emulator = emulator_for(config, args.config)
                try:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_bytes(emulator.capture().image)
                    print(f"Screenshot saved to {args.output.resolve()}")
                finally:
                    emulator.close()
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Held keys are released; the emulator keeps running.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
