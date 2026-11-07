import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

from .agent.lmstudio import LMStudio
from .config import load_config
from .agent.schemas import BUTTONS
from .control.diagnostics import input_test
from .control.session import RunControl, request_stop
from .emulator.interface import ReplayEmulator
from .memory.database import Memory
from .runner import run
from .telemetry.recorder import write_json


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
                      and path.parent.name != "demo"]
        if not candidates:
            raise ValueError("No saved runs found. Start one with pokemon-harness run")
        directory = max(candidates, key=lambda path: (path / "config.json").stat().st_mtime_ns)
    directory = Path(directory).resolve()
    if not (directory / "memory.sqlite").is_file() or not (directory / "config.json").is_file():
        raise ValueError(f"No saved run at {directory}")
    return directory


def play_run(args):
    if args.turns < 1:
        raise ValueError("turns must be a positive integer")
    if args.command == "resume":
        directory = saved_run(args.run_dir, args.runs_dir)
        config_path = directory / "config.json"
        config = load_config(config_path)
        config["run_dir"] = str(directory)
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
    agent = LMStudio(config["model"])
    agent.prepare()
    thinking = ("on" if config["model"]["thinking"] else "off") if "thinking" in config["model"] else "model default"
    print(f"Generation: max_tokens={config['model'].get('max_tokens', 32768)}, thinking={thinking}", flush=True)
    emulator = emulator_for(config, config_path)
    print(f"Run: {directory}\nObjective: {config['objective']}", flush=True)
    print("Stop: F8, Ctrl+C, or pokemon-harness stop in another terminal. Emulator keeps running.", flush=True)
    print(json.dumps(run(config, emulator, agent, args.turns, control=RunControl(directory))))


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
    stop = commands.add_parser("stop", aliases=["pause"], help="Stop the latest harness without pausing the emulator")
    stop.add_argument("run_dir", nargs="?", type=Path)
    stop.add_argument("--runs-dir", type=Path, default=Path("runs"))
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
    demo = commands.add_parser("demo", help="Offline synthetic loop: no emulator or LM Studio required")
    demo.add_argument("--run-dir", type=Path, default=Path("runs/demo"))
    args = parser.parse_args(argv)
    try:
        if args.command in {"run", "resume"}:
            play_run(args)
        elif args.command in {"stop", "pause"}:
            directory = saved_run(args.run_dir, args.runs_dir)
            request_stop(directory)
            print(f"Stop requested for {directory}. The current move may finish; no further moves will start.")
        elif args.command == "memory":
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
                emulator = emulator_for(config, args.config)
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
