import argparse
import json
import sys
from pathlib import Path

from .agent.lmstudio import LMStudio
from .config import load_config
from .agent.schemas import BUTTONS
from .control.diagnostics import input_test
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Screenshot-only Pokemon agent harness")
    commands = parser.add_subparsers(dest="command", required=True)
    play = commands.add_parser("run", help="Run bounded model turns; reuse run_dir to resume")
    play.add_argument("--config", required=True, type=Path)
    play.add_argument("--turns", type=int, default=20)
    capture = commands.add_parser("capture", help="Save a paused screenshot to verify cropping and bindings")
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
        if args.command == "memory":
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
            if args.command == "run":
                # Validate model selection before opening the emulator backend.
                agent = LMStudio(config["model"])
                emulator = emulator_for(config, args.config)
                print(json.dumps(run(config, emulator, agent, args.turns)))
            elif args.command == "input-test":
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
        print("Interrupted. Check emulator pause state before resuming.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
