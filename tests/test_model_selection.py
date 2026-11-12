import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pokemon_harness.cli import main
from pokemon_harness.control.session import RunControl
from pokemon_harness.demo import DemoAgent, create_images
from pokemon_harness.memory.database import Memory
from pokemon_harness.telemetry.recorder import write_json


class FixtureModel(DemoAgent):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def prepare(self):
        pass


class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "run"
        create_images(self.root / "images")
        self.config = self.root / "config.json"
        write_json(self.config, {"run_dir": "unused", "objective": "Keep this goal", "rules": ["Keep this rule"],
            "model": {"name": "original"}, "emulator": {"backend": "replay", "screenshots_dir": "images"}})
        env = patch.dict("os.environ", {"POKEMON_HARNESS_SESSION_DIR": str(self.root / "registry")})
        env.start()
        self.addCleanup(env.stop)

    def invoke(self, arguments, factory=FixtureModel):
        with patch("pokemon_harness.cli.LMStudio", side_effect=factory) as model, \
                patch("pokemon_harness.cli.RunControl", side_effect=lambda directory: RunControl(directory, hotkey=False)), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = main(arguments)
        return result, model

    def start(self):
        return self.invoke(["run", "--config", str(self.config), "--run-dir", str(self.directory),
                            "--model", "larger", "--turns", "1"])

    def test_run_and_resume_switch_models_without_resetting_goal_rules_or_turn_history(self):
        result, model = self.start()
        self.assertEqual(result, 0)
        self.assertEqual(model.call_args.args[0]["name"], "larger")
        with contextlib.closing(Memory(self.directory / "memory.sqlite")) as memory:
            first = memory.recent_turns()[0]["decision"]
        result, model = self.invoke(["resume", str(self.directory), "--model", " larger-loaded-instance ",
                                    "--thinking", "off", "--turns", "1"])
        self.assertEqual(result, 0)
        self.assertEqual(model.call_args.args[0]["name"], "larger-loaded-instance")
        config = json.loads((self.directory / "config.json").read_text())
        self.assertEqual(config["model"]["name"], "larger-loaded-instance")
        self.assertEqual(config["objective"], "Keep this goal")
        self.assertEqual(config["rules"], ["Keep this rule"])
        with contextlib.closing(Memory(self.directory / "memory.sqlite")) as memory:
            turns = memory.recent_turns()
            self.assertEqual(len(turns), 2)
            self.assertEqual(next(turn for turn in turns if turn["id"] == 1)["decision"], first)

    def test_failed_model_load_does_not_replace_the_saved_model(self):
        self.assertEqual(self.start()[0], 0)
        original = (self.directory / "config.json").read_bytes()

        class FailingModel(FixtureModel):
            def prepare(self):
                raise RuntimeError("Load failed")

        result, _ = self.invoke(["resume", str(self.directory), "--model", "cannot-load", "--turns", "1"], FailingModel)
        self.assertEqual(result, 1)
        self.assertEqual((self.directory / "config.json").read_bytes(), original)

    def test_blank_model_identifier_is_rejected_before_model_loading(self):
        result, model = self.invoke(["run", "--config", str(self.config), "--model", "  ", "--turns", "1"])
        self.assertEqual(result, 1)
        model.assert_not_called()
