import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pokemon_harness.cli import main
from pokemon_harness.control.registry import active_sessions, read_record, select_session
from pokemon_harness.control.session import RunControl, StopRequested, request_stop
from pokemon_harness.telemetry.recorder import RunLock, write_json


class SessionRegistryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.registry = self.root / "registry"
        environment = patch.dict("os.environ", {"POKEMON_HARNESS_SESSION_DIR": str(self.registry)})
        environment.start()
        self.addCleanup(environment.stop)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)

    def start_session(self, name):
        directory = self.root / name
        directory.mkdir(exist_ok=True)
        self.stack.enter_context(RunLock(directory / ".lock"))
        control = RunControl(directory, hotkey=False)
        control.start()
        self.stack.callback(control.close)
        return directory, control

    def test_default_stop_targets_active_session_instead_of_latest_saved_run(self):
        directory, control = self.start_session("running")
        inactive = self.root / "newer-stopped-run"
        inactive.mkdir()
        write_json(inactive / "config.json", {"newer": True})
        self.assertEqual(main(["stop"]), 0)
        self.assertFalse((inactive / ".stop").exists())
        with self.assertRaises(StopRequested):
            control.check()
        self.assertEqual(read_record(directory / ".stop")["session_id"], control.record["session_id"])

    def test_ambiguous_stop_requires_selection_and_targeted_stop_spares_other_session(self):
        first_dir, first = self.start_session("first")
        second_dir, second = self.start_session("second")
        self.assertEqual(main(["stop"]), 1)
        first.check()
        second.check()
        self.assertEqual(main(["stop", "--session", second.record["session_id"]]), 0)
        first.check()
        with self.assertRaises(StopRequested):
            second.check()
        self.assertFalse((first_dir / ".stop").exists())

    def test_old_stop_request_cannot_stop_resumed_session(self):
        directory, old = self.start_session("resume")
        old_id = old.record["session_id"]
        old.close()
        new = RunControl(directory, hotkey=False)
        new.start()
        self.stack.callback(new.close)
        write_json(directory / ".stop", {"session_id": old_id})
        new.check()
        with self.assertRaisesRegex(ValueError, "no longer active"):
            request_stop(directory, old_id)
        request_stop(directory, new.record["session_id"])
        with self.assertRaises(StopRequested):
            new.check()
        # Delayed cleanup of an old controller must not remove the new record.
        old.close()
        self.assertEqual(select_session()["session_id"], new.record["session_id"])

    def test_unlocked_crashed_session_and_malformed_records_are_not_active(self):
        directory = self.root / "crashed"
        directory.mkdir()
        lock = RunLock(directory / ".lock")
        lock.__enter__()
        control = RunControl(directory, hotkey=False)
        control.start()
        self.stack.callback(control.close)
        self.assertEqual(len(active_sessions()), 1)
        lock.__exit__()
        (self.registry / "broken.json").write_text("{", encoding="utf-8")
        self.assertEqual(active_sessions(), [])
        self.assertEqual(main(["stop"]), 1)

    def test_status_lists_current_sessions_and_cleanup_removes_registration(self):
        directory, control = self.start_session("status")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["status", "--json"]), 0)
        records = json.loads(output.getvalue())
        self.assertEqual(records[0]["run_dir"], str(directory.resolve()))
        control.close()
        self.assertEqual(active_sessions(), [])
        self.assertFalse((directory / ".session.json").exists())

    def test_explicit_directory_and_scope_select_only_matching_session(self):
        first_dir, first = self.start_session("first")
        self.start_session("second")
        self.assertEqual(select_session(run_dir=first_dir)["session_id"], first.record["session_id"])
        self.assertEqual(select_session(runs_dir=first_dir)["session_id"], first.record["session_id"])
        with self.assertRaisesRegex(ValueError, "not both"):
            select_session(first_dir, first.record["session_id"])

    def test_another_process_can_discover_and_stop_active_session(self):
        directory, control = self.start_session("external")
        result = subprocess.run([sys.executable, "-m", "pokemon_harness", "status", "--json"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)[0]["session_id"], control.record["session_id"])
        result = subprocess.run([sys.executable, "-m", "pokemon_harness", "stop"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        with self.assertRaises(StopRequested):
            control.check()
