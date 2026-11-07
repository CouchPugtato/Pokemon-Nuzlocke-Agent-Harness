import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pokemon_harness.agent.schemas import BUTTONS
from pokemon_harness.cli import main
from pokemon_harness.control.ownership import EmulatorOwnership, ProcessIdentity, WindowsTarget
from pokemon_harness.emulator.desktop import WindowsDesktopEmulator
from pokemon_harness.telemetry.recorder import write_json


class EmulatorOwnershipTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = patch.dict("os.environ", {"POKEMON_HARNESS_SESSION_DIR": str(self.root / "sessions")})
        environment.start()
        self.addCleanup(environment.stop)
        self.identity = ProcessIdentity(12345, 999999)

    def test_same_process_is_exclusive_and_released_owner_cannot_send_input(self):
        first = EmulatorOwnership(self.identity)
        with first:
            first.check()
            with self.assertRaisesRegex(RuntimeError, "Another harness owns emulator"):
                with EmulatorOwnership(self.identity):
                    self.fail("A second controller acquired the emulator")
        with self.assertRaisesRegex(RuntimeError, "ownership has been released"):
            first.check()
        with EmulatorOwnership(self.identity):
            pass

    def test_reused_pid_with_new_creation_time_has_a_different_identity(self):
        with EmulatorOwnership(self.identity):
            with EmulatorOwnership(ProcessIdentity(self.identity.pid, self.identity.created_at + 1)):
                pass

    def test_another_process_cannot_acquire_an_owned_emulator(self):
        script = """
from pokemon_harness.control.ownership import EmulatorOwnership, ProcessIdentity
try:
    with EmulatorOwnership(ProcessIdentity(12345, 999999)):
        pass
except RuntimeError as exc:
    print(exc)
    raise SystemExit(23)
"""
        with EmulatorOwnership(self.identity):
            result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertIn("Another harness owns emulator", result.stdout)
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_crashed_owner_releases_os_lock_despite_remaining_file(self):
        script = """
import sys
from pokemon_harness.control.ownership import EmulatorOwnership, ProcessIdentity
with EmulatorOwnership(ProcessIdentity(12345, 999999)):
    print('READY', flush=True)
    sys.stdin.readline()
"""
        child = subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), "READY")
            with self.assertRaises(RuntimeError):
                with EmulatorOwnership(self.identity):
                    pass
            child.kill()
            child.wait(timeout=10)
            with EmulatorOwnership(self.identity):
                pass
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=10)

    def test_failed_model_preparation_closes_emulator_without_running(self):
        path = self.root / "config.json"
        write_json(path, {"run_dir": "ignored", "objective": "Test", "model": {"name": "test"},
                          "emulator": {"backend": "vbam"}})
        with patch("pokemon_harness.cli.emulator_for") as factory, \
                patch("pokemon_harness.cli.LMStudio") as model, patch("pokemon_harness.cli.run") as run:
            model.return_value.prepare.side_effect = RuntimeError("Load failed")
            self.assertEqual(main(["run", "--config", str(path), "--runs-dir", str(self.root / "runs")]), 1)
        factory.return_value.close.assert_called_once()
        run.assert_not_called()

    def test_conflicting_ownership_prevents_model_loading(self):
        path = self.root / "config.json"
        write_json(path, {"run_dir": "ignored", "objective": "Test", "model": {"name": "test"},
                          "emulator": {"backend": "vbam"}})
        with patch("pokemon_harness.cli.emulator_for", side_effect=RuntimeError("Already owned")), \
                patch("pokemon_harness.cli.LMStudio") as model:
            self.assertEqual(main(["run", "--config", str(path), "--runs-dir", str(self.root / "runs")]), 1)
        model.return_value.prepare.assert_not_called()

    def test_failed_key_release_still_releases_ownership(self):
        from contextlib import ExitStack
        emulator = WindowsDesktopEmulator.__new__(WindowsDesktopEmulator)
        emulator._resources = ExitStack()
        emulator._resources.enter_context(EmulatorOwnership(self.identity))
        emulator._release = Mock(side_effect=RuntimeError("Release failed"))
        with self.assertRaisesRegex(RuntimeError, "Release failed"):
            emulator.close()
        with EmulatorOwnership(self.identity):
            pass

    @unittest.skipUnless(os.name == "nt", "Windows target API")
    def test_failed_constructor_after_acquisition_releases_resources(self):
        class Target:
            identity = self.identity
            check = Mock(side_effect=RuntimeError("Target changed"))
            close = Mock()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        target = Target()
        config = {"buttons": {button: "A" for button in BUTTONS}}
        with patch("pokemon_harness.emulator.desktop.WindowsTarget", return_value=target), \
                patch.object(WindowsDesktopEmulator, "_configure_api"), \
                patch.object(WindowsDesktopEmulator, "_find_window", return_value=7):
            with self.assertRaisesRegex(RuntimeError, "Target changed"):
                WindowsDesktopEmulator(config)
        target.close.assert_called_once()
        with EmulatorOwnership(self.identity):
            pass

    @unittest.skipUnless(os.name == "nt", "Windows target API")
    def test_real_process_handle_detects_exit_and_window_reassignment(self):
        child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)
        self.addCleanup(lambda: child.poll() is None and child.kill())
        current_pid = [child.pid]

        def window_pid(hwnd, output):
            output._obj.value = current_pid[0]
            return 1

        user32 = SimpleNamespace(GetWindowThreadProcessId=Mock(side_effect=window_pid))
        try:
            with WindowsTarget(user32, 7) as target:
                self.assertEqual(target.identity.pid, child.pid)
                self.assertGreater(target.identity.created_at, 0)
                target.check()
                current_pid[0] = os.getpid()
                with self.assertRaisesRegex(RuntimeError, "target window changed"):
                    target.check()
                current_pid[0] = child.pid
                child.terminate()
                child.wait(timeout=10)
                with self.assertRaisesRegex(RuntimeError, "process exited"):
                    target.check()
            with self.assertRaisesRegex(RuntimeError, "target has been closed"):
                target.check()
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=10)

    def test_target_check_prevents_focus_and_input_when_identity_changes(self):
        emulator = WindowsDesktopEmulator.__new__(WindowsDesktopEmulator)
        emulator.ownership = Mock()
        emulator.target = Mock()
        emulator.target.check.side_effect = RuntimeError("Target changed")
        emulator.user32 = Mock()
        with self.assertRaisesRegex(RuntimeError, "Target changed"):
            emulator._focus()
        self.assertEqual(emulator.user32.mock_calls, [])
