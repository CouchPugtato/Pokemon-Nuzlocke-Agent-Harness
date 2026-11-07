"""Process-level emulator ownership shared by runs, captures, and input tests."""

import ctypes
from ctypes import wintypes
from dataclasses import dataclass

from .registry import registry_directory
from ..telemetry.recorder import RunLock


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    created_at: int


class EmulatorOwnership:
    def __init__(self, identity):
        if identity.pid < 1 or identity.created_at < 1:
            raise ValueError("Emulator process identity must contain a PID and creation time")
        self.identity = identity
        self.lock = None

    def __enter__(self):
        root = registry_directory().parent / "emulator-locks"
        root.mkdir(parents=True, exist_ok=True)
        lock = RunLock(root / f"{self.identity.pid}-{self.identity.created_at}.lock")
        try:
            lock.__enter__()
        except RuntimeError as exc:
            raise RuntimeError(f"Another harness owns emulator process {self.identity.pid}. "
                               "Stop that session before controlling this emulator") from exc
        self.lock = lock
        return self

    def check(self):
        if self.lock is None or self.lock.stream.closed:
            raise RuntimeError("Emulator ownership has been released; no further input is allowed")

    def __exit__(self, *args):
        if self.lock is not None:
            self.lock.__exit__(*args)
            self.lock = None


class WindowsTarget:
    """Retain the original process handle to detect exits and PID/window replacement.

    Only OS process identity is inspected, never emulator/game memory.
    """

    def __init__(self, user32, hwnd):
        self.user32, self.hwnd = user32, hwnd
        self.handle = None
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel = self.kernel32
        user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
        kernel.GetProcessTimes.restype = wintypes.BOOL
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        pid = self._window_pid()
        # PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE
        self.handle = kernel.OpenProcess(0x1000 | 0x100000, False, pid)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            created, exited, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
            if not kernel.GetProcessTimes(self.handle, ctypes.byref(created), ctypes.byref(exited),
                                          ctypes.byref(kernel_time), ctypes.byref(user_time)):
                raise ctypes.WinError(ctypes.get_last_error())
            self.identity = ProcessIdentity(pid, (created.dwHighDateTime << 32) | created.dwLowDateTime)
            self.check()
        except BaseException:
            self.close()
            raise

    def _window_pid(self):
        pid = wintypes.DWORD()
        if not self.user32.GetWindowThreadProcessId(self.hwnd, ctypes.byref(pid)) or not pid.value:
            raise RuntimeError("Emulator window no longer identifies a process; reconnect before continuing")
        return pid.value

    def check(self):
        if self.handle is None:
            raise RuntimeError("Emulator target has been closed")
        status = self.kernel32.WaitForSingleObject(self.handle, 0)
        if status == 0xFFFFFFFF:
            raise ctypes.WinError(ctypes.get_last_error())
        if status != 258 or self._window_pid() != self.identity.pid:  # WAIT_TIMEOUT means still running.
            raise RuntimeError("Emulator process exited or target window changed; reconnect before continuing")

    def close(self):
        if self.handle is not None:
            handle, self.handle = self.handle, None
            if not self.kernel32.CloseHandle(handle):
                raise ctypes.WinError(ctypes.get_last_error())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
