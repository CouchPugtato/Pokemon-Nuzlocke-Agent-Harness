"""Shared Windows desktop backend. Screenshots + keyboard only; no game memory.

Frame counts are translated into wall-clock input durations. Configure a dedicated
pause toggle in the emulator; the frontend remains responsive while inference runs.
"""

import ctypes
import io
import os
import time
from ctypes import wintypes

from .interface import Capture


class WindowsDesktopEmulator:
    display_name = "Emulator"
    default_window_title = "Emulator"

    def __init__(self, config):
        if os.name != "nt":
            raise RuntimeError(f"The {self.display_name} desktop backend currently supports Windows only")
        self.config = config
        self.fps = config.get("fps", 60)
        if not isinstance(self.fps, (float, int)) or not 1 <= self.fps <= 240:
            raise ValueError("emulator.fps must be between 1 and 240")
        if config.get("initial_state") not in {"paused", "running"}:
            raise ValueError("Set emulator.initial_state to the actual paused/running state")
        self.running = config["initial_state"] == "running"
        self.held = set()
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._configure_api()
        self.bindings = config["buttons"]
        from ..agent.schemas import BUTTONS
        if set(self.bindings) != set(BUTTONS):
            raise ValueError("Configure a keyboard binding for every GBA button")
        for binding in list(self.bindings.values()) + [config["pause_key"]]:
            self._combo(binding)
        pause = set(self._combo(config["pause_key"]))
        if any(pause.intersection(self._combo(binding)) for binding in self.bindings.values()):
            raise ValueError("The pause shortcut must not overlap a game button binding")
        self.hwnd = self._find_window(config.get("window_title", self.default_window_title))

    def _configure_api(self):
        u = self.user32
        self.enum_callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        u.EnumWindows.argtypes = [self.enum_callback, wintypes.LPARAM]
        u.EnumWindows.restype = wintypes.BOOL
        u.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        u.GetWindowTextLengthW.restype = ctypes.c_int
        u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        u.IsWindowVisible.argtypes = [wintypes.HWND]
        u.IsIconic.argtypes = [wintypes.HWND]
        u.IsWindow.argtypes = [wintypes.HWND]
        u.GetForegroundWindow.restype = wintypes.HWND
        u.SetForegroundWindow.argtypes = [wintypes.HWND]
        u.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        u.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        try:
            u.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
            u.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except AttributeError:
            u.SetProcessDPIAware()

        class MouseInput(ctypes.Structure):
            _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                        ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                        ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]

        class KeyboardInput(ctypes.Structure):
            _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                        ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                        ("dwExtraInfo", ctypes.c_size_t)]

        class HardwareInput(ctypes.Structure):
            _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]

        class InputUnion(ctypes.Union):
            _fields_ = [("ki", KeyboardInput), ("mi", MouseInput), ("hi", HardwareInput)]

        class Input(ctypes.Structure):
            _anonymous_ = ("value",)
            _fields_ = [("type", wintypes.DWORD), ("value", InputUnion)]

        self.Input = Input
        self.KeyboardInput = KeyboardInput
        u.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(Input), ctypes.c_int]
        u.SendInput.restype = wintypes.UINT

    @staticmethod
    def _combo(binding):
        special = {"CTRL": 0x11, "SHIFT": 0x10, "ALT": 0x12, "ENTER": 0x0D,
                   "SPACE": 0x20, "TAB": 0x09, "BACKSPACE": 0x08, "ESC": 0x1B,
                   "CAPSLOCK": 0x14, "CAPITAL": 0x14,
                   "UP": 0x26, "DOWN": 0x28, "LEFT": 0x25, "RIGHT": 0x27}
        keys = []
        for part in binding.upper().split("+"):
            if part in special:
                keys.append(special[part])
            elif len(part) == 1 and part.isascii() and part.isalnum():
                keys.append(ord(part))
            elif part.startswith("F") and part[1:].isdigit() and 1 <= int(part[1:]) <= 24:
                keys.append(0x70 + int(part[1:]) - 1)
            else:
                raise ValueError(f"Unsupported keyboard key: {part}")
        return keys

    def _find_window(self, title):
        matches = []

        @self.enum_callback
        def visit(hwnd, _):
            if self.user32.IsWindowVisible(hwnd):
                text = ctypes.create_unicode_buffer(self.user32.GetWindowTextLengthW(hwnd) + 1)
                self.user32.GetWindowTextW(hwnd, text, len(text))
                if title.casefold() in text.value.casefold():
                    matches.append(hwnd)
            return True

        self.user32.EnumWindows(visit, 0)
        if len(matches) != 1:
            raise RuntimeError(f"Expected one window matching {title!r}, found {len(matches)}")
        return matches[0]

    def _check_focus(self):
        if not self.user32.IsWindow(self.hwnd) or self.user32.IsIconic(self.hwnd):
            raise RuntimeError(f"{self.display_name} window is closed or minimized")
        if self.user32.GetForegroundWindow() != self.hwnd:
            raise RuntimeError(f"{self.display_name} lost focus; no further key-down events will be sent")

    def _focus(self):
        if self.user32.GetForegroundWindow() != self.hwnd:
            self.user32.SetForegroundWindow(self.hwnd)
            time.sleep(0.15)
        self._check_focus()

    def _key(self, key, up=False):
        if not up:
            self._check_focus()
        flags = (2 if up else 0) | (1 if key in {0x25, 0x26, 0x27, 0x28} else 0)
        value = self.Input(type=1)
        value.ki = self.KeyboardInput(wVk=key, dwFlags=flags)
        if self.user32.SendInput(1, ctypes.byref(value), ctypes.sizeof(value)) != 1:
            raise ctypes.WinError(ctypes.get_last_error())
        if up:
            self.held.discard(key)
        else:
            self.held.add(key)

    def _release(self):
        failures = []
        for key in list(self.held):
            try:
                self._key(key, up=True)
            except OSError as exc:
                failures.append(str(exc))
        if failures:
            raise RuntimeError("Failed releasing keys: " + "; ".join(failures))

    def _tap(self, binding):
        try:
            for key in self._combo(binding):
                self._key(key)
            time.sleep(0.06)
        finally:
            self._release()
        time.sleep(0.08)

    def _pause(self):
        if self.running:
            self._tap(self.config["pause_key"])
            self.running = False

    def capture(self):
        from PIL import ImageGrab

        self._focus()
        self._pause()
        rect = wintypes.RECT()
        point = wintypes.POINT(0, 0)
        if not self.user32.GetClientRect(self.hwnd, ctypes.byref(rect)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not self.user32.ClientToScreen(self.hwnd, ctypes.byref(point)):
            raise ctypes.WinError(ctypes.get_last_error())
        crop = self.config.get("crop")
        if crop is None:
            left, top, width, height = 0, 0, rect.right, rect.bottom
        else:
            if len(crop) != 4 or any(type(x) is not int for x in crop):
                raise ValueError("crop must be [left, top, width, height] in client pixels")
            left, top, width, height = crop
        if min(left, top) < 0 or min(width, height) <= 0 or left + width > rect.right or top + height > rect.bottom:
            raise ValueError(f"Screenshot crop lies outside the {self.display_name} client area")
        bounds = (point.x + left, point.y + top, point.x + left + width, point.y + top + height)
        image = ImageGrab.grab(bbox=bounds, all_screens=True)
        self._check_focus()
        output = io.BytesIO()
        image.save(output, format="PNG")
        return Capture(output.getvalue(), "paused screenshot; inputs use approximate wall-clock frame durations")

    def execute(self, action):
        self._check_focus()
        if self.running:
            raise RuntimeError("Expected paused emulator before action")
        # Resume before pressing: controls intended for the game must not overlap
        # the frontend pause shortcut. Timing includes frontend dispatch overhead.
        self._tap(self.config["pause_key"])
        self.running = True
        try:
            for button in action["buttons"]:
                for key in self._combo(self.bindings[button]):
                    if key not in self.held:
                        self._key(key)
            deadline = time.monotonic() + action["hold_frames"] / self.fps
            while time.monotonic() < deadline:
                self._check_focus()
                time.sleep(min(0.01, max(0, deadline - time.monotonic())))
            self._release()
            deadline = time.monotonic() + action["settle_frames"] / self.fps
            while time.monotonic() < deadline:
                self._check_focus()
                time.sleep(min(0.01, max(0, deadline - time.monotonic())))
        finally:
            try:
                self._release()
            finally:
                self._pause()

    def close(self):
        try:
            self._release()
        finally:
            self._pause()
