"""Shared Windows desktop backend. Screenshots + keyboard only; no game memory.

Frame counts are translated into wall-clock input durations. The game continues
running during capture and inference; this backend never changes its pause state.
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
        self.min_hold_frames = config.get("min_hold_frames", 6)
        if type(self.min_hold_frames) is not int or not 1 <= self.min_hold_frames <= 120:
            raise ValueError("emulator.min_hold_frames must be an integer between 1 and 120")
        self.input_mode = config.get("input_mode", "scancode")
        if self.input_mode not in {"scancode", "virtual_key"}:
            raise ValueError("emulator.input_mode must be scancode or virtual_key")
        self.held = set()
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._configure_api()
        self.bindings = config["buttons"]
        from ..agent.schemas import BUTTONS
        if set(self.bindings) != set(BUTTONS):
            raise ValueError("Configure a keyboard binding for every GBA button")
        for binding in self.bindings.values():
            self._combo(binding)
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
        u.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
        u.SetCursorPos.restype = wintypes.BOOL
        u.WindowFromPoint.argtypes = [wintypes.POINT]
        u.WindowFromPoint.restype = wintypes.HWND
        u.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        u.GetAncestor.restype = wintypes.HWND
        u.IsWindowEnabled.argtypes = [wintypes.HWND]
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
        u.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
        u.MapVirtualKeyW.restype = wintypes.UINT

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
        if not self.user32.IsWindowEnabled(self.hwnd):
            raise RuntimeError("Close the emulator's settings dialog before running controller input")

    def _focus(self):
        if self.user32.GetForegroundWindow() != self.hwnd:
            self.user32.SetForegroundWindow(self.hwnd)
            time.sleep(0.15)
        self._check_focus()
        if self.config.get("focus_game_area", False):
            self._focus_game_area()

    def _focus_game_area(self):
        """A foreground frame can still have its toolbar focused rather than GLCanvas."""
        self._check_focus()
        rect = wintypes.RECT()
        point = wintypes.POINT()
        if not self.user32.GetClientRect(self.hwnd, ctypes.byref(rect)):
            raise ctypes.WinError(ctypes.get_last_error())
        point.x, point.y = rect.right // 2, rect.bottom // 2
        if not self.user32.ClientToScreen(self.hwnd, ctypes.byref(point)):
            raise ctypes.WinError(ctypes.get_last_error())
        under_cursor = self.user32.WindowFromPoint(point)
        if self.user32.GetAncestor(under_cursor, 2) != self.hwnd:
            raise RuntimeError("Game display is covered by another window; cannot focus it safely")
        if not self.user32.SetCursorPos(point.x, point.y):
            raise ctypes.WinError(ctypes.get_last_error())
        self._check_focus()
        down, up = self.Input(type=0), self.Input(type=0)
        down.mi.dwFlags, up.mi.dwFlags = 2, 4
        try:
            if self.user32.SendInput(1, ctypes.byref(down), ctypes.sizeof(down)) != 1:
                raise RuntimeError("Windows rejected game-panel focus click")
        finally:
            if self.user32.SendInput(1, ctypes.byref(up), ctypes.sizeof(up)) != 1:
                raise RuntimeError("Windows rejected mouse-button release")
        time.sleep(0.1)

    def _keyboard_input(self, key, up=False):
        # Include real scan codes, especially for standalone Shift (GBA Start).
        # Scan-code input follows the physical-key path instead of relying on a
        # zero-scan-code virtual key event being translated by the frontend.
        scan = self.user32.MapVirtualKeyW(key, 4)  # MAPVK_VK_TO_VSC_EX
        if not scan:
            raise ValueError(f"Windows could not map virtual key {key:#x} to a scan code")
        flags = (2 if up else 0) | (1 if scan & 0xFF00 == 0xE000 else 0)
        physical = self.input_mode == "scancode"
        if physical:
            flags |= 8  # KEYEVENTF_SCANCODE
        value = self.Input(type=1)
        value.ki = self.KeyboardInput(wVk=0 if physical else key, wScan=scan & 0xFF, dwFlags=flags)
        return value

    def _key(self, key, up=False):
        if not up:
            self._check_focus()
        value = self._keyboard_input(key, up)
        if self.user32.SendInput(1, ctypes.byref(value), ctypes.sizeof(value)) != 1:
            raise RuntimeError(f"Windows rejected keyboard input (error {ctypes.get_last_error()}); "
                               "check focus and that the emulator and harness have matching privilege levels")
        if up:
            self.held.discard(key)
        else:
            self.held.add(key)

    def _release(self):
        failures = []
        # Release ordinary keys before modifiers used in compound bindings.
        for key in sorted(self.held, key=lambda k: k in {0x10, 0x11, 0x12}):
            try:
                self._key(key, up=True)
            except (OSError, RuntimeError, ValueError) as exc:
                failures.append(str(exc))
        if failures:
            raise RuntimeError("Failed releasing keys: " + "; ".join(failures))

    def capture(self):
        from PIL import ImageGrab

        self._focus()
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
        return Capture(output.getvalue(), "live screenshot; game continues during inference; "
                       "inputs use approximate wall-clock durations")

    def execute(self, action):
        self._check_focus()
        if self.config.get("focus_game_area", False):
            self._focus_game_area()
        effective = dict(action)
        if action["kind"] == "press":
            effective["hold_frames"] = max(action["hold_frames"], self.min_hold_frames)
        started = time.monotonic()
        try:
            for button in action["buttons"]:
                for key in self._combo(self.bindings[button]):
                    if key not in self.held:
                        self._key(key)
            deadline = time.monotonic() + effective["hold_frames"] / self.fps
            while time.monotonic() < deadline:
                self._check_focus()
                time.sleep(min(0.01, max(0, deadline - time.monotonic())))
            self._release()
            deadline = time.monotonic() + action["settle_frames"] / self.fps
            while time.monotonic() < deadline:
                self._check_focus()
                time.sleep(min(0.01, max(0, deadline - time.monotonic())))
        finally:
            self._release()
        return {"requested_action": dict(action), "effective_action": effective,
                "input_mode": self.input_mode, "elapsed_seconds": time.monotonic() - started,
                "game_timing": "continuous",
                "verification": "Input submitted; game response requires screenshot inspection"}

    def close(self):
        self._release()
