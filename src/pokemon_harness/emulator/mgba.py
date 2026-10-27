"""mGBA profile for the shared screenshot/keyboard backend."""

from .desktop import WindowsDesktopEmulator


class MGBA(WindowsDesktopEmulator):
    display_name = "mGBA"
    default_window_title = "mGBA"
