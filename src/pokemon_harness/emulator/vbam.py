"""VisualBoyAdvance-M profile for the shared screenshot/keyboard backend."""

from .desktop import WindowsDesktopEmulator


class VBAM(WindowsDesktopEmulator):
    display_name = "VisualBoyAdvance-M"
    default_window_title = "VisualBoyAdvance-M"
    pause_menu_label = "Pause"
