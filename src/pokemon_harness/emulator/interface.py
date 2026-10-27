from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Capture:
    image: bytes
    timing: str


class Emulator(Protocol):
    def capture(self) -> Capture: ...
    def execute(self, action: dict) -> None: ...
    def close(self) -> None: ...


class ReplayEmulator:
    """Consume a sequence of PNGs without touching a real emulator."""

    def __init__(self, images: list[Path]):
        if not images:
            raise ValueError("Replay needs at least one PNG")
        self.images = images
        self.index = 0
        self.actions = []

    def capture(self):
        return Capture(self.images[self.index].read_bytes(), "replay; no game execution")

    def execute(self, action):
        self.actions.append(action)
        self.index = min(self.index + 1, len(self.images) - 1)

    def close(self):
        pass
