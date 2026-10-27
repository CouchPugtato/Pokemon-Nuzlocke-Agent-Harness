import hashlib
import json
import os
import uuid
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_capture(run_dir, capture):
    if not capture.image.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Emulator must return a PNG screenshot")
    path = run_dir / "screenshots" / (uuid.uuid4().hex + ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(capture.image)
    return path.relative_to(run_dir), hashlib.sha256(capture.image).hexdigest()


class RunLock:
    """OS lock releases on crash; stale lock files do not prevent recovery."""

    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        self.stream = self.path.open("a+b")
        try:
            # Reading an already locked byte raises PermissionError on Windows.
            # File size can be checked without accessing that byte.
            if os.fstat(self.stream.fileno()).st_size == 0:
                self.stream.write(b"0")
                self.stream.flush()
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            raise RuntimeError("Another process owns this run directory") from exc
        return self

    def __exit__(self, *args):
        self.stream.close()
