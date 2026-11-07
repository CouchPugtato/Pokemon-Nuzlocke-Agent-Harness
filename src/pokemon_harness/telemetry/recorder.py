import hashlib
import json
import os
import uuid
import sys
import errno
from pathlib import Path
from .cleanup import cleanup_operations


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        cleanup_operations([("temporary json", lambda: temporary.unlink(missing_ok=True))], sys.exc_info()[1])


def save_capture(run_dir, capture):
    if not capture.image.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Emulator must return a PNG screenshot")
    path = run_dir / "screenshots" / (uuid.uuid4().hex + ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(capture.image)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException as exc:
        cleanup_operations([("partial screenshot", lambda: path.unlink(missing_ok=True))], exc)
        raise
    return path.relative_to(run_dir), hashlib.sha256(capture.image).hexdigest()


class RunLock:
    """automatic lock release after a crash"""

    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        self.stream = self.path.open("a+b")
        try:
            # checking size without reading a locked byte
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
            cleanup_operations([("lock stream", self.stream.close)], exc)
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise RuntimeError("Another process owns this run directory") from exc
            raise
        return self

    def __exit__(self, *args):
        self.stream.close()
