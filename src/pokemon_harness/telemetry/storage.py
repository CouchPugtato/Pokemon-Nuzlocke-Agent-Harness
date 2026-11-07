"""capacity checks before input and space for emergency records"""

import errno
import os
import shutil
from pathlib import Path


MIB = 1024 * 1024
DEFAULTS = {"max_run_mb": 10240, "max_backup_mb": 20480, "min_free_mb": 512,
            "action_headroom_mb": 8, "emergency_reserve_mb": 1}


def storage_settings(config):
    settings = config.get("storage", {})
    if not isinstance(settings, dict):
        raise ValueError("storage must be an object")
    settings = dict(DEFAULTS, **settings)
    for key in DEFAULTS:
        if type(settings[key]) is not int or settings[key] < (0 if key == "min_free_mb" else 1):
            raise ValueError(f"storage.{key} must be an integer with a valid positive capacity")
    if settings["max_run_mb"] <= settings["action_headroom_mb"] + settings["emergency_reserve_mb"]:
        raise ValueError("storage.max_run_mb must exceed action headroom plus emergency reserve")
    if settings["max_backup_mb"] <= settings["action_headroom_mb"] + settings["emergency_reserve_mb"]:
        raise ValueError("storage.max_backup_mb must exceed action headroom plus emergency reserve")
    return settings


class StorageLimitError(OSError):
    def __init__(self, message):
        super().__init__(errno.ENOSPC, message)


class StorageGuard:
    def __init__(self, config):
        self.directory = Path(config["run_dir"]).resolve()
        self.settings = storage_settings(config)
        self.reserve = self.directory / ".emergency-reserve"
        self.reserve_owned = False

    def _run_bytes(self):
        total = 0
        pending = [self.directory] if self.directory.exists() else []
        while pending:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    stat = entry.stat(follow_symlinks=False)
                    if entry.is_symlink() or getattr(stat, "st_file_attributes", 0) & 0x400:
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    else:
                        total += stat.st_size
        return total

    def check(self, additional_bytes=None):
        headroom = self.settings["action_headroom_mb"] * MIB
        needed = headroom if additional_bytes is None else max(headroom, additional_bytes)
        existing = self.directory
        while not existing.exists():
            existing = existing.parent
        free = shutil.disk_usage(existing).free
        used = self._run_bytes()
        maximum = self.settings["max_run_mb"] * MIB
        if used + needed > maximum:
            raise StorageLimitError(f"Run storage limit reached: {used} bytes used, {needed} bytes headroom required, "
                                    f"limit {maximum}. Free space or increase storage.max_run_mb before resuming")
        if free < self.settings["min_free_mb"] * MIB + needed:
            raise StorageLimitError(f"Insufficient free disk space: {free} bytes available. "
                                    "Free space before resuming; no further game input is allowed")
        return {"run_bytes": used, "free_bytes": free, "max_run_bytes": maximum}

    def start(self):
        amount = self.settings["emergency_reserve_mb"] * MIB
        self.check(amount + self.settings["action_headroom_mb"] * MIB)
        if self.reserve.exists() or self.reserve.is_symlink():
            if self.reserve.is_symlink() or not self.reserve.is_file() or self.reserve.stat().st_size != amount:
                raise StorageLimitError("Unexpected emergency reserve file; inspect it before resuming")
            self.reserve_owned = True
            return
        # allocating real bytes instead of a sparse placeholder
        with self.reserve.open("xb") as stream:
            self.reserve_owned = True
            chunk = b"\0" * min(amount, 65536)
            remaining = amount
            while remaining:
                written = stream.write(chunk[:remaining])
                if not written:
                    raise StorageLimitError("Emergency reserve allocation did not complete")
                remaining -= written
            stream.flush()
            os.fsync(stream.fileno())

    def release_reserve(self):
        if self.reserve_owned:
            self.reserve.unlink(missing_ok=True)
            self.reserve_owned = False
