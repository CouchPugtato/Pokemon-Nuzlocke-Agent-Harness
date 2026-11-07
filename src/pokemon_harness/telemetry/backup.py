"""consistent local snapshots of harness memory and evidence"""

import hashlib
import json
import os
import shutil
import sqlite3
import sys
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from ..config import load_config
from .cleanup import cleanup_operations
from .recorder import RunLock, write_json
from .storage import StorageGuard


EXCLUDED = {"memory.sqlite", "memory.sqlite-wal", "memory.sqlite-shm", "config.json",
            ".lock", ".stop", ".session.json", ".emergency-reserve", "manifest.json"}


def _linked(path):
    return path.is_symlink() or bool(getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0) & 0x400)


def _child(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Backup paths must be relative portable paths")
    parts = PurePosixPath(relative).parts
    if (str(PurePosixPath(relative)) != relative or relative.startswith("/")
            or any(part in {".", ".."} or ":" in part for part in parts)):
        raise ValueError("Unsafe path in backup")
    path = root
    for part in parts:
        path = path / part
        if path.exists() and _linked(path):
            raise ValueError("Linked files are not allowed in backups")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Backup path escapes its directory")
    return path


def _fingerprint(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return {"size": size, "sha256": digest.hexdigest()}


def _files(root):
    pending = [root]
    while pending:
        directory = pending.pop()
        for path in sorted(directory.iterdir()):
            if _linked(path):
                raise ValueError(f"Linked archive entry cannot be backed up: {path}")
            if path.is_dir():
                pending.append(path)
            elif path.is_file():
                yield path
            else:
                raise ValueError(f"Unsupported archive entry: {path}")


def _read_database(path):
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def verify_backup(directory):
    root = Path(directory).resolve()
    manifest_path = _child(root, "manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("scope") != "harness_state_only":
        raise ValueError("Unsupported backup manifest")
    files = manifest.get("files")
    if not isinstance(files, dict) or not {"memory.sqlite", "config.json"}.issubset(files):
        raise ValueError("Backup must include the database and config")
    actual_files = {path.relative_to(root).as_posix() for path in _files(root)}
    if actual_files != set(files) | {"manifest.json"}:
        raise ValueError("Backup contains missing or unlisted files")
    seen = set()
    for relative, expected in files.items():
        folded = relative.casefold() if os.name == "nt" else relative
        if folded in seen:
            raise ValueError("Conflicting backup paths")
        seen.add(folded)
        path = _child(root, relative)
        if not path.is_file() or _fingerprint(path) != expected:
            raise ValueError(f"Missing or corrupted backup file: {relative}")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    with closing(_read_database(root / "memory.sqlite")) as database:
        if database.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Backup database integrity check failed")
        if database.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("Backup database has broken references")
        row = database.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        identity = {"objective": config["objective"], "rules": config["rules"]}
        if not row or json.loads(row[0]) != identity or manifest.get("identity") != identity:
            raise ValueError("Backup run identity does not match its config")
        observations = database.execute("SELECT id,image_path,image_hash FROM observations").fetchall()
        for observation in observations:
            relative = observation["image_path"].replace("\\", "/")
            _child(root, relative)
            if relative not in files or files[relative]["sha256"] != observation["image_hash"]:
                raise ValueError(f"Missing or mismatched evidence for observation {observation['id']}")
        if len(observations) != manifest.get("observation_count"):
            raise ValueError("Backup observation count does not match its manifest")
    return {"backup_dir": str(root), "files": len(files), "observations": len(observations),
            "scope": "harness_state_only", "verified": True}


def _discard_stage(stage, parent):
    # checking the exact staging target before recursive cleanup
    if (stage.parent.resolve() != parent.resolve() or stage.resolve().parent != parent.resolve()
            or not stage.name.startswith(".backup-") or stage.is_symlink()):
        raise ValueError("Refusing cleanup outside the backup staging directory")
    if stage.exists():
        shutil.rmtree(stage)


def create_backup(run_dir, backup_dir=None):
    source = Path(run_dir).resolve()
    if not (source / "memory.sqlite").is_file() or not (source / "config.json").is_file():
        raise ValueError("No saved run at the requested backup source")
    root = Path(backup_dir).resolve() if backup_dir is not None else source.parent / ".backups"
    if root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Backup storage must be separate from the run directory")
    stage = root / (".backup-" + uuid.uuid4().hex)
    with RunLock(source / ".lock"):
        config = load_config(source / "config.json")
        settings = dict(config["storage"], max_run_mb=config["storage"]["max_backup_mb"])
        guard = StorageGuard({"run_dir": str(root), "storage": settings})
        paths = [path for path in _files(source)
                 if path.relative_to(source).as_posix() not in EXCLUDED and not path.name.endswith(".tmp")]
        with closing(_read_database(source / "memory.sqlite")) as database:
            pages = database.execute("PRAGMA page_count").fetchone()[0]
            page_size = database.execute("PRAGMA page_size").fetchone()[0]
            estimate = sum(path.stat().st_size for path in paths) + pages * page_size + 1024 * 1024 + len(paths) * 500
            guard.check(estimate)
            root.mkdir(parents=True, exist_ok=True)
            with RunLock(root / ".backup.lock"):
                guard.check(estimate)
                stage.mkdir()
                published = False
                try:
                    # using the database backup api to include committed journal data
                    with closing(sqlite3.connect(stage / "memory.sqlite")) as destination:
                        database.backup(destination)
                        destination.execute("PRAGMA journal_mode=DELETE")
                    with (stage / "memory.sqlite").open("r+b") as stream:
                        os.fsync(stream.fileno())
                    portable_config = dict(config, run_dir=".")
                    write_json(stage / "config.json", portable_config)
                    for path in paths:
                        relative = path.relative_to(source).as_posix()
                        target = _child(stage, relative)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        guard.check(path.stat().st_size)
                        with path.open("rb") as incoming, target.open("xb") as outgoing:
                            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
                            outgoing.flush()
                            os.fsync(outgoing.fileno())
                    files = {path.relative_to(stage).as_posix(): _fingerprint(path) for path in _files(stage)}
                    with closing(_read_database(stage / "memory.sqlite")) as snapshot:
                        count = snapshot.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
                    stamp = datetime.now(timezone.utc)
                    manifest = {"schema_version": 1, "scope": "harness_state_only", "emulator_save_included": False,
                                "created_at": stamp.isoformat(), "source_run_dir": str(source),
                                "identity": {"objective": config["objective"], "rules": config["rules"]},
                                "observation_count": count, "files": files}
                    guard.check(len(json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")))
                    write_json(stage / "manifest.json", manifest)
                    verify_backup(stage)
                    guard.check()
                    final = root / (source.name + "-" + stamp.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8])
                    stage.rename(final)
                    published = True
                    return final
                finally:
                    if not published:
                        cleanup_operations([("partial backup", lambda: _discard_stage(stage, root))], sys.exc_info()[1])
