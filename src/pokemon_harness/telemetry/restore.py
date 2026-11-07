"""verified restoration into a new run with mandatory review"""

import hashlib
import json
import os
import shutil
import sys
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from ..config import load_config
from ..memory.database import Memory
from .backup import _child, _discard_stage, _fingerprint, _read_database, verify_backup
from .cleanup import cleanup_operations
from .recorder import RunLock, write_json
from .storage import StorageGuard


def _verify_restored(directory, identity):
    with closing(_read_database(directory / "memory.sqlite")) as database:
        if database.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Restored database integrity check failed")
        if database.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("Restored database has broken references")
        row = database.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if not row or json.loads(row[0]) != identity:
            raise ValueError("Restored run identity changed")
        for observation in database.execute("SELECT image_path,image_hash FROM observations"):
            path = _child(directory, observation["image_path"].replace("\\", "/"))
            if _fingerprint(path)["sha256"] != observation["image_hash"]:
                raise ValueError("Restored evidence does not match the database")


def restore_backup(backup_dir, run_dir=None, runs_dir=Path("runs")):
    source = Path(backup_dir).resolve()
    verify_backup(source)
    manifest_bytes = (source / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    config = load_config(source / "config.json")
    restore_id = uuid.uuid4().hex
    stamp = datetime.now(timezone.utc)
    name = "restored-" + stamp.strftime("%Y%m%d-%H%M%S-") + restore_id[:8]
    target = Path(run_dir).resolve() if run_dir is not None else (Path(runs_dir) / name).resolve()
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("Restore destination must be separate from its backup")
    if target.exists() or target.is_symlink():
        raise ValueError("Restore requires a new destination and never overwrites an existing directory")
    stage = target.parent / (".restore-" + restore_id)
    guard = StorageGuard(dict(config, run_dir=str(stage)))
    estimate = sum(item["size"] for item in manifest["files"].values()) + len(manifest_bytes) + 1024 * 1024
    guard.check(estimate)
    target.parent.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(os.path.normcase(str(target)).encode("utf-8")).hexdigest()
    with RunLock(target.parent / (".restore-" + key + ".lock")):
        if target.exists() or target.is_symlink():
            raise ValueError("Restore destination was created by another process")
        stage.mkdir()
        published = False
        try:
            # copying only files covered by the verified manifest
            for relative, fingerprint in manifest["files"].items():
                incoming = _child(source, relative)
                outgoing = _child(stage, relative)
                outgoing.parent.mkdir(parents=True, exist_ok=True)
                guard.check(fingerprint["size"])
                with incoming.open("rb") as reader, outgoing.open("xb") as writer:
                    shutil.copyfileobj(reader, writer, length=1024 * 1024)
                    writer.flush()
                    os.fsync(writer.fileno())
                if _fingerprint(outgoing) != fingerprint:
                    raise ValueError(f"Backup changed while restoring: {relative}")
            (stage / "manifest.json").write_bytes(manifest_bytes)
            verify_backup(stage)
            (stage / "manifest.json").unlink()
            archive = stage / "restores" / restore_id
            archive.mkdir(parents=True)
            write_json(archive / "manifest.json", manifest)
            restored_config = dict(config, run_dir=str(target))
            write_json(stage / "config.json", restored_config)
            info = {"restore_id": restore_id, "backup_dir": str(source), "run_dir": str(target),
                    "restored_at": stamp.isoformat(), "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                    "emulator_save_included": False, "game_state_verified": False}
            with closing(Memory(stage / "memory.sqlite")) as memory:
                memory.initialize(restored_config)
                recovered = memory.recover()
                if recovered:
                    memory.note("recovery", recovered)
                previous = memory.previous_observation()
                evidence = [previous["id"]] if previous else []
                details = dict(info, kind="restore_reconciliation",
                               missing_fact="Whether the current emulator state matches the restored history",
                               required_checks=["location", "party", "badges", "encounter and death history"])
                case_id = memory.escalations.open("restore:" + restore_id,
                    "Restored harness memory has no matching emulator save verification", evidence, details)
                memory.note("restore", dict(info, case_id=case_id))
                write_json(stage / ".handoff.json", {"reason": "backup_restored", "session_id": None,
                    "message": "Restored agent state may not match the emulator; do not replay old actions",
                    "last_observation_id": previous["id"] if previous else None, "restore_id": restore_id})
                export = memory.export()
                guard.check(len(json.dumps(export, ensure_ascii=False, indent=2).encode("utf-8")))
                write_json(stage / "memory-export.json", export)
            with (stage / "memory.sqlite").open("r+b") as stream:
                os.fsync(stream.fileno())
            _verify_restored(stage, manifest["identity"])
            guard.check()
            if target.exists() or target.is_symlink():
                raise ValueError("Restore destination already exists")
            # publishing only after verification and the review gate are durable
            stage.rename(target)
            published = True
            return dict(info, case_id=case_id, review_required=True)
        finally:
            if not published:
                cleanup_operations([("partial restore", lambda: _discard_stage(stage, target.parent, ".restore-"))],
                                   sys.exc_info()[1])
