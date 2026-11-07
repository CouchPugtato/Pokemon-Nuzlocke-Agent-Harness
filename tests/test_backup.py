import contextlib
import errno
import hashlib
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pokemon_harness.cli import main
from pokemon_harness.config import load_config
from pokemon_harness.demo import DemoAgent, create_images
from pokemon_harness.emulator.interface import ReplayEmulator
from pokemon_harness.memory.database import Memory
from pokemon_harness.runner import run
from pokemon_harness.telemetry.backup import _child, _discard_stage, _fingerprint, create_backup, verify_backup
from pokemon_harness.telemetry.recorder import RunLock, write_json
from pokemon_harness.telemetry.storage import MIB, StorageLimitError


class BackupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        path = self.root / "config.json"
        write_json(path, {"run_dir": "run", "objective": "Test", "rules": ["Test rule"],
                          "model": {"name": "test"}, "emulator": {"backend": "replay"},
                          "storage": {"max_run_mb": 32, "max_backup_mb": 32, "min_free_mb": 0,
                                      "action_headroom_mb": 1, "emergency_reserve_mb": 1}})
        self.config = load_config(path)
        self.source = Path(self.config["run_dir"])
        self.images = create_images(self.root / "fixtures")
        run(self.config, ReplayEmulator(self.images), DemoAgent(), turns=2)
        self.backups = self.root / "backups"

    def read_manifest(self, backup):
        return json.loads((backup / "manifest.json").read_text())

    def test_backup_is_portable_verified_and_excludes_live_control_files(self):
        write_json(self.source / ".session.json", {"session_id": "old"})
        write_json(self.source / ".stop", {"session_id": "old"})
        backup = create_backup(self.source, self.backups)
        result = verify_backup(backup)
        self.assertTrue(result["verified"])
        self.assertEqual(result["observations"], 3)
        self.assertEqual(json.loads((backup / "config.json").read_text())["run_dir"], ".")
        manifest = self.read_manifest(backup)
        self.assertFalse(manifest["emulator_save_included"])
        for name in (".lock", ".stop", ".session.json", "memory.sqlite-wal", ".emergency-reserve"):
            self.assertNotIn(name, manifest["files"])
            self.assertFalse((backup / name).exists())
        self.assertTrue((self.source / ".stop").is_file())

    def test_committed_wal_data_and_case_history_are_included(self):
        with contextlib.closing(Memory(self.source / "memory.sqlite")) as memory:
            payload = self.images[0].read_bytes()
            path = self.source / "screenshots" / "wal-evidence.png"
            path.write_bytes(payload)
            observation = memory.observe("screenshots/wal-evidence.png", hashlib.sha256(payload).hexdigest(), "review")
            case = memory.escalations.open("review", "Uncertain history", [observation])
            memory.escalations.resolve(case, "unverifiable", "Missing historical evidence", allow_continue=True)
            self.assertGreater((self.source / "memory.sqlite-wal").stat().st_size, 0)
            backup = create_backup(self.source, self.backups)
        with contextlib.closing(sqlite3.connect((backup / "memory.sqlite").as_uri() + "?mode=ro", uri=True)) as snapshot:
            self.assertEqual(snapshot.execute("SELECT COUNT(*) FROM observations").fetchone()[0], 4)
            self.assertEqual(snapshot.execute("SELECT outcome FROM case_resolutions").fetchone()[0], "unverifiable")
        self.assertTrue(verify_backup(backup)["verified"])

    def test_active_run_is_refused_without_creating_a_snapshot(self):
        with RunLock(self.source / ".lock"):
            with self.assertRaisesRegex(RuntimeError, "Another process"):
                create_backup(self.source, self.backups)
        self.assertFalse(self.backups.exists())

    def test_manifest_detects_changed_evidence(self):
        backup = create_backup(self.source, self.backups)
        screenshot = next((backup / "screenshots").glob("*.png"))
        screenshot.write_bytes(b"damaged")
        with self.assertRaisesRegex(ValueError, "corrupted backup file"):
            verify_backup(backup)

    def test_database_evidence_hash_is_checked_even_if_manifest_is_rewritten(self):
        backup = create_backup(self.source, self.backups)
        screenshot = next((backup / "screenshots").glob("*.png"))
        screenshot.write_bytes(b"different evidence")
        manifest = self.read_manifest(backup)
        manifest["files"][screenshot.relative_to(backup).as_posix()] = _fingerprint(screenshot)
        write_json(backup / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "mismatched evidence"):
            verify_backup(backup)

    def test_unlisted_files_are_rejected_before_database_access(self):
        backup = create_backup(self.source, self.backups)
        (backup / "memory.sqlite-wal").write_bytes(b"unexpected journal")
        with self.assertRaisesRegex(ValueError, "unlisted files"):
            verify_backup(backup)

    def test_missing_source_evidence_prevents_publication_and_keeps_prior_backups(self):
        prior = create_backup(self.source, self.backups)
        next((self.source / "screenshots").glob("*.png")).unlink()
        with self.assertRaisesRegex(ValueError, "evidence"):
            create_backup(self.source, self.backups)
        self.assertTrue(verify_backup(prior)["verified"])
        self.assertEqual(list(self.backups.glob(".backup-*")), [])
        self.assertEqual([path for path in self.backups.iterdir() if path.is_dir()], [prior])

    def test_copy_failure_removes_partial_snapshot_and_preserves_original_error(self):
        original = OSError(errno.ENOSPC, "Simulated full disk")

        def fail_copy(incoming, outgoing, **kwargs):
            outgoing.write(b"partial")
            raise original

        with patch("pokemon_harness.telemetry.backup.shutil.copyfileobj", side_effect=fail_copy):
            with self.assertRaises(OSError) as failure:
                create_backup(self.source, self.backups)
        self.assertIs(failure.exception, original)
        self.assertEqual([path for path in self.backups.iterdir() if path.is_dir()], [])
        self.assertTrue((self.source / "memory.sqlite").exists())

    def test_cleanup_refuses_unrelated_directory(self):
        unrelated = self.root / "keep"
        unrelated.mkdir()
        (unrelated / "data").write_bytes(b"keep")
        with self.assertRaisesRegex(ValueError, "Refusing cleanup"):
            _discard_stage(unrelated, self.root)
        self.assertEqual((unrelated / "data").read_bytes(), b"keep")

    def test_backup_quota_counts_existing_archives_without_deleting_them(self):
        config = json.loads((self.source / "config.json").read_text())
        config["storage"]["max_backup_mb"] = 3
        write_json(self.source / "config.json", config)
        self.backups.mkdir()
        archive = self.backups / "prior-archive.bin"
        with archive.open("wb") as stream:
            stream.truncate(3 * MIB)
        with self.assertRaises(StorageLimitError):
            create_backup(self.source, self.backups)
        self.assertEqual(archive.stat().st_size, 3 * MIB)
        self.assertEqual(list(self.backups.glob(".backup-*")), [])

    def test_backup_storage_cannot_overlap_source(self):
        for target in (self.source, self.source / "backups", self.source.parent):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "separate"):
                create_backup(self.source, target)

    def test_portable_paths_reject_traversal_drives_and_alternate_streams(self):
        for relative in ("../outside", "/outside", "C:/outside", "file:stream", "folder\\file", "folder/./file"):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                _child(self.root, relative)

    def test_nested_configs_are_preserved_and_old_manifest_is_replaced(self):
        checks = self.source / "checks"
        checks.mkdir()
        write_json(checks / "config.json", {"diagnostic": True})
        write_json(self.source / "manifest.json", {"old": True})
        backup = create_backup(self.source, self.backups)
        self.assertEqual(json.loads((backup / "checks" / "config.json").read_text()), {"diagnostic": True})
        self.assertEqual(self.read_manifest(backup)["schema_version"], 1)
        self.assertTrue(verify_backup(backup)["verified"])

    def test_run_identity_mismatch_prevents_publication(self):
        config = json.loads((self.source / "config.json").read_text())
        config["objective"] = "Changed goal"
        write_json(self.source / "config.json", config)
        with self.assertRaisesRegex(ValueError, "identity"):
            create_backup(self.source, self.backups)
        self.assertEqual(list(self.backups.glob(".backup-*")), [])

    def test_backup_and_verification_commands_do_not_open_emulator(self):
        output = io.StringIO()
        with patch("pokemon_harness.cli.emulator_for") as emulator, contextlib.redirect_stdout(output):
            self.assertEqual(main(["backup", str(self.source), "--backup-dir", str(self.backups)]), 0)
        emulator.assert_not_called()
        backup = Path(json.loads(output.getvalue())["backup_dir"])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["verify-backup", str(backup)]), 0)
