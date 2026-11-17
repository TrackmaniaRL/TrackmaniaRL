"""Operator-supervised retirement of an isolated, quarantined endpoint.

This helper cannot establish which OS process hosted a historical plugin epoch.
It is only for an operator who has independently verified the facts below. It
never contacts the game or changes the old journal's operation state.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import socket
import sqlite3
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from trackmaniarl.trackmania.map_staging import (
    _close_handle,
    _lock_directory,
    _open_read_only_locked,
)
from trackmaniarl.trackmania.map_switch_controller import (
    _InstanceLock,
    _registry_root,
    _sync_registry_directory,
)

_SENTINEL = b"TrackmaniaRL endpoint retirement in progress; do not use\n"
_MAX_JOURNAL_BYTES = 128 * 1024 * 1024
_MAX_EVIDENCE_BYTES = 1024 * 1024
_CONNECT_PORTS = (9000, 9001, 9002)
_windows_ctypes: Any = ctypes


@dataclass(frozen=True, slots=True)
class RetirementAttestation:
    """Assertions by a trusted operator, not machine-verified identity proof."""

    old_game_process_exit_verified: bool
    no_external_workers_or_clients: bool
    exclusive_isolated_environment: bool
    no_active_training_resources: bool
    evidence_path: Path


def _phase(name: str) -> None:
    """Test hook for failures at durable transaction boundaries."""


def _directory_handle_identity(handle: int) -> tuple[int, int]:
    """Identify the opened Windows directory, not merely its current path."""

    class FileIdInfo(ctypes.Structure):
        _fields_ = [("volume", ctypes.c_ulonglong), ("file_id", ctypes.c_ubyte * 16)]

    kernel = _windows_ctypes.WinDLL("kernel32", use_last_error=True)
    query = kernel.GetFileInformationByHandleEx
    query.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    query.restype = wintypes.BOOL
    info = FileIdInfo()
    if not query(handle, 18, ctypes.byref(info), ctypes.sizeof(info)):
        raise _windows_ctypes.WinError(_windows_ctypes.get_last_error())
    return int(info.volume), int.from_bytes(info.file_id, "little")


@contextmanager
def _lock_path_directories(paths: tuple[Path, ...]) -> Iterator[None]:
    """Pin every ancestor against replacement until retirement has finished."""
    directories: set[Path] = set()
    for path in paths:
        absolute = Path(os.path.abspath(path))
        directories.update((absolute, *absolute.parents))
    # Lock ancestors before descendants; recheck identity around each open.
    with ExitStack() as stack:
        for directory in sorted(directories, key=lambda path: (len(path.parts), str(path))):
            before = directory.lstat()
            if not stat.S_ISDIR(before.st_mode):
                raise ValueError(f"not a directory: {directory}")
            handle = _lock_directory(directory)
            stack.callback(_close_handle, handle)
            opened = _directory_handle_identity(handle)
            after = directory.lstat()
            if (before.st_dev, before.st_ino) != opened or opened != (
                after.st_dev,
                after.st_ino,
            ):
                raise ValueError(f"directory changed while locking: {directory}")
        yield


def _checked_path(path: Path) -> Path:
    """Reject reparse points, then match MapSwitchJournal's canonical paths."""
    absolute = Path(os.path.abspath(path))
    for component in reversed((absolute, *absolute.parents)):
        entry = component.lstat()
        if stat.S_ISLNK(entry.st_mode) or getattr(entry, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"linked path component: {component}")
    return absolute.resolve(strict=True)


def _regular_bytes(path: Path, limit: int) -> bytes:
    entry = path.lstat()
    if (
        not stat.S_ISREG(entry.st_mode)
        or entry.st_nlink != 1
        or getattr(entry, "st_file_attributes", 0) & 0x400
    ):
        raise ValueError(f"unsafe or linked file: {path}")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (entry.st_dev, entry.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError(f"file changed while opening: {path}")
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"file exceeds retirement size limit: {path}")
    current = path.lstat()
    if (entry.st_dev, entry.st_ino) != (current.st_dev, current.st_ino):
        raise ValueError(f"file changed while reading: {path}")
    return data


def _require_no_sidecars(path: Path) -> None:
    for suffix in ("-journal", "-wal", "-shm"):
        try:
            Path(str(path) + suffix).lstat()
        except FileNotFoundError:
            continue
        raise ValueError(f"journal sidecar requires separate investigation: {path}{suffix}")


@contextmanager
def _locked_journal(path: Path) -> Iterator[sqlite3.Connection]:
    """Refuse sidecars and concurrent SQLite writers without changing the DB."""
    # Windows denies this handle if a writer already has the file open, and
    # denies later write/delete opens while allowing SQLite read-only opens.
    with _open_read_only_locked(path) as locked:
        opened = os.fstat(locked.fileno())
        entry = path.lstat()
        if (opened.st_dev, opened.st_ino) != (entry.st_dev, entry.st_ino):
            raise ValueError("journal changed before locking")
        _require_no_sidecars(path)
        _regular_bytes(path, _MAX_JOURNAL_BYTES)
        db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=0, isolation_level=None)
        try:
            db.execute("BEGIN")
            if db.execute("PRAGMA journal_mode").fetchone() != ("delete",):
                raise ValueError("retirement requires DELETE journal mode")
            if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise ValueError("journal integrity check failed")
            _require_no_sidecars(path)
            current = path.lstat()
            if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
                raise ValueError("journal changed while locking")
            yield db
        finally:
            db.close()


@contextmanager
def _reserve_game_ports() -> Iterator[None]:
    """Reserve all fixed local Connect ports until retirement finishes."""
    with ExitStack() as stack:
        for port in _CONNECT_PORTS:
            server = stack.enter_context(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
            if os.name == "nt":
                # This constant is exposed only by Windows socket stubs.
                exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE")  # noqa: B009
                server.setsockopt(socket.SOL_SOCKET, exclusive, 1)
            server.bind(("127.0.0.1", port))
        yield


def _write_new(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        if stream.write(data) != len(data):
            raise OSError("short write while preserving retirement evidence")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_replace(  # noqa: PLR0913 -- the hook identifies the replacement boundary
    root: Path, destination: Path, data: bytes, *, phase: str = ""
) -> None:
    temporary = root / (".retirement-write-" + uuid4().hex)
    _write_new(temporary, data)
    os.replace(temporary, destination)
    if phase:
        _phase(phase + "_replaced")
    _sync_registry_directory(root)


def _binding(instance_id: str, path: Path) -> bytes:
    return json.dumps(
        {"instance_id": instance_id, "journal": os.path.normcase(str(path))},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def retire_quarantined_endpoint(  # noqa: PLR0913 -- identities and attestation are explicit
    *,
    old_journal: Path,
    old_instance_id: str,
    new_journal: Path,
    new_instance_id: str,
    attestation: RetirementAttestation,
) -> Path:
    """Archive and retire a dispatching operation under explicit supervision.

    This experimental procedure cannot verify the operator's process-identity
    evidence. Do not use it unless every assertion has been independently
    established for an isolated environment with no active training resources.
    Before new binding publication the original quarantine or a non-JSON
    sentinel blocks normal controllers. If publication or cleanup reports an
    error after replacement, the binding may already be live; stop using the
    endpoint. Windows directory-entry durability after power loss is not
    established.
    """
    if os.name != "nt":
        raise RuntimeError("retirement requires Windows read-locked journals")
    assertions = (
        attestation.old_game_process_exit_verified,
        attestation.no_external_workers_or_clients,
        attestation.exclusive_isolated_environment,
        attestation.no_active_training_resources,
    )
    if any(value is not True for value in assertions):
        raise ValueError("all explicit operator attestations are required")
    with _lock_path_directories(
        (
            _registry_root(),
            Path(old_journal).absolute().parent,
            Path(new_journal).absolute().parent,
            Path(attestation.evidence_path).absolute().parent,
        )
    ):
        return _retire_locked_paths(
            old_journal=old_journal,
            old_instance_id=old_instance_id,
            new_journal=new_journal,
            new_instance_id=new_instance_id,
            attestation=attestation,
        )


def _retire_locked_paths(  # noqa: PLR0913
    *,
    old_journal: Path,
    old_instance_id: str,
    new_journal: Path,
    new_instance_id: str,
    attestation: RetirementAttestation,
) -> Path:
    if not old_instance_id.strip() or not new_instance_id.strip():
        raise ValueError("old and new instance IDs are required")
    if old_instance_id == new_instance_id:
        raise ValueError("the new instance ID must differ from the old instance ID")
    evidence_path = _checked_path(Path(attestation.evidence_path))
    old_path, new_path = _checked_path(Path(old_journal)), _checked_path(Path(new_journal))
    # Refuse aliases, including hardlinks, before opening SQLite.
    old_bytes = _regular_bytes(old_path, _MAX_JOURNAL_BYTES)
    _regular_bytes(new_path, _MAX_JOURNAL_BYTES)
    if os.path.samefile(old_path, new_path):
        raise ValueError("old and new journals must be distinct files")
    if os.path.samefile(evidence_path, old_path) or os.path.samefile(evidence_path, new_path):
        raise ValueError("operator evidence must be separate from both journals")
    evidence = _regular_bytes(evidence_path, _MAX_EVIDENCE_BYTES)
    if not evidence.strip():
        raise ValueError("nonempty independently collected operator evidence is required")

    root = _checked_path(_registry_root())
    key = hashlib.sha256(b"endpoint:127.0.0.1:9001").hexdigest()
    binding, pending, telemetry = (
        root / (key + ".json"),
        root / (key + ".pending"),
        root / (key + ".telemetry-uncertain"),
    )
    lock_path = root / (key + ".lock")
    for path in (lock_path, binding, pending):
        if os.path.samefile(evidence_path, path):
            raise ValueError("operator evidence must be separate from registry records")
    _regular_bytes(lock_path, 4096)  # Never silently create a missing ownership lock.
    with _InstanceLock(lock_path), _reserve_game_ports(), ExitStack() as stack:
        # A session guard needs a separate procedure; never erase it here.
        try:
            telemetry.lstat()
        except FileNotFoundError:
            pass
        else:
            raise ValueError("telemetry quarantine requires separate investigation")
        original_binding = _regular_bytes(binding, 4096)
        if original_binding != _binding(old_instance_id, old_path):
            raise ValueError("endpoint binding differs from the old journal")
        marker = _regular_bytes(pending, 32)
        if len(marker) != 32 or any(c not in b"0123456789abcdef" for c in marker):
            raise ValueError("invalid pending operation marker")
        evidence_handle = stack.enter_context(_open_read_only_locked(evidence_path))
        evidence_opened = os.fstat(evidence_handle.fileno())
        evidence_entry = evidence_path.lstat()
        if (evidence_opened.st_dev, evidence_opened.st_ino) != (
            evidence_entry.st_dev,
            evidence_entry.st_ino,
        ) or _regular_bytes(evidence_path, _MAX_EVIDENCE_BYTES) != evidence:
            raise ValueError("operator evidence changed before locking")
        old_db = stack.enter_context(_locked_journal(old_path))
        new_db = stack.enter_context(_locked_journal(new_path))
        if old_db.execute("SELECT instance_id, version FROM metadata").fetchall() != [
            (old_instance_id, 1)
        ]:
            raise ValueError("old journal identity differs")
        rows = old_db.execute(
            "SELECT operation_id, state FROM operations WHERE state IN ('prepared', 'dispatching')"
        ).fetchall()
        if rows != [(marker.decode("ascii"), "dispatching")]:
            raise ValueError("marker does not match one dispatched operation")
        if new_db.execute("SELECT instance_id, version FROM metadata").fetchall() != [
            (new_instance_id, 1)
        ] or new_db.execute("SELECT COUNT(*) FROM operations").fetchone() != (0,):
            raise ValueError("new journal must be empty and have the expected identity")
        if _regular_bytes(old_path, _MAX_JOURNAL_BYTES) != old_bytes:
            raise ValueError("old journal changed before it could be locked")
        new_bytes = _regular_bytes(new_path, _MAX_JOURNAL_BYTES)

        archive = root / ("retired-" + uuid4().hex)
        archive.mkdir(mode=0o700)
        archive_before = archive.lstat()
        archive_handle = _lock_directory(archive)
        stack.callback(_close_handle, archive_handle)
        archive_opened = _directory_handle_identity(archive_handle)
        archive_after = archive.lstat()
        if (archive_before.st_dev, archive_before.st_ino) != archive_opened or archive_opened != (
            archive_after.st_dev,
            archive_after.st_ino,
        ):
            raise ValueError("archive directory changed while locking")
        _phase("archive_created")
        items = {
            "binding.json": original_binding,
            "pending": marker,
            "old-journal.sqlite": old_bytes,
            "operator-evidence": evidence,
        }
        for name, data in items.items():
            _write_new(archive / name, data)
            _phase("archived:" + name)
        for name, data in items.items():
            if _regular_bytes(archive / name, max(_MAX_JOURNAL_BYTES, _MAX_EVIDENCE_BYTES)) != data:
                raise ValueError("retirement evidence readback differs")
        manifest = {
            "endpoint": "127.0.0.1:9001",
            "reserved_ports": [9000, 9001, 9002],
            "old_instance_id": old_instance_id,
            "old_journal": str(old_path),
            "new_instance_id": new_instance_id,
            "new_journal": str(new_path),
            "new_journal_sha256": hashlib.sha256(new_bytes).hexdigest(),
            "operation_id": marker.decode("ascii"),
            "operator_assertions": {
                "old_game_process_exit_verified": True,
                "no_external_workers_or_clients": True,
                "exclusive_isolated_environment": True,
                "no_active_training_resources": True,
                "verification": "operator assertion; not checked by this helper",
            },
            "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in items.items()},
        }
        manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
        _write_new(archive / "manifest.json", manifest_bytes)
        if _regular_bytes(archive / "manifest.json", _MAX_EVIDENCE_BYTES) != manifest_bytes:
            raise ValueError("retirement manifest readback differs")
        _phase("manifest_written")
        _sync_registry_directory(archive)
        _sync_registry_directory(root)
        _phase("archive_synced")

        # Legacy controllers compare this file to a JSON object. This fixed
        # non-JSON sentinel cannot equal any valid binding they construct.
        if _regular_bytes(binding, 4096) != original_binding:
            raise ValueError("endpoint binding changed before retirement")
        _atomic_replace(root, binding, _SENTINEL, phase="sentinel")
        _phase("sentinel_installed")
        if _regular_bytes(pending, 32) != marker:
            raise ValueError("pending marker changed during retirement")
        pending.unlink()
        _phase("pending_unlinked")
        _sync_registry_directory(root)
        _phase("pending_retired")
        if _regular_bytes(binding, 4096) != _SENTINEL:
            raise ValueError("retirement sentinel changed")
        if _regular_bytes(new_path, _MAX_JOURNAL_BYTES) != new_bytes:
            raise ValueError("new journal changed during retirement")
        for path in (pending, telemetry):
            try:
                path.lstat()
            except FileNotFoundError:
                continue
            raise ValueError("a quarantine marker appeared during retirement")
        _require_no_sidecars(old_path)
        _require_no_sidecars(new_path)
        try:
            _atomic_replace(root, binding, _binding(new_instance_id, new_path), phase="new_binding")
        except BaseException as error:
            # os.replace may have succeeded before the directory sync failed.
            # Best-effort reinstallation reduces that ambiguity; if it also
            # fails, the operator must treat publication as uncertain.
            try:
                _atomic_replace(root, binding, _SENTINEL)
            except BaseException as restore_error:
                error.add_note(f"Could not restore retirement sentinel: {restore_error!r}")
            raise
        return archive
