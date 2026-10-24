"""Bounded run accounting and atomic owned metadata for Linux headless training.

Callers supply only their own run/checkpoint paths and explicitly owned extra
evidence roots. Raw evidence is never deleted. This is a checked byte guard,
not a kernel quota; native per-session limits remain independently necessary.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import IO, Any, BinaryIO, cast
from uuid import uuid4

from trackmaniarl.trackmania.native_headless._runtime.posix_api import linux_os


class TrainingStorageError(OSError):
    """Storage cannot safely publish or account for this owned run."""


MAX_CLEANUP_RECEIPT_BYTES = 1024 * 1024


def _nofollow_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise TrainingStorageError("native training storage requires Linux no-follow directory I/O")
    return os.O_RDONLY | linux_os.O_DIRECTORY | linux_os.O_NOFOLLOW


def directory_bytes(paths: Iterable[Path], *, max_entries: int = 100_000) -> int:
    """Sum logical file/link bytes, deduplicating nested roots without following links.

    Hard-linked file names are charged separately, conservatively. Missing
    explicit roots contribute zero. Directory descriptors prevent a directory
    being replaced with a symlink between classification and traversal.
    """
    if type(max_entries) is not int or max_entries <= 0:
        raise ValueError("max_entries must be a positive integer")
    flags = _nofollow_flags()
    roots: set[Path] = set()
    for index, value in enumerate(paths):
        if index >= max_entries:
            raise TrainingStorageError("evidence root entry limit exceeded")
        path = Path(os.path.abspath(value))
        if path.parent.resolve() != path.parent:
            raise TrainingStorageError("evidence root must not traverse a symlink")
        roots.add(path)
    selected: list[Path] = []
    for path in sorted(roots, key=lambda root: (len(root.parts), str(root))):
        if not any(parent in roots for parent in path.parents):
            selected.append(path)
    total, entries = 0, 0

    def visit(descriptor: int, depth: int) -> None:
        nonlocal total, entries
        if depth > 128:
            raise TrainingStorageError("evidence directory depth limit exceeded")
        with os.scandir(descriptor) as children:
            for child in children:
                entries += 1
                if entries > max_entries:
                    raise TrainingStorageError("evidence entry limit exceeded")
                observed = child.stat(follow_symlinks=False)
                if stat.S_ISDIR(observed.st_mode):
                    nested = os.open(child.name, flags, dir_fd=descriptor)
                    try:
                        actual = os.fstat(nested)
                        if (actual.st_dev, actual.st_ino) != (observed.st_dev, observed.st_ino):
                            raise TrainingStorageError("evidence directory changed during scan")
                        visit(nested, depth + 1)
                    finally:
                        os.close(nested)
                elif stat.S_ISREG(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
                    total += observed.st_size
                elif stat.S_ISSOCK(observed.st_mode):
                    # Closed native leases retain their Unix control socket name.
                    continue
                else:
                    raise TrainingStorageError("unexpected non-file evidence entry")

    for path in selected:
        try:
            observed = path.lstat()
        except FileNotFoundError:
            continue
        entries += 1
        if entries > max_entries:
            raise TrainingStorageError("evidence entry limit exceeded")
        if stat.S_ISDIR(observed.st_mode):
            descriptor = os.open(path, flags)
            try:
                actual = os.fstat(descriptor)
                if (actual.st_dev, actual.st_ino) != (observed.st_dev, observed.st_ino):
                    raise TrainingStorageError("evidence root changed during scan")
                visit(descriptor, 0)
            finally:
                os.close(descriptor)
        elif stat.S_ISREG(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
            total += observed.st_size
        elif stat.S_ISSOCK(observed.st_mode):
            continue
        else:
            raise TrainingStorageError("unexpected non-file evidence root")
    return total


class _BoundedWriter:
    def __init__(self, stream: BinaryIO, max_bytes: int | None) -> None:
        self.stream, self.max_bytes = stream, max_bytes
        self.size = 0
        self.digest = hashlib.sha256()
        self.failure: TrainingStorageError | None = None

    def write(self, value: bytes) -> int:
        if self.max_bytes is not None and self.size + len(value) > self.max_bytes:
            self.failure = TrainingStorageError("atomic publication byte limit exceeded")
            raise self.failure
        written = self.stream.write(value)
        if written != len(value):
            self.failure = TrainingStorageError("atomic publication short write")
            raise self.failure
        self.size += written
        self.digest.update(value)
        return written

    def flush(self) -> None:
        self.stream.flush()

    def tell(self) -> int:
        return self.size


def _atomic(  # noqa: PLR0913 -- target, payload, serializer and byte ceiling are independent
    path: Path, payload: Any, *, checkpoint: bool, max_bytes: int | None
) -> dict[str, Any]:
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
        raise ValueError("max_bytes must be a nonnegative integer or None")
    path = Path(path)
    if not path.is_absolute() or path.parent.resolve(strict=True) != path.parent:
        raise TrainingStorageError("owned atomic target needs an exact canonical parent")
    directory = os.open(path.parent, _nofollow_flags())
    stage = f".{path.name}.stage-{uuid4().hex}"
    try:
        try:
            previous = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            previous = None
        if previous is not None and not stat.S_ISREG(previous.st_mode):
            raise TrainingStorageError("atomic target must not be a symlink or non-file")
        descriptor = os.open(
            stage,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | linux_os.O_NOFOLLOW,
            0o600,
            dir_fd=directory,
        )
        with os.fdopen(descriptor, "wb") as stream:
            writer = _BoundedWriter(stream, max_bytes)
            if checkpoint:
                import torch

                try:
                    torch.save(payload, cast(IO[bytes], writer))
                except Exception as error:
                    # Zip finalization can wrap a previous bounded-write error.
                    if writer.failure is not None:
                        raise writer.failure from error
                    raise
            else:
                raw = (
                    json.dumps(
                        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
                    ).encode("utf-8")
                    + b"\n"
                )
                writer.write(raw)
            writer.flush()
            os.fsync(stream.fileno())
        # A failed serialization/fsync retains the stage as counted evidence;
        # it cannot destroy the previously published checkpoint/report.
        os.replace(stage, path.name, src_dir_fd=directory, dst_dir_fd=directory)
        os.fsync(directory)
        return {
            "path": str(path),
            "bytes": writer.size,
            "sha256": writer.digest.hexdigest(),
            "file_and_directory_fsync": True,
        }
    finally:
        os.close(directory)


def atomic_checkpoint(path: Path, state: Any, *, max_bytes: int | None = None) -> dict[str, Any]:
    """Atomically replace an explicitly owned checkpoint; lazy-import torch.

    max_bytes caps the new stage, including its peak coexistence with the old
    checkpoint in a run-wide budget computed by the caller. It is not a cap
    on the sum of other concurrently growing evidence files.
    """
    return _atomic(path, state, checkpoint=True, max_bytes=max_bytes)


def atomic_json(path: Path, data: Any, *, max_bytes: int | None = None) -> dict[str, Any]:
    """Atomically replace an explicitly owned JSON report/index with finite data."""
    return _atomic(path, data, checkpoint=False, max_bytes=max_bytes)


def verify_native_cleanup(environment: Any) -> dict[str, Any]:
    """Close this environment and inspect its existing owned V7 cleanup receipt."""
    path = environment.config.trace_dir / "close-v7.json"
    errors: list[str] = []
    supervisor: Any = None
    retained: str | None = None
    failures: list[str] = []
    trace_accounting: Any = None
    try:
        environment.close()
    except (Exception, KeyboardInterrupt) as error:
        errors.append(repr(error))
    try:
        if path.is_symlink() or path.parent.resolve() != path.parent:
            raise TrainingStorageError("cleanup receipt must not traverse a symlink")
        directory = os.open(path.parent, _nofollow_flags())
        try:
            descriptor = os.open(
                path.name, os.O_RDONLY | linux_os.O_NOFOLLOW | linux_os.O_NONBLOCK, dir_fd=directory
            )
            with os.fdopen(descriptor, "rb") as stream:
                metadata = os.fstat(stream.fileno())
                if not stat.S_ISREG(metadata.st_mode):
                    raise TrainingStorageError("cleanup receipt must be a regular file")
                if metadata.st_size > MAX_CLEANUP_RECEIPT_BYTES:
                    raise TrainingStorageError("cleanup receipt byte limit exceeded")
                raw = stream.read(MAX_CLEANUP_RECEIPT_BYTES + 1)
                if len(raw) > MAX_CLEANUP_RECEIPT_BYTES:
                    raise TrainingStorageError("cleanup receipt byte limit exceeded during read")
        finally:
            os.close(directory)
        report = json.loads(raw)
        supervisor = report["supervisor"]
        trace_accounting = report.get("traceAccounting")
        if not isinstance(report["failures"], list):
            raise TrainingStorageError("invalid native cleanup failures")
        failures = report["failures"]
        location = Path(report["retainedLocalDirectory"])
        if (
            location.parent != Path("/tmp")
            or not location.name.startswith("tmre-immediate-v7-")
            or location.resolve() != location
        ):
            raise TrainingStorageError("invalid retained owned native directory")
        retained = str(location)
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(repr(error))
    gone = (
        isinstance(supervisor, Mapping)
        and supervisor.get("remainingOwned") == []
        and supervisor.get("cleanupErrors") == []
        and not supervisor.get("supervisorError")
    )
    return {
        "path": str(path),
        "verified": gone and not errors and not failures and not supervisor.get("publicationError"),
        "owned_processes_gone": gone,
        "close_errors": errors,
        "native_failures": failures,
        "retained_local_directory": retained,
        "supervisor": supervisor,
        "trace_accounting": trace_accounting,
    }
