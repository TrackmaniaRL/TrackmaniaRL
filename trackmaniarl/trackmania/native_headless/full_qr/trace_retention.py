"""Bounded private archives of fully closed actor sessions; never prune live data."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tarfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trackmaniarl.trackmania.native_headless._runtime import ownership


@dataclass(frozen=True, slots=True)
class TraceRetentionSettings:
    trace_parent: Path
    work_parent: Path
    archive_root: Path
    keep_archives: int = 3
    max_source_bytes: int = 512 * 1024**2

    def __post_init__(self) -> None:
        if self.keep_archives < 1 or self.max_source_bytes < 1:
            raise ValueError("positive trace retention limits are required")
        for path in (self.trace_parent, self.work_parent, self.archive_root):
            if not path.is_absolute() or path.resolve() != path:
                raise ValueError("canonical absolute session/archive paths required")
        if self.archive_root.exists():
            raise ValueError("a fresh actor archive directory is required")


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            result.update(chunk)
    return result.hexdigest()


def _atomic(path: Path, value: dict[str, Any]) -> None:
    stage = path.with_suffix(".tmp")
    with stage.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(stage, path)


class ClosedTraceRetention:
    """Retain the last archives and all small receipts for one fresh actor.

    Regular files are archived literally; symlinks/sockets are recorded as
    metadata and never followed or recreated. Older archive bodies may be
    pruned only if they still match this instance's own recorded digest.
    """

    def __init__(self, settings: TraceRetentionSettings) -> None:
        self.settings = settings
        self._sequence = 0
        self._archives: deque[tuple[Path, Path, dict[str, Any]]] = deque()

    def _closed_roots(self, session: Any, children: list[dict[str, Any]]) -> dict[str, Path]:
        settings = self.settings
        trace, work, temporary = session.out, session.work, session.tmp
        if (
            not session.closed
            or trace.parent != settings.trace_parent
            or work.parent != settings.work_parent
            or trace.name != work.name
            or not (
                trace.name.startswith("driver-")
                and trace.name[7:].isdigit()
                and len(trace.name) == 15
            )
            or trace != session.settings.trace_dir
            or work != session.settings.work_dir
            or temporary.parent != Path("/tmp")
            or not temporary.name.startswith("tmre-immediate-v7-")
        ):
            raise ValueError("only this actor's fully closed driver session may be archived")
        roots = {"trace": trace, "work": work, "temporary": temporary}
        for path in roots.values():
            if path.resolve(strict=True) != path or not path.is_dir():
                raise ValueError("session root changed or is not canonical")
            if settings.archive_root.is_relative_to(path) or path.is_relative_to(
                settings.archive_root
            ):
                raise ValueError("archive and source roots must not overlap")
        close = json.loads((trace / "close-v7.json").read_text())
        supervisor = json.loads((trace / "native-supervisor.json").read_text())
        if (
            close["error"]
            or close["failures"]
            or close["retainedLocalDirectory"] != str(temporary)
            or supervisor["remainingOwned"]
            or supervisor["cleanupErrors"]
            or supervisor.get("supervisorError")
            or supervisor["returncode"] != 0
            or session.proc.supervisor.poll() is None
            or not ownership.same_identity(
                supervisor["driverIdentity"], session.proc.driver_identity
            )
            or not ownership.same_identity(
                supervisor["supervisorIdentity"], session.proc.supervisor_identity
            )
        ):
            raise ValueError("native closure and exact supervisor identities must be proven")
        identities = [*children, supervisor["driverIdentity"], supervisor["supervisorIdentity"]]
        if not children:
            raise ValueError("captured native child identities are required")
        for identity in identities:
            if ownership.same_identity(ownership.process_identity(identity["pid"]), identity):
                raise ValueError("an exact owned process identity still exists")
        return roots

    def archive(
        self, session: Any, children: list[dict[str, Any]], *, reason: str
    ) -> dict[str, Any]:
        roots = self._closed_roots(session, children)
        records: list[dict[str, Any]] = []
        paths: dict[str, Path] = {}
        total = 0
        for prefix, root in roots.items():
            for directory, directories, files in os.walk(root, followlinks=False):
                for name in [*directories, *files]:
                    path = Path(directory) / name
                    observed = path.lstat()
                    member = prefix + "/" + path.relative_to(root).as_posix()
                    record: dict[str, Any]
                    if stat.S_ISREG(observed.st_mode):
                        total += observed.st_size
                        record = {
                            "name": member,
                            "kind": "file",
                            "bytes": observed.st_size,
                            "sha256": _digest(path),
                        }
                        paths[member] = path
                    elif stat.S_ISLNK(observed.st_mode):
                        record = {
                            "name": member,
                            "kind": "symlink-metadata",
                            "target": os.readlink(path),
                        }
                    elif stat.S_ISSOCK(observed.st_mode):
                        record = {"name": member, "kind": "closed-socket-metadata"}
                    elif stat.S_ISDIR(observed.st_mode):
                        continue
                    else:
                        raise ValueError("unsupported node in owned session data")
                    records.append(record)
        if total > self.settings.max_source_bytes:
            raise ValueError("closed trace exceeds its archive input quota")
        root = self.settings.archive_root
        root.mkdir(parents=True, exist_ok=True)
        self._sequence += 1
        stem = f"session-{self._sequence:08d}"
        archive, stage, receipt = (
            root / (stem + ".tar.gz"),
            root / (stem + ".tar.gz.stage"),
            root / (stem + ".json"),
        )
        if archive.exists() or receipt.exists():
            raise ValueError("never replace an existing actor archive")
        started = time.monotonic()
        with (
            stage.open("xb") as stream,
            tarfile.open(fileobj=stream, mode="w:gz", compresslevel=1) as tar,
        ):
            for record in records:
                if record["kind"] != "file":
                    continue
                path = paths[record["name"]]
                info = tarfile.TarInfo(record["name"])
                info.size, info.mode = record["bytes"], 0o600
                with path.open("rb") as source:
                    tar.addfile(info, source)
        # Read the compressed archive back, without extracting paths or links.
        verified = {}
        with tarfile.open(stage, "r:gz") as tar:
            for entry in tar:
                if not entry.isfile() or entry.name not in paths:
                    raise ValueError("unexpected archive member")
                archived_source = tar.extractfile(entry)
                if archived_source is None:
                    raise ValueError("archive member unavailable")
                digest = hashlib.sha256()
                with archived_source:
                    while chunk := archived_source.read(8 * 1024**2):
                        digest.update(chunk)
                verified[entry.name] = digest.hexdigest()
        expected = {row["name"]: row["sha256"] for row in records if row["kind"] == "file"}
        if verified != expected or any(
            _digest(paths[name]) != sha for name, sha in expected.items()
        ):
            raise ValueError("closed source/archive changed during verification")
        os.link(stage, archive)
        stage.unlink()
        value = {
            "schema": "tmrl.closed-actor-trace/v1",
            "reason": reason,
            "archive": str(archive),
            "archive_sha256": _digest(archive),
            "source_bytes": total,
            "archive_bytes": archive.stat().st_size,
            "records": records,
            "children": children,
            "sources_pruned": False,
            "archive_body_pruned": False,
            "roots": {key: str(path) for key, path in roots.items()},
        }
        _atomic(receipt, value)
        # Recheck ownership and canonical absolute paths immediately before
        # removal. shutil.rmtree does not follow child symlinks.
        if self._closed_roots(session, children) != roots:
            raise ValueError("owned session roots changed before pruning")
        for path in roots.values():
            shutil.rmtree(path)
        value.update(sources_pruned=True, archive_seconds=time.monotonic() - started)
        _atomic(receipt, value)
        self._archives.append((archive, receipt, value))
        while len(self._archives) > self.settings.keep_archives:
            old_archive, old_receipt, old = self._archives[0]
            if (
                old_archive.parent != root
                or old_archive.is_symlink()
                or _digest(old_archive) != old["archive_sha256"]
            ):
                raise ValueError("old owned archive changed; refusing retention pruning")
            old_archive.unlink()
            old["archive_body_pruned"] = True
            _atomic(old_receipt, old)
            self._archives.popleft()
        return value
