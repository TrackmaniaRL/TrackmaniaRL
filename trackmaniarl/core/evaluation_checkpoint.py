"""Bind local evaluation completion and artifacts to immutable checkpoint bytes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path


def checkpoint_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def receipt_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(checkpoint.name + ".evaluation.json")


@dataclass(frozen=True, slots=True)
class EvaluationReceiptCheck:
    checkpoint: Path
    fingerprint: str
    run_dir: Path
    require_artifact: bool
    require_snapshot: bool


def evaluation_completed(check: EvaluationReceiptCheck) -> bool:
    checkpoint = check.checkpoint
    run_dir = check.run_dir
    try:
        receipt = json.loads(receipt_path(checkpoint).read_text(encoding="utf-8"))
        if not isinstance(receipt, dict) or set(receipt) != {
            "checkpoint_sha256",
            "run_fingerprint",
            "snapshot_name",
            "artifact_sha256",
        }:
            return False
        name = receipt["snapshot_name"]
        artifact_digest = receipt["artifact_sha256"]
        if not (name is None or _valid_snapshot_name(name)) or not (
            artifact_digest is None or isinstance(artifact_digest, str)
        ):
            return False
        if (check.require_artifact and artifact_digest is None) or (
            name is None and artifact_digest is not None
        ):
            return False
        digest = checkpoint_digest(checkpoint)
        if (
            not isinstance(receipt["checkpoint_sha256"], str)
            or receipt["checkpoint_sha256"] != digest
            or receipt["run_fingerprint"] != check.fingerprint
        ):
            return False
        if name is None:
            return not check.require_artifact and not check.require_snapshot
        snapshot = run_dir / "evaluation-checkpoints" / str(name)
        if snapshot.is_symlink() or not snapshot.is_file() or checkpoint_digest(snapshot) != digest:
            return False
        return artifact_digest is None or _artifact_digest(run_dir, snapshot) == artifact_digest
    except (OSError, ValueError, TypeError):
        return False


@dataclass(frozen=True, slots=True)
class EvaluationReceipt:
    checkpoint: Path
    fingerprint: str
    digest: str
    snapshot: Path | None
    run_dir: Path
    require_artifact: bool


def record_evaluation(receipt: EvaluationReceipt) -> None:
    checkpoint = receipt.checkpoint
    digest = receipt.digest
    snapshot = receipt.snapshot
    if checkpoint_digest(checkpoint) != digest:
        raise RuntimeError("checkpoint changed during evaluation")
    if snapshot is not None and checkpoint_digest(snapshot) != digest:
        raise RuntimeError("evaluation snapshot changed during evaluation")
    artifact_digest = None if snapshot is None else _artifact_digest(receipt.run_dir, snapshot)
    if receipt.require_artifact and artifact_digest is None:
        raise RuntimeError("evaluation artifact is missing or does not match its checkpoint")
    payload = {
        "checkpoint_sha256": digest,
        "run_fingerprint": receipt.fingerprint,
        "snapshot_name": None if snapshot is None else snapshot.name,
        "artifact_sha256": artifact_digest,
    }
    _atomic_json(receipt_path(checkpoint), payload)


def _atomic_json(destination: Path, payload: Mapping[str, object]) -> None:
    fd, name = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)


def create_evaluation_snapshot(checkpoint: Path, run_dir: Path, keep: int | None) -> Path:
    """Copy a full checkpoint to a unique path, never modifying earlier artifacts."""

    def copy(destination: Path) -> None:
        with checkpoint.open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())

    return create_evaluation_snapshot_with(run_dir, copy, keep)


def create_evaluation_snapshot_with(
    run_dir: Path, save: Callable[[Path], None], keep: int | None
) -> Path:
    """Create a new full snapshot before exposing its path to an evaluator."""
    directory = run_dir / "evaluation-checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise ValueError("evaluation checkpoint directory must not be a symlink")
    manifest = _read_manifest(directory)
    protected = {value for value in manifest.values() if value is not None}
    exempt = _retention_exempt(directory, keep)
    artifact_snapshot = _artifact_snapshot(run_dir, directory)
    if artifact_snapshot is not None:
        protected.add(artifact_snapshot)
    _prune_snapshots(directory, (protected, exempt), keep)
    destination = directory / f"snapshot-{uuid.uuid4().hex}.pt"
    try:
        save(destination)
        if destination.is_symlink() or not destination.is_file():
            raise ValueError("evaluation snapshot must be a regular file")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    # Persist the in-progress reference before giving its path to an evaluator.
    _atomic_json(
        directory / "manifest.json",
        {"completed": manifest["completed"], "pending": destination.name},
    )
    return destination


def prune_evaluation_snapshots(current: Path, keep: int | None) -> None:
    """Retain the latest completed snapshot after its artifact has been written."""
    directory = current.parent
    if directory.is_symlink() or current.is_symlink() or not current.is_file():
        raise ValueError("evaluation checkpoint path is unsafe")
    _atomic_json(directory / "manifest.json", {"completed": current.name, "pending": None})
    protected = {current.name}
    exempt = _retention_exempt(directory, keep)
    artifact_snapshot = _artifact_snapshot(directory.parent, directory)
    if artifact_snapshot is not None:
        protected.add(artifact_snapshot)
    _prune_snapshots(directory, (protected, exempt), keep)


def _read_manifest(directory: Path) -> dict[str, str | None]:
    path = directory / "manifest.json"
    if not path.exists():
        return {"completed": None, "pending": None}
    if path.is_symlink():
        raise ValueError("evaluation checkpoint manifest must not be a symlink")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"completed", "pending"}:
        raise ValueError("evaluation checkpoint manifest is invalid")
    if any(value is not None and not _valid_snapshot_name(value) for value in data.values()):
        raise ValueError("evaluation checkpoint manifest is invalid")
    return data


def _valid_snapshot_name(name: object) -> bool:
    if not isinstance(name, str) or not name.startswith("snapshot-") or not name.endswith(".pt"):
        return False
    suffix = name[len("snapshot-") : -len(".pt")]
    return len(suffix) == 32 and all(char in "0123456789abcdef" for char in suffix)


def _retention_exempt(directory: Path, keep: int | None) -> set[str]:
    """Preserve snapshots that predate the introduction of bounded retention."""
    if keep is None:
        return set()
    path = directory / "retention-exempt.json"
    if path.is_symlink():
        raise ValueError("evaluation retention exemption file must not be a symlink")
    if not path.exists():
        names = sorted(
            item.name
            for item in directory.iterdir()
            if _valid_snapshot_name(item.name) and item.is_file() and not item.is_symlink()
        )
        _atomic_json(path, {"snapshots": names})
        return set(names)
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or set(data) != {"snapshots"}
        or not isinstance(data["snapshots"], list)
        or not all(_valid_snapshot_name(name) for name in data["snapshots"])
    ):
        raise ValueError("evaluation retention exemption file is invalid")
    return set(data["snapshots"])


def _artifact_snapshot(run_dir: Path, directory: Path) -> str | None:
    try:
        data = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
        path = Path(data["checkpoint"])
        if path.parent.resolve() == directory.resolve() and _valid_snapshot_name(path.name):
            return path.name
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return None


def _artifact_digest(run_dir: Path, snapshot: Path) -> str | None:
    artifact = run_dir / "evaluation.json"
    try:
        data = json.loads(artifact.read_text(encoding="utf-8"))
        if (
            not isinstance(data, dict)
            or data.get("status") != "complete"
            or Path(data["checkpoint"]).resolve() != snapshot.resolve()
            or data.get("checkpoint_sha256") != checkpoint_digest(snapshot)
        ):
            return None
        return checkpoint_digest(artifact)
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _prune_snapshots(
    directory: Path, protections: tuple[set[str], set[str]], keep: int | None
) -> None:
    if keep is None:
        return
    protected, exempt = protections
    candidates = [
        path
        for path in directory.glob("snapshot-*.pt")
        if _valid_snapshot_name(path.name) and not path.is_symlink() and path.is_file()
    ]
    retained = {path.name for path in candidates if path.name in protected or path.name in exempt}
    unprotected = sorted(
        (path for path in candidates if path.name not in retained),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    retained.update(path.name for path in unprotected[: max(0, keep - len(retained - exempt))])
    for path in candidates:
        if path.name not in retained:
            path.unlink()
