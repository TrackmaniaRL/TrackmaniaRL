"""Optional immutable copies of scheduled checkpoints at transition milestones."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CheckpointProgress:
    baseline: int
    transitions: int
    updates: int


def milestone_due(milestones: tuple[int, ...], previous: int, current: int) -> bool:
    return any(previous < threshold <= current for threshold in milestones)


def preserve_milestones(
    path: Path, milestones: tuple[int, ...], progress: CheckpointProgress
) -> tuple[Path, ...]:
    """Copy only durable snapshots, without synthesizing checkpoints before resume.

    Each copy records both its requested threshold and actual snapshot counters.
    A completion sidecar is written last; incomplete copies fail closed on retry.
    Copies are outside the rotating recovery filename families.
    """

    if not milestones:
        return ()
    if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
        raise ValueError("milestone source must be a regular checkpoint")
    directory = path.parent.resolve(strict=True)
    if directory.name != "checkpoints":
        raise ValueError("milestones require the configured checkpoints directory")
    saved: list[Path] = []
    for threshold in milestones:
        if not progress.baseline < threshold <= progress.transitions:
            continue
        prefix = f"milestone-{threshold:012d}-"
        existing = list(directory.glob(f"{prefix}*.pt"))
        if existing:
            _validate_existing(existing)
            continue
        name = f"{prefix}transitions-{progress.transitions:012d}-update-{progress.updates:08d}.pt"
        destination = directory / name
        _copy_checkpoint(path, destination)
        metadata = {
            "requested_transitions": threshold,
            "transitions": progress.transitions,
            "updates": progress.updates,
            "checkpoint": name,
            "bytes": destination.stat().st_size,
        }
        with destination.with_suffix(".json").open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        saved.append(destination)
    return tuple(saved)


def _copy_checkpoint(source: Path, destination: Path) -> None:
    # Independent bytes also survive a custom codec rewriting its recovery file
    # in place. Exclusive creation prevents replacing historical milestone data.
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        outgoing.flush()
        os.fsync(outgoing.fileno())


def _validate_existing(paths: list[Path]) -> None:
    if len(paths) != 1:
        raise ValueError("multiple checkpoints exist for one transition milestone")
    path = paths[0]
    sidecar = path.with_suffix(".json")
    if path.is_symlink() or sidecar.is_symlink() or not sidecar.is_file():
        raise ValueError("milestone checkpoint is unsafe or incomplete")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if (
        not isinstance(metadata, dict)
        or metadata.get("checkpoint") != path.name
        or metadata.get("bytes") != path.stat().st_size
    ):
        raise ValueError("milestone checkpoint completion metadata differs")
