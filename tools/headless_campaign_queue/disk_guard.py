"""Check actual volume free space for an explicitly configured WSL campaign.

This is a point-in-time reserve check, not a forecast or a campaign size quota.
Call it before work and periodically while owned training processes are running.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

GIB = 1024**3
SYSTEM_RESERVE_BYTES = 10 * GIB
DATA_RESERVE_BYTES = 4 * GIB


def _existing_directory(path: Path) -> Path:
    """Resolve a prospective output to its nearest existing directory.

    A missing /mnt/<drive> must not silently fall back to the Linux root volume.
    """
    if not path.is_absolute():
        raise ValueError(f"disk check requires an absolute path: {path}")
    path = path.resolve(strict=False)
    parts = path.parts
    if len(parts) >= 3 and parts[1] == "mnt" and len(parts[2]) == 1 and parts[2].isalpha():
        mount = Path("/mnt") / parts[2]
        if not mount.is_dir() or not os.path.ismount(mount):
            raise OSError(f"required Windows volume is not mounted: {mount}")
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise OSError(f"no existing parent for disk check: {path}")
        candidate = parent
    if not candidate.is_dir():
        raise ValueError(f"disk check target or existing parent is not a directory: {candidate}")
    return candidate


def _config_path(value: Any, *, name: str, base_dir: Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"missing disk path in configuration: {name}")
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def check_disk_space(
    spec: Any, base_dir: Path, *, system_path: Path = Path("/mnt/c")
) -> list[dict[str, Any]]:
    """Check every configured output volume and return a serializable receipt.

    `spec` is the validated RunSpec. Errors are fatal: unknown/missing locations,
    inaccessible mounts, and free space below a reserve never imply readiness.
    """
    base_dir = Path(base_dir).resolve(strict=True)
    if not base_dir.is_dir():
        raise ValueError("configuration base must be a directory")
    try:
        environment = spec.components.environment.kwargs["config"]
        runtime = environment["runtime"]
        archive = environment["trace_archive_root"]
        trace = runtime["trace_dir"]
        work = runtime["work_dir"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError("native output paths are missing from the run configuration") from exc
    configured = [
        ("system", Path(system_path), SYSTEM_RESERVE_BYTES),
        (
            "artifacts",
            _config_path(spec.artifacts_dir, name="artifacts_dir", base_dir=base_dir),
            DATA_RESERVE_BYTES,
        ),
        (
            "checkpoints",
            _config_path(
                spec.checkpoint_root or spec.artifacts_dir,
                name="checkpoint_root",
                base_dir=base_dir,
            ),
            DATA_RESERVE_BYTES,
        ),
        (
            "native_traces",
            _config_path(trace, name="trace_dir", base_dir=base_dir),
            DATA_RESERVE_BYTES,
        ),
        ("native_work", _config_path(work, name="work_dir", base_dir=base_dir), DATA_RESERVE_BYTES),
        (
            "native_archives",
            _config_path(archive, name="trace_archive_root", base_dir=base_dir),
            DATA_RESERVE_BYTES,
        ),
    ]
    result = []
    for label, target, minimum in configured:
        directory = _existing_directory(target)
        available = shutil.disk_usage(directory).free
        if available < minimum:
            raise OSError(
                f"{label}: insufficient free disk space at {directory}: "
                f"{available} bytes available, {minimum} required"
            )
        result.append(
            {
                "name": label,
                "target": str(target),
                "checked_at": str(directory),
                "available_bytes": available,
                "reserve_bytes": minimum,
            }
        )
    return result
