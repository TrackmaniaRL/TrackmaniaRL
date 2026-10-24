"""Portable reserve checks for explicitly owned campaign and report locations."""

from __future__ import annotations

from pathlib import Path

from trackmaniarl.experiments.full_qr.campaign_budget import StorageReserve


def _existing_directory(path: Path) -> Path:
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError("canonical absolute storage location required")
    while not path.exists():
        path = path.parent
    if not path.is_dir() or path.resolve(strict=True) != path:
        raise ValueError("storage reserve requires an existing canonical directory")
    return path


def campaign_storage_reserves(directory: Path, artifact_root: Path) -> tuple[StorageReserve, ...]:
    """Keep 10GiB on the host system volume and 4GiB on each output volume.

    WSL's Windows system mount retains its independent reserve. Native Linux
    uses its system root. Reports follow the explicitly pinned artifact path,
    rather than assuming a particular user's secondary drive letter.
    """
    system = Path("/mnt/c") if Path("/mnt/c").is_dir() else Path(directory.anchor)
    return (
        StorageReserve(_existing_directory(system), 10 * 1024**3),
        StorageReserve(_existing_directory(directory), 4 * 1024**3),
        StorageReserve(_existing_directory(artifact_root), 4 * 1024**3),
    )
