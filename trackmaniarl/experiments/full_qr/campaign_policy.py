"""Campaign timing, storage reserves and learning-schedule policy.

These checks describe a campaign without starting workers or a watchdog.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from trackmaniarl.core.spec import RunSpec

PILOT_SECONDS = 14_400
SUPPORTED_PILOT_SECONDS = (PILOT_SECONDS, 25_200)
SUPPORTED_CLOCK_SOURCES = ("monotonic", "monotonic_raw")


@dataclass(frozen=True, slots=True)
class StorageReserve:
    path: Path
    minimum_free_bytes: int

    def __post_init__(self) -> None:
        if self.minimum_free_bytes <= 0:
            raise ValueError("storage reserve must be positive")


def validate_clock_source(value: object) -> str:
    if not isinstance(value, str) or value not in SUPPORTED_CLOCK_SOURCES:
        raise ValueError("supported explicit campaign clock source required")
    return value


def clock_seconds(source: str) -> float:
    """RAW avoids WSL's observed rate adjustment; never silently fall back."""
    validate_clock_source(source)
    if source == "monotonic_raw":
        clock_id = getattr(time, "CLOCK_MONOTONIC_RAW", None)
        read_clock = getattr(time, "clock_gettime", None)
        if clock_id is None or not callable(read_clock):
            raise RuntimeError("CLOCK_MONOTONIC_RAW is unavailable on this host")
        value = read_clock(clock_id)
    else:
        value = time.monotonic()
    if not math.isfinite(value) or value < 0:
        raise RuntimeError("campaign clock returned an invalid timestamp")
    return float(value)


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


def requested_pilot_seconds(spec: RunSpec) -> int:
    value = spec.metadata.get("requested_wall_time_seconds")
    if type(value) is not int or value not in SUPPORTED_PILOT_SECONDS:
        raise ValueError("explicit supported four-hour or seven-hour wall-clock budget required")
    return value


def requested_pilot_clock(spec: RunSpec) -> str:
    return validate_clock_source(spec.metadata.get("wall_clock_source", "monotonic"))


def for_four_hour_budget(spec: RunSpec) -> RunSpec:
    if requested_pilot_seconds(spec) != 14400:
        raise ValueError("four-hour wall-clock stop required by this compatibility entry")
    return for_wall_clock_budget(spec)


def for_wall_clock_budget(spec: RunSpec) -> RunSpec:
    """Raise only the incidental transition stop; preserve the approved task.

    SDK epsilon has its own explicit transition duration. Reject implicit PER
    annealing or an evaluation stop rather than silently changing either.
    The production coordinator independently enforces the pinned requested duration.
    """
    requested_pilot_seconds(spec)
    requested_pilot_clock(spec)
    if (
        spec.distributed.epsilon_decay_transitions != 500000
        or spec.distributed.epsilon_decay_updates is not None
        or (
            spec.training.per_beta_final is not None
            and spec.training.per_beta_anneal_transitions is None
        )
        or spec.training.evaluation_stop_min_finish_rate is not None
    ):
        raise ValueError("explicit historical schedules and four-hour wall-clock stop required")
    data = spec.model_dump(mode="json")
    data["training"]["total_transitions"] = sys.maxsize
    return RunSpec.model_validate(data)
