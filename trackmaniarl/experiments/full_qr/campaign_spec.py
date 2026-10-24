"""Use the wall-clock watchdog without changing historical learning schedules."""

from __future__ import annotations

import sys

from trackmaniarl.core.spec import RunSpec
from trackmaniarl.experiments.full_qr.campaign_budget import SUPPORTED_PILOT_SECONDS
from trackmaniarl.experiments.full_qr.campaign_clock import validate_clock_source


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
