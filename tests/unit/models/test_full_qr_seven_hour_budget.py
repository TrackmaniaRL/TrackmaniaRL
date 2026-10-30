"""Seven hours must be explicit at every readiness/completion boundary."""

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from threading import Event

import pytest

from tests.unit.models.test_full_qr_campaign_launcher import evidence
from tests.unit.models.test_full_qr_campaign_policy import specimen
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.distributed.coordinator_types import CoordinatorConfig
from trackmaniarl.experiments.full_qr.campaign_budget import (
    BudgetSettings,
    CampaignBudget,
    StorageReserve,
    next_monitor_offset,
)
from trackmaniarl.experiments.full_qr.campaign_coordinator import PilotCoordinatorSettings
from trackmaniarl.experiments.full_qr.campaign_launcher import _read_ready, verify_campaign_result
from trackmaniarl.experiments.full_qr.campaign_policy import (
    for_four_hour_budget,
    for_wall_clock_budget,
    requested_pilot_seconds,
)


def test_seven_hours_preserve_all_learning_schedules() -> None:
    spec = specimen()
    spec = spec.model_copy(
        update={"metadata": {**spec.metadata, "requested_wall_time_seconds": 25200}}
    )
    result = for_wall_clock_budget(spec)
    expected = for_four_hour_budget(specimen()).model_dump(mode="json")
    expected["metadata"]["requested_wall_time_seconds"] = 25200
    assert result.model_dump(mode="json") == expected
    assert requested_pilot_seconds(result) == 25200
    with pytest.raises(ValueError, match="four-hour"):
        for_four_hour_budget(spec)


@pytest.mark.parametrize("value", [True, 25200.0, "25200", 240, 28800, None])
def test_unapproved_or_ambiguous_duration_is_rejected(value: object) -> None:
    data = specimen().model_dump(mode="json")
    data["metadata"]["requested_wall_time_seconds"] = value
    with pytest.raises(ValueError, match="explicit supported"):
        for_wall_clock_budget(RunSpec.model_validate(data))


def test_four_hours_cannot_be_reported_as_seven_hour_completion(tmp_path: Path) -> None:
    result, closure, exits = evidence(tmp_path)
    with pytest.raises(ValueError, match="worker completion"):
        verify_campaign_result(result, closure, exits, 2, duration_seconds=25200)
    result["budget"]["duration_seconds"] = 25200
    with pytest.raises(ValueError, match="worker completion"):
        verify_campaign_result(result, closure, exits, 2, duration_seconds=25200)
    result["budget"]["elapsed_seconds"] = 25200.01
    verify_campaign_result(result, closure, exits, 2, duration_seconds=25200)
    with pytest.raises(ValueError, match="worker completion"):
        verify_campaign_result(result, closure, exits, 2)


def test_ready_must_match_the_requested_seven_hour_budget(tmp_path: Path) -> None:
    path = tmp_path / "ready.json"
    identity = {"worker": {"pid": 2}}
    path.write_text(json.dumps({"pid": 2, "port": 32123, "duration_seconds": 14400}))
    with pytest.raises(ValueError, match="duration differs"):
        _read_ready(path, identity, 25200)
    path.write_text(json.dumps({"pid": 2, "port": 32123, "duration_seconds": 25200}))
    assert _read_ready(path, identity, 25200)["duration_seconds"] == 25200


def test_actual_timer_records_seven_hours_and_early_close_is_not_success(tmp_path: Path) -> None:
    signal = Event()
    settings = BudgetSettings(tmp_path / "budget", (StorageReserve(tmp_path, 1),), 25200)
    coordinator = CoordinatorConfig(
        "127.0.0.1:0", "test-token", "fingerprint", external_stop=signal
    )
    PilotCoordinatorSettings(coordinator, settings, lambda _: None)
    budget = CampaignBudget(settings, stop_signal=signal)
    state = budget.start()
    assert (
        datetime.fromisoformat(state["deadline_utc"]) - datetime.fromisoformat(state["started_utc"])
    ).total_seconds() == 25200
    closed = budget.close()
    assert closed["stop_reason"] == "launcher_closed"
    assert closed["elapsed_seconds"] < 25200
    assert not closed["final_checkpoint_verified"]
    assert signal.is_set()
    signal.clear()
    with pytest.raises(ValueError, match="25200"):
        PilotCoordinatorSettings(
            coordinator, replace(settings, duration_seconds=240), lambda _: None
        )


def test_monitor_offsets_continue_beyond_four_hours() -> None:
    assert next_monitor_offset(14399, 25200) == 14400
    assert next_monitor_offset(14400, 25200) == 16200
    assert next_monitor_offset(23400, 25200) == 25200
    assert next_monitor_offset(25200, 25200) is None
