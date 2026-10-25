"""Campaign policy preserves clock, disk reserves and learning schedules."""

import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from trackmaniarl.core.spec import RunSpec
from trackmaniarl.distributed.coordinator_rpc import epsilon
from trackmaniarl.experiments.full_qr import campaign_policy as clocks
from trackmaniarl.experiments.full_qr.campaign_budget import (
    BudgetSettings,
    CampaignBudget,
    StorageReserve,
)
from trackmaniarl.experiments.full_qr.campaign_coordinator import BudgetedCoordinator
from trackmaniarl.experiments.full_qr.campaign_launcher import _budget_deadline
from trackmaniarl.experiments.full_qr.campaign_policy import (
    campaign_storage_reserves,
    for_four_hour_budget,
    requested_pilot_clock,
)


def _raw_clock(monkeypatch: pytest.MonkeyPatch, now: list[float]) -> None:
    monkeypatch.setattr(clocks.time, "CLOCK_MONOTONIC_RAW", 4242, raising=False)

    def read(clock_id: int) -> float:
        assert clock_id == 4242
        return now[0]

    monkeypatch.setattr(clocks.time, "clock_gettime", read, raising=False)


def test_raw_budget_is_independent_of_adjusted_monotonic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = [100.0]
    _raw_clock(monkeypatch, raw)
    budget = CampaignBudget(
        BudgetSettings(
            tmp_path / "budget",
            (StorageReserve(tmp_path, 1),),
            duration_seconds=10,
            clock_source="monotonic_raw",
        ),
        free_bytes=lambda _: 100,
    )
    state = budget.start()
    try:
        assert state["started_monotonic"] == raw[0]
        assert state["clock_source"] == "monotonic_raw"
        monkeypatch.setattr(clocks.time, "monotonic", lambda: 999999.0)
        raw[0] = 109.999
        assert not budget.stop_requested()
        raw[0] = 110.0
        assert budget.stop_requested()
        # Exercise the actual coordinator boundary: it must classify this as
        # the budget limit, without consulting the unrelated monotonic domain.
        coordinator = cast(BudgetedCoordinator, SimpleNamespace(budget=budget))
        assert BudgetedCoordinator._external_stop_requested(coordinator)
        assert budget.state["stop_reason"] == "wall_clock_limit"
        assert budget.state["elapsed_seconds"] == 10
    finally:
        budget.close()


def test_raw_clock_unavailable_fails_before_creating_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(clocks.time, "CLOCK_MONOTONIC_RAW", raising=False)
    settings = BudgetSettings(
        tmp_path / "budget", (StorageReserve(tmp_path, 1),), clock_source="monotonic_raw"
    )
    budget = CampaignBudget(settings, free_bytes=lambda _: 100)
    with pytest.raises(RuntimeError, match="unavailable"):
        budget.start()
    assert not settings.directory.exists()


@pytest.mark.parametrize("value", [None, True, "utc", "", 123])
def test_clock_source_is_explicitly_validated(value: object, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="clock source"):
        requested_pilot_clock(cast(RunSpec, SimpleNamespace(metadata={"wall_clock_source": value})))
    with pytest.raises(ValueError, match="clock source"):
        replace(
            BudgetSettings(tmp_path / "budget", (StorageReserve(tmp_path, 1),)), clock_source=value
        )


def test_parent_and_learner_ready_use_the_same_declared_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _raw_clock(monkeypatch, [101.0])
    monkeypatch.setattr(clocks.time, "monotonic", lambda: 99999.0)
    ready = {"started_utc": "start", "deadline_utc": "deadline", "clock_source": "monotonic_raw"}
    budget = dict(**ready, started_monotonic=100.0, duration_seconds=25200)
    kwargs = {"duration_seconds": 25200, "clock_source": "monotonic_raw", "startup_seconds": 240}
    assert _budget_deadline(budget, ready, **kwargs) == 25300
    for field, value in (
        ("clock_source", "monotonic"),
        ("started_monotonic", 102.0),
        ("duration_seconds", 14400),
        ("deadline_utc", "wrong"),
    ):
        with pytest.raises(ValueError, match="differs"):
            _budget_deadline(dict(budget, **{field: value}), ready, **kwargs)
    with pytest.raises(ValueError, match="differs"):
        _budget_deadline(budget, dict(ready, clock_source="monotonic"), **kwargs)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_invalid_clock_samples_are_not_accepted(
    value: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    _raw_clock(monkeypatch, [value])
    with pytest.raises(RuntimeError, match="invalid timestamp"):
        clocks.clock_seconds("monotonic_raw")


def test_old_campaign_clock_default_remains_monotonic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clocks.time, "monotonic", lambda: 123.0)
    assert requested_pilot_clock(cast(RunSpec, SimpleNamespace(metadata={}))) == "monotonic"
    assert clocks.clock_seconds("monotonic") == 123.0


@pytest.mark.skipif(
    not hasattr(clocks.time, "CLOCK_MONOTONIC_RAW"), reason="Linux RAW clock required"
)
def test_real_raw_watchdog_signals_at_its_own_deadline(tmp_path: Path) -> None:
    signal = threading.Event()
    budget = CampaignBudget(
        BudgetSettings(
            tmp_path / "raw-budget",
            (StorageReserve(tmp_path, 1),),
            duration_seconds=0.05,
            check_interval_seconds=0.01,
            clock_source="monotonic_raw",
        ),
        free_bytes=lambda _: 100,
        stop_signal=signal,
    )
    state = budget.start()
    try:
        assert signal.wait(3)
    finally:
        closed = budget.close()
    assert state["clock_source"] == "monotonic_raw"
    assert closed["stop_reason"] == "wall_clock_limit"
    assert 0.05 <= closed["elapsed_seconds"] < 3


def test_output_reserves_follow_the_explicit_report_path_on_any_volume(tmp_path: Path) -> None:
    directory = tmp_path / "private"
    directory.mkdir()
    reports = tmp_path / "reports"
    reports.mkdir()
    reserves = campaign_storage_reserves(directory, reports / "new-run/artifacts")
    assert reserves[0].minimum_free_bytes == 10 * 1024**3
    assert reserves[1].path == directory
    assert reserves[2].path == reports
    assert all(reserve.minimum_free_bytes == 4 * 1024**3 for reserve in reserves[1:])
    assert not (reports / "new-run").exists()


def test_noncanonical_output_cannot_redirect_the_report_reserve(tmp_path: Path) -> None:
    aliased = tmp_path / "unused/../reports"
    with pytest.raises(ValueError, match="canonical absolute"):
        campaign_storage_reserves(tmp_path, aliased)


def specimen() -> RunSpec:
    components = {
        "learner": {"class_path": "trackmaniarl.algorithms.value_based:DiscreteValueLearner"},
        "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
        "sampler": {"class_path": "trackmaniarl.core.replay:UniformSampler"},
        "feature_pipeline": {"class_path": "trackmaniarl.core.builtins:IdentityFeaturePipeline"},
    }
    return RunSpec.model_validate(
        {
            "api_version": "2.0",
            "run_id": "fixture-no-model-or-environment",
            "components": components,
            "training": {"total_transitions": 8192000},
            "distributed": {
                "epsilon_start": 0.3,
                "epsilon_final": 0.01,
                "epsilon_decay_transitions": 500000,
            },
            "metadata": {"requested_wall_time_seconds": 14400},
        }
    )


def test_only_transition_stop_changes_and_actual_sdk_epsilon_is_identical() -> None:
    before = specimen()
    after = for_four_hour_budget(before)
    expected = before.model_dump(mode="json")
    expected["training"]["total_transitions"] = sys.maxsize
    assert after.model_dump(mode="json") == expected
    for count in (0, 10000, 250000, 500000, 8192000, 10000000):
        values = [
            epsilon(
                SimpleNamespace(
                    run=SimpleNamespace(spec=spec),
                    counters=SimpleNamespace(transitions=count, updates=count // 4),
                ),
                0,
            )
            for spec in (before, after)
        ]
        assert values[0] == values[1]
        assert before.training.replay_beta(count) == after.training.replay_beta(count)


def test_implicit_replay_annealing_cannot_change_with_the_cap() -> None:
    data = specimen().model_dump(mode="json")
    data["training"].update(beta=0.4, per_beta_final=1.0)
    with pytest.raises(ValueError, match="explicit historical schedules"):
        for_four_hour_budget(RunSpec.model_validate(data))


def test_other_exploration_schedule_or_wall_budget_is_rejected() -> None:
    for section, key, value in (
        ("distributed", "epsilon_decay_updates", 100),
        ("metadata", "requested_wall_time_seconds", 28800),
    ):
        data = specimen().model_dump(mode="json")
        data[section][key] = value
        with pytest.raises(ValueError, match="four-hour"):
            for_four_hour_budget(RunSpec.model_validate(data))
