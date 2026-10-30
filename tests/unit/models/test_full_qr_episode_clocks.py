"""Clock accounting only; synthetic fixtures are not native qualification."""

import pytest

from tests.unit.models.test_full_qr_environment import setup
from trackmaniarl.distributed.actor_metrics import _outcome_summary


def test_host_delay_is_not_reported_as_extra_physics(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter((100.0, 100.7, 120.0))
    monkeypatch.setattr(
        "trackmaniarl.trackmania.native_headless.full_qr.environment.monotonic", lambda: next(ticks)
    )
    environment, _, _ = setup()
    environment.reset()
    first = environment.step(0)[4]
    second = environment.step(0)[4]

    assert first["episode_elapsed_s"] == pytest.approx(0.7)
    assert second["episode_elapsed_s"] == pytest.approx(20.0)
    assert first["step_race_time_ms"] == second["step_race_time_ms"] == 50
    assert first["observed_physics_time_ms"] == 50
    assert second["observed_physics_time_ms"] == 100
    summary = _outcome_summary(second, "time_limit")
    assert summary["timing/episode_started_monotonic"] == 100
    assert summary["timing/episode_observed_monotonic"] == 120
    assert summary["headless/environment_episode"] == second["episode"]
    assert summary["duration_s"] == 20
    assert summary["race_time_s"] == 0.1
    environment.close()


def test_missing_clock_witnesses_are_not_filled_with_zero() -> None:
    summary = _outcome_summary({"episode_elapsed_s": 5}, "time_limit")
    assert "timing/episode_started_monotonic" not in summary
    assert "timing/episode_observed_monotonic" not in summary


def test_reset_restarts_host_clock_without_reusing_episode_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter((100.0, 103.0, 900.0, 901.0))
    monkeypatch.setattr(
        "trackmaniarl.trackmania.native_headless.full_qr.environment.monotonic", lambda: next(ticks)
    )
    environment, _, _ = setup()
    environment.reset()
    first = environment.step(0)[4]
    environment.reset()
    second = environment.step(0)[4]

    assert first["episode"] != second["episode"]
    assert first["episode_elapsed_s"] == 3
    assert second["episode_elapsed_s"] == 1
    assert first["step_race_time_ms"] == second["step_race_time_ms"] == 50
    environment.close()
