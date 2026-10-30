"""Readiness I/O must not make the first decision use an older reset frame."""

from types import SimpleNamespace

import numpy as np
import pytest

from trackmaniarl.trackmania.environment import OpenPlanetEnvironment
from trackmaniarl.trackmania.telemetry import TelemetryFrame


def reset_environment(*, clock: float = 40, finished: bool = False) -> OpenPlanetEnvironment:
    environment = OpenPlanetEnvironment.__new__(OpenPlanetEnvironment)
    old = np.zeros(33, dtype=np.float32)
    old[3], old[4] = 10, 1
    fresh = old.copy()
    fresh[2], fresh[3], fresh[4] = finished, clock, 2
    calls = []
    environment.config = SimpleNamespace(
        reset_settle_s=0,
        position_indices=(4, 5, 6),
        velocity_indices=(7, 8, 9),
        countdown_throttle=False,
    )
    environment.evaluation_map = None
    environment._expected_map_uid = "test"
    environment._session = SimpleNamespace(verify_loaded_map=lambda _: None)
    environment._restart_race = lambda: TelemetryFrame(old)
    environment.client = SimpleNamespace(read=lambda: TelemetryFrame(fresh))
    environment.controller = SimpleNamespace(apply=lambda _: None)
    environment.reward = SimpleNamespace(reset=lambda *args, **kwargs: calls.append((args, kwargs)))
    environment._test_reward_calls = calls
    return environment


def test_reset_rewards_and_clock_use_frame_after_confirmation_even_without_settle() -> None:
    environment = reset_environment()
    observation, info = environment.reset()
    assert observation[3] == 40
    assert observation[4] == 2
    assert environment._last_race_time_ms == 40
    assert environment._test_reward_calls[0][1]["race_time_ms"] == 40
    assert info["reset_confirmation_race_time_ms"] == 10
    assert info["reset_observation_race_time_ms"] == 40


@pytest.mark.parametrize(("clock", "finished"), [(0, False), (40, True)])
def test_changed_race_or_finished_fresh_frame_cannot_fall_back_to_old_state(
    clock: float, *, finished: bool
) -> None:
    environment = reset_environment(clock=clock, finished=finished)
    with pytest.raises(RuntimeError, match="Race changed or finished"):
        environment.reset()
    assert not environment._test_reward_calls
