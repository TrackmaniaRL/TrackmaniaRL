"""Acceleration is armed during reset, before the first positive race clock."""

from types import SimpleNamespace

import numpy as np
import pytest

from trackmaniarl.trackmania.actions import build_brake_tap_action_table
from trackmaniarl.trackmania.control import RecordingController
from trackmaniarl.trackmania.environment import OpenPlanetEnvironment
from trackmaniarl.trackmania.environment_config import TrackmaniaEnvironmentConfig
from trackmaniarl.trackmania.reward import RewardResult
from trackmaniarl.trackmania.telemetry import TelemetryFrame


class _CountdownController(RecordingController):
    def __init__(self) -> None:
        super().__init__()
        self.current = np.zeros(3, dtype=np.float32)
        self.commands: list[np.ndarray] = []

    def reset(self) -> None:
        super().reset()
        self.current.fill(0)

    def apply(self, action: np.ndarray) -> None:
        super().apply(action)
        self.current = action.copy()
        self.commands.append(action.copy())


def _environment(*, enabled: bool = True) -> OpenPlanetEnvironment:
    environment = object.__new__(OpenPlanetEnvironment)
    environment.config = TrackmaniaEnvironmentConfig(
        geometry_path="test.geometry.npz",
        countdown_throttle=enabled,
        start_poll_s=0,
        action_repeat_frames=1,
        decision_interval_ms=50,
    )
    environment.controller = _CountdownController()
    environment.evaluation_map = None
    environment._expected_map_uid = "map"
    environment._last_race_time_ms = 500.0
    environment._finish_confirmation_pending = False
    times = iter((0.0, 0.0, 10.0, 60.0))

    def read() -> TelemetryFrame:
        values = np.zeros(33, dtype=np.float32)
        values[3] = next(times)
        # Validate throttle while countdown packets are being consumed, not
        # merely after reset() has already returned control to the policy.
        if values[3] <= 10:
            np.testing.assert_array_equal(environment.controller.current, [float(enabled), 0, 0])
        return TelemetryFrame(values)

    environment.client = SimpleNamespace(read=read, close=lambda: None)
    environment._session = SimpleNamespace(
        verify_loaded_map=lambda _: None, confirm_ready=lambda _: None
    )
    environment.reward = SimpleNamespace(
        reset=lambda *args, **kwargs: None,
        step=lambda _: RewardResult(0.0, False, None),
        progress_m=0.0,
        progress_pct=0.0,
    )
    environment._action_count, environment._action_table = build_brake_tap_action_table()
    return environment


@pytest.mark.parametrize("enabled", [True, False])
def test_throttle_is_held_during_countdown_and_first_policy_action_takes_over(
    *, enabled: bool
) -> None:
    environment = _environment(enabled=enabled)
    observation, info = environment.reset()
    assert observation[3] == 10
    assert info["reset/countdown_throttle"] is enabled
    assert info["reset/race_time_ms"] == 10
    np.testing.assert_array_equal(environment.controller.current, [float(enabled), 0, 0])
    # ID 36 is neutral: there is no hidden forced throttle after reset.
    _, _, _, _, step_info = environment.step(36)
    np.testing.assert_array_equal(environment.controller.current, [0, 0, 0])
    assert step_info["control_gas"] == 0


def test_throttle_remains_held_during_readiness_and_optional_settling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment()
    environment.config = environment.config.model_copy(update={"reset_settle_s": 0.1})

    def check_throttle(*args: object) -> None:
        np.testing.assert_array_equal(environment.controller.current, [1, 0, 0])

    environment._session.confirm_ready = check_throttle
    monkeypatch.setattr("trackmaniarl.trackmania.environment.sleep", check_throttle)
    observation, _ = environment.reset()
    assert observation[3] == 60
    np.testing.assert_array_equal(environment.controller.current, [1, 0, 0])


@pytest.mark.parametrize("failure", ["read", "ready", "reward", "interrupt"])
def test_failed_reset_releases_throttle(failure: str) -> None:
    environment = _environment()

    def fail(*args: object, **kwargs: object) -> None:
        if failure == "interrupt":
            raise KeyboardInterrupt
        raise RuntimeError("reset failure")

    if failure in {"read", "interrupt"}:
        environment.client.read = fail
    elif failure == "ready":
        environment._session.confirm_ready = fail
    else:
        environment.reward.reset = fail
    with pytest.raises((RuntimeError, KeyboardInterrupt)):
        environment.reset()
    assert any(command[0] == 1 for command in environment.controller.commands)
    np.testing.assert_array_equal(environment.controller.current, [0, 0, 0])


def test_timeout_releases_before_recovery_and_arms_each_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment()
    original_read = environment.client.read
    attempts = 0

    def read() -> TelemetryFrame:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("missing frame")
        return original_read()

    def confirm_finish() -> None:
        np.testing.assert_array_equal(environment.controller.current, [0, 0, 0])

    environment.client.read = read
    environment.controller.confirm_finish = confirm_finish
    monkeypatch.setattr("trackmaniarl.trackmania.environment.sleep", lambda _: None)
    environment.reset()
    np.testing.assert_array_equal(environment.controller.commands[:2], [[1, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(environment.controller.current, [1, 0, 0])


def test_exhausted_reset_timeouts_release_throttle(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = _environment()

    def read() -> TelemetryFrame:
        raise TimeoutError("missing frame")

    environment.client.read = read
    monkeypatch.setattr("trackmaniarl.trackmania.environment.sleep", lambda _: None)
    with pytest.raises(TimeoutError, match="missing frame"):
        environment.reset()
    np.testing.assert_array_equal(environment.controller.current, [0, 0, 0])


def test_countdown_throttle_enabled_by_default() -> None:
    assert TrackmaniaEnvironmentConfig(geometry_path="test.geometry.npz").countdown_throttle is True


def test_release_failure_keeps_original_reset_error() -> None:
    environment = _environment()
    apply = environment.controller.apply

    def failing_read() -> TelemetryFrame:
        raise RuntimeError("original telemetry failure")

    def failing_release(command: np.ndarray) -> None:
        if command[0] == 0:
            raise OSError("controller disconnected")
        apply(command)

    environment.client.read = failing_read
    environment.controller.apply = failing_release
    with pytest.raises(RuntimeError, match="original telemetry failure") as captured:
        environment.reset()
    assert "controller disconnected" in captured.value.__notes__[0]
