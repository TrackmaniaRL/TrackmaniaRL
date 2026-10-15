from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from trackmaniarl.trackmania.actions import BRAKE_TAP_TABLE_N_STEER
from trackmaniarl.trackmania.control import GamepadController
from trackmaniarl.trackmania.environment import TrackmaniaEnvironmentConfig
from trackmaniarl.trackmania.steering_curve import (
    INPUT_STEER_INDEX,
    SteeringCurve,
    SteeringMeasurement,
    distinct_steering_levels,
    measure_steering_curve,
)
from trackmaniarl.trackmania.telemetry import DEFAULT_TELEMETRY_FIELD_COUNT, TelemetryFrame

DEFAULT_LEVELS = np.linspace(-1.0, 1.0, BRAKE_TAP_TABLE_N_STEER)


def _steep_game_steer(stick: float) -> float:
    """A low-sensitivity pad: dead below 60% stick, then a steep power curve."""

    magnitude = max(0.0, (abs(stick) - 0.6) / 0.4) ** 3
    return float(np.copysign(magnitude, stick))


def _steep_curve() -> SteeringCurve:
    sticks = np.linspace(0.0, 1.0, 201)
    return SteeringCurve(
        tuple(float(s) for s in sticks), tuple(_steep_game_steer(float(s)) for s in sticks)
    )


def test_steep_analog_curve_collapses_default_steering_levels() -> None:
    # 0, ±1/6, ±1/3 and ±1/2 all steer straight: seven levels become one.
    assert distinct_steering_levels(_steep_curve(), DEFAULT_LEVELS) == 7


def test_inverted_curve_makes_every_steering_level_distinct() -> None:
    curve = _steep_curve()

    applied = [_steep_game_steer(curve.stick_for(float(level))) for level in DEFAULT_LEVELS]

    assert applied == pytest.approx(DEFAULT_LEVELS, abs=0.01)
    assert len(set(np.round(applied, 3))) == BRAKE_TAP_TABLE_N_STEER


def test_smallest_steer_leaves_the_dead_zone() -> None:
    curve = _steep_curve()

    assert curve.stick_for(0.0) == 0.0
    assert curve.stick_for(0.001) > 0.6
    assert curve.stick_for(-0.001) < -0.6


def test_quantized_curve_aims_for_the_middle_of_each_band() -> None:
    curve = SteeringCurve((0.0, 0.2, 0.4, 0.6, 0.8, 1.0), (0.0, 0.0, 0.5, 0.5, 1.0, 1.0))

    assert curve.stick_for(0.5) == pytest.approx(0.5)
    assert curve.stick_for(1.0) == pytest.approx(0.9)


def test_steering_curve_round_trips_through_json(tmp_path: Path) -> None:
    curve = _steep_curve()

    loaded = SteeringCurve.load(curve.save(tmp_path / "curve.json"))

    assert loaded == curve


@pytest.mark.parametrize(
    ("stick", "steer"),
    [
        ((0.0,), (0.0,)),
        ((0.1, 1.0), (0.0, 1.0)),
        ((0.0, 1.0), (0.5, 0.2)),
        ((0.0, 1.0), (0.0, 0.0)),
    ],
)
def test_steering_curve_rejects_invalid_samples(
    stick: tuple[float, ...], steer: tuple[float, ...]
) -> None:
    with pytest.raises(ValueError, match=r"steer|stick"):
        SteeringCurve(stick, steer)


class _SimulatedPad:
    """Game stand-in: telemetry reports the steer the analog curve made of the stick."""

    def __init__(self, response: float = 1.0) -> None:
        self.stick = 0.0
        self.response = response

    def apply(self, action: np.ndarray) -> None:
        self.stick = float(action[2])

    def read(self) -> TelemetryFrame:
        values = np.zeros(DEFAULT_TELEMETRY_FIELD_COUNT, dtype=np.float32)
        values[INPUT_STEER_INDEX] = self.response * _steep_game_steer(self.stick)
        return TelemetryFrame(values)


def test_measurement_recovers_the_game_curve_and_releases_the_stick() -> None:
    pad = _SimulatedPad()

    curve = measure_steering_curve(SteeringMeasurement(pad, pad, points=81))

    for level in DEFAULT_LEVELS:
        assert curve.steer_at(float(level)) == pytest.approx(
            _steep_game_steer(float(level)), abs=0.02
        )
    assert pad.stick == 0.0


def test_measurement_rejects_telemetry_that_ignores_the_stick() -> None:
    pad = _SimulatedPad(response=0.0)

    with pytest.raises(RuntimeError, match="InputSteer did not follow"):
        measure_steering_curve(SteeringMeasurement(pad, pad))
    assert pad.stick == 0.0


class _StickGamepad:
    def __init__(self) -> None:
        self.sticks: list[float] = []

    def register_notification(self, *, callback_function: object) -> None:
        del callback_function

    def right_trigger_float(self, value: float) -> None:
        del value

    def left_trigger_float(self, value: float) -> None:
        del value

    def left_joystick_float(self, x_value: float, y_value: float) -> None:
        del y_value
        self.sticks.append(x_value)

    def update(self) -> None:
        return None


def test_gamepad_sends_the_stick_that_produces_the_requested_steer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gamepad = _StickGamepad()
    module = SimpleNamespace(VX360Gamepad=lambda: gamepad)
    monkeypatch.setattr("trackmaniarl.trackmania.control._vgamepad_module", lambda: module)
    curve = _steep_curve()
    controller = GamepadController(steering_curve=curve)

    controller.apply(np.asarray([1.0, 0.0, 0.5], dtype=np.float32))
    controller.apply(np.asarray([1.0, 0.0, -1.0 / 6.0], dtype=np.float32))

    assert _steep_game_steer(gamepad.sticks[0]) == pytest.approx(0.5, abs=0.01)
    assert _steep_game_steer(gamepad.sticks[1]) == pytest.approx(-1.0 / 6.0, abs=0.01)


def test_steering_curve_requires_the_gamepad_backend() -> None:
    with pytest.raises(ValidationError, match="steering_curve_path requires the gamepad"):
        TrackmaniaEnvironmentConfig(
            geometry_path=Path("geometry.npz"),
            control_backend="keyboard",
            steering_curve_path=Path("curve.json"),
        )
