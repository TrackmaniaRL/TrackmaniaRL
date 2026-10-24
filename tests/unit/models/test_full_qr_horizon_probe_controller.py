"""Synthetic controller checks; no native qualification or learning claim."""

import numpy as np
import pytest

from experiments.full_qr.horizon_probe_controller import HorizonProbeController
from trackmaniarl.trackmania.native_headless.full_qr.native_features import NativeGraphFrame


def frame(clock: int, x: float = 0, z: float = 0) -> NativeGraphFrame:
    channels = [0.0] * 23
    channels[0], channels[1], channels[3], channels[9] = clock, x, z, 1
    return NativeGraphFrame(tuple(channels))


def straight() -> np.ndarray:
    return np.column_stack((np.zeros(100), np.zeros(100), np.arange(100)))


def test_valid_controls_use_native_displacement_and_clock_for_braking() -> None:
    controller = HorizonProbeController(straight())
    assert controller.act(frame(0)) == 39  # neutral steering, throttle
    assert controller.act(frame(50, z=0.5)) == 37  # measured10m/s, full brake


def test_bend_generates_historical_right_steering_action() -> None:
    trajectory = np.column_stack((np.arange(100) * 0.25, np.zeros(100), np.arange(100)))
    action = HorizonProbeController(trajectory).act(frame(0))
    assert 0 <= action < 39
    assert action % 6 == 3


def test_missing_native_tick_and_reused_reset_are_rejected() -> None:
    controller = HorizonProbeController(straight())
    controller.act(frame(0))
    with pytest.raises(ValueError, match="50ms"):
        controller.act(frame(100, z=0.1))


def test_moving_or_late_first_frame_is_not_assumed_to_be_a_reset() -> None:
    with pytest.raises(ValueError, match="neutral reset"):
        HorizonProbeController(straight()).act(frame(50))


def test_actual_reset_vertical_gravity_is_not_horizontal_motion() -> None:
    channels = list(frame(0).channels)
    # Actual archived715 record:2190/2200 have identical positions;2200 has
    # visual velocity(0,-.22500000894069672,0), not a stationary total speed.
    channels[5], channels[10] = -0.22500000894069672, 0.22500000894069672
    assert HorizonProbeController(straight()).act(NativeGraphFrame(tuple(channels))) == 39


def test_horizontal_motion_at_race_zero_still_rejects_reset() -> None:
    channels = list(frame(0).channels)
    channels[4], channels[10] = 0.1, 0.1
    with pytest.raises(ValueError, match="neutral reset"):
        HorizonProbeController(straight()).act(NativeGraphFrame(tuple(channels)))


def test_offtrack_and_invalid_centerline_fail_instead_of_default_controls() -> None:
    with pytest.raises(ValueError, match="left"):
        HorizonProbeController(straight()).act(frame(0, x=20))
    invalid = straight()
    invalid[10, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        HorizonProbeController(invalid)
