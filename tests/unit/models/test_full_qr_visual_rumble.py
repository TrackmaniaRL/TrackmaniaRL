"""Interpolation tested with constructed physical endpoints."""

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.gameplay_rumble import (
    interpolate_visual_rumble,
)
from trackmaniarl.trackmania.native_headless.full_qr.rumble_producer import RumbleState


@pytest.mark.parametrize(
    ("weight", "frequency", "amplitude"),
    [(0.0, 4.0, 0.25), (0.25, 5.0, 0.375), (0.5, 6.0, 0.5), (1.0, 8.0, 0.75)],
)
def test_synthetic_interpolation(weight: float, frequency: float, amplitude: float) -> None:
    actual = interpolate_visual_rumble(RumbleState(4.0, 0.25), RumbleState(8.0, 0.75), weight)
    assert actual == RumbleState(frequency, amplitude)


def test_missing_physical_endpoint_is_rejected() -> None:
    with pytest.raises(ValueError, match="two actual typed"):
        interpolate_visual_rumble(None, RumbleState(1.0, 0.5), 0.5)


def test_unbounded_weight_cannot_extrapolate_a_motor_signal() -> None:
    with pytest.raises(ValueError, match="bounded physical"):
        interpolate_visual_rumble(RumbleState(1.0, 0.5), RumbleState(1.0, 0.4), 1.1)
