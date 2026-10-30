"""Synthetic boundary cases for the gameplay rumble conversion."""

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.gameplay_rumble import unity_gameplay_pulse


@pytest.mark.parametrize(
    ("frequency", "expected"), [(3.5, (0.0, 0.5)), (36.75, (0.5, 0.5)), (70.0, (0.5, 0.0))]
)
def test_synthetic_frequency_boundaries(frequency: float, expected: tuple[float, float]) -> None:
    pulse = unity_gameplay_pulse(0.5, frequency, 1.0, 100, 7)
    assert pulse is not None
    assert (pulse.first, pulse.second) == expected
    assert pulse.expiry_input_clock == 180


def test_low_intensity_produces_no_pulse() -> None:
    assert unity_gameplay_pulse(0.0, 8.0, 1.0, 100, 7) is None


def test_synthetic_quarter_frequency_and_unsigned_clock_wrap() -> None:
    pulse = unity_gameplay_pulse(0.5, 20.125, 1.0, 0xFFFFFFF0, 100)
    assert pulse is not None
    assert pulse.first == pytest.approx(0.5 * (5 / 32) / (27 / 32))
    assert pulse.second == pytest.approx(0.5)
    assert pulse.expiry_input_clock == 84


def test_unmeasured_intensity_cannot_silently_use_unity_conversion() -> None:
    with pytest.raises(ValueError, match="only measured unity"):
        unity_gameplay_pulse(0.8, 8.0, 0.5, 100, 7)


def test_nonfinite_visual_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite actual"):
        unity_gameplay_pulse(float("nan"), 8.0, 1.0, 100, 7)
