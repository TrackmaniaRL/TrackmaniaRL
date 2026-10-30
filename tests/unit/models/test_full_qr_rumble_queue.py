"""Expiry/combination regressions confirmed against the original Windows evaluator."""

import math
import struct

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.rumble_queue import (
    RumblePulse,
    evaluate_rumble_queue,
)


def test_expiry_is_strict_and_zero_is_indefinite() -> None:
    pulses = [RumblePulse(0, 0.2, 0.1), RumblePulse(10, 0.8, 0.9)]
    assert len(evaluate_rumble_queue(pulses, 10).retained) == 2
    assert evaluate_rumble_queue(pulses, 11).retained == (
        evaluate_rumble_queue(pulses[:1], 11).retained[0],
    )


def test_removal_preserves_order_and_does_not_skip_adjacent_expired_entries() -> None:
    pulses = [RumblePulse(t, float(t), 0.0) for t in (2, 3, 12, 4, 13)]
    result = evaluate_rumble_queue(pulses, 10)
    assert [pulse.expiry_input_clock for pulse in result.retained] == [12, 13]
    assert result.channels == (1.0, 0.0)


def test_channels_take_independent_maxima_rather_than_sum() -> None:
    result = evaluate_rumble_queue([RumblePulse(0, 0.25, 0.75), RumblePulse(0, 0.5, 0.125)], 100)
    assert result.channels == (0.5, 0.75)


def test_unsigned_clock_comparison_does_not_infer_wraparound() -> None:
    pulse = RumblePulse(1, 1.0, 1.0)
    assert evaluate_rumble_queue([pulse], 0xFFFFFFFF).channels == (0.0, 0.0)
    assert evaluate_rumble_queue([RumblePulse(0xFFFFFFFF, 1.0, 1.0)], 1).channels == (1.0, 1.0)


def test_canonical_float32_and_negative_zero() -> None:
    result = evaluate_rumble_queue([RumblePulse(0, 0.1, -0.0)], 0)
    assert result.retained[0].first == struct.unpack("<f", struct.pack("<f", 0.1))[0]
    assert struct.pack("<ff", *result.channels) == struct.pack("<ff", result.retained[0].first, 0.0)


@pytest.mark.parametrize("clock", [-1, 2**32, True, 1.0])
def test_invalid_clock_rejected(clock: int | float) -> None:
    with pytest.raises(ValueError, match="unsigned 32-bit"):
        evaluate_rumble_queue([], clock)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, 1e100, True, 1])
def test_invalid_channel_rejected_even_if_pulse_expired(value: float) -> None:
    with pytest.raises(ValueError, match="finite float32"):
        evaluate_rumble_queue([RumblePulse(1, value, 0.0)], 2)


def test_capacity_and_empty_output() -> None:
    assert evaluate_rumble_queue([], 123).channels == (0.0, 0.0)
    assert len(evaluate_rumble_queue([RumblePulse(0, 0.0, 0.0)] * 16, 0).retained) == 16
    with pytest.raises(ValueError, match="at most 16"):
        evaluate_rumble_queue([RumblePulse(0, 0.0, 0.0)] * 17, 0)
