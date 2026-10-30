"""Staged Windows queue evaluation; this does not derive collision pulses."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from struct import pack, unpack


def _clock(value: int) -> None:
    if type(value) is not int or not 0 <= value <= 0xFFFFFFFF:
        raise ValueError("unsigned 32-bit input clock required")


def _channel(value: float) -> float:
    if type(value) is not float or not isfinite(value):
        raise ValueError("finite float32 queue channel required")
    try:
        _decoded_scalar: float = unpack("<f", pack("<f", value))[0]
        return _decoded_scalar
    except OverflowError as exc:
        raise ValueError("finite float32 queue channel required") from exc


@dataclass(frozen=True)
class RumblePulse:
    expiry_input_clock: int
    first: float
    second: float


@dataclass(frozen=True)
class RumbleEvaluation:
    retained: tuple[RumblePulse, ...]
    channels: tuple[float, float]


def evaluate_rumble_queue(pulses: Sequence[RumblePulse], input_clock: int) -> RumbleEvaluation:
    """Match 1402bf660, using its explicit input clock and unsigned comparison.

    Expiry 0 is indefinite; equality with the current clock is still active.
    Channels combine by independent maxima, then clamp to [0, 1]. A caller must
    qualify the source pulses and clock before using this in an RL transition.
    """
    _clock(input_clock)
    if len(pulses) > 16:
        raise ValueError("at most 16 native queue entries required")
    retained = []
    first = second = 0.0
    for pulse in pulses:
        _clock(pulse.expiry_input_clock)
        canonical = RumblePulse(
            pulse.expiry_input_clock, _channel(pulse.first), _channel(pulse.second)
        )
        if canonical.expiry_input_clock and canonical.expiry_input_clock < input_clock:
            continue
        retained.append(canonical)
        first = max(first, canonical.first)
        second = max(second, canonical.second)
    return RumbleEvaluation(tuple(retained), (min(first, 1.0), min(second, 1.0)))
