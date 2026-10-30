"""Staged pieces of the Windows producer; no physical pulse source is inferred."""

from dataclasses import dataclass
from math import copysign, inf, isfinite
from struct import pack, unpack


def _round_float32(value: float) -> float:
    try:
        _decoded_scalar: float = unpack("<f", pack("<f", value))[0]
        return _decoded_scalar
    except OverflowError:
        return copysign(inf, value)


def approach_rumble_channel(current: float, target: float, max_delta: float) -> float:
    """Port the pinned 1407cd990 helper, preserving each float32 operation.

    This only approaches a supplied target. The producer's real target inputs,
    update cadence and physical source must be established before integration.
    """
    values = (current, target, max_delta)
    if any(type(value) is not float or not isfinite(value) for value in values):
        raise ValueError("finite float32 producer inputs required")
    current, target, max_delta = (_round_float32(value) for value in values)
    if not all(isfinite(value) for value in (current, target, max_delta)) or max_delta < 0:
        raise ValueError("finite float32 inputs and nonnegative producer delta required")
    if current > _round_float32(target + max_delta):
        return _round_float32(current - max_delta)
    if _round_float32(target - max_delta) > current:
        return _round_float32(current + max_delta)
    return target


@dataclass(frozen=True, slots=True)
class RumbleState:
    frequency: float
    amplitude: float


@dataclass(frozen=True, slots=True)
class SurfaceRumbleInput:
    longitudinal_speed: float
    surface_id: int
    enabled: bool
    amplitude_sensitivity: float
    amplitude_cap: float
    frequency_divisor: float
    amplitude_gain: float


def update_surface_rumble(current: RumbleState, surface: SurfaceRumbleInput) -> RumbleState:
    """Port 1407ced08..1407ced77 after explicit material selection.

    The caller must bind the material table and real source values. In
    particular, this function does not select a material from skidding or speed.
    """
    if type(surface.surface_id) is not int or not 0 <= surface.surface_id <= 80:
        raise ValueError("explicit native surface id required")
    if type(surface.enabled) is not bool:
        raise ValueError("explicit native surface enable flag required")
    values = (
        current.frequency,
        current.amplitude,
        surface.longitudinal_speed,
        surface.amplitude_sensitivity,
        surface.amplitude_cap,
        surface.frequency_divisor,
        surface.amplitude_gain,
    )
    if any(type(value) is not float or not isfinite(value) for value in values):
        raise ValueError("finite float32 surface inputs required")
    frequency, amplitude, speed, sensitivity, cap, divisor, gain = map(_round_float32, values)
    if (
        not all(
            isfinite(value)
            for value in (frequency, amplitude, speed, sensitivity, cap, divisor, gain)
        )
        or min(frequency, amplitude, sensitivity, cap, gain) < 0
        or amplitude > 1
        or divisor <= 0
    ):
        raise ValueError("bounded float32 surface inputs required")
    if surface.surface_id == 80:
        return RumbleState(frequency, amplitude)
    magnitude = abs(speed)
    target_amplitude = min(_round_float32(magnitude * sensitivity), cap) if surface.enabled else 0.0
    target_amplitude = min(_round_float32(target_amplitude * gain), _round_float32(0.15))
    return RumbleState(
        approach_rumble_channel(frequency, _round_float32(magnitude / divisor), 0.5),
        approach_rumble_channel(amplitude, target_amplitude, _round_float32(0.05)),
    )


def produce_rumble(
    current: RumbleState,
    surface: SurfaceRumbleInput,
    normalized_impacts: tuple[float, float, float],
) -> RumbleState:
    """Apply both staged blocks; this is not a qualified physical signal source."""
    return combine_contact_rumble(update_surface_rumble(current, surface), normalized_impacts)


def combine_contact_rumble(
    current: RumbleState, normalized_impacts: tuple[float, float, float]
) -> RumbleState:
    """Port 1407ced77..1407cee37; inputs are not inferred from velocity changes.

    Impact order is the normalized +204, +208 and +20c accumulators. Their real
    producer, normalization and delivery remain separate qualification work.
    The frequency branches deliberately follow the original instructions.
    """
    if len(normalized_impacts) != 3:
        raise ValueError("three normalized contact inputs required")
    values = (current.frequency, current.amplitude, *normalized_impacts)
    if any(type(value) is not float or not isfinite(value) for value in values):
        raise ValueError("finite float32 contact inputs required")
    frequency, amplitude, main, first, second = map(_round_float32, values)
    if (
        not isfinite(frequency)
        or frequency < 0
        or not 0 <= amplitude <= 1
        or any(not 0 <= value <= 1 for value in (main, first, second))
    ):
        raise ValueError("bounded normalized contact inputs required")
    mix = _round_float32(
        _round_float32(second * _round_float32(0.45)) + _round_float32(first * _round_float32(0.65))
    )
    strength = max(main, mix)
    if strength < 0.5:
        return RumbleState(frequency, amplitude)
    excess = _round_float32(strength - 0.5)
    doubled = _round_float32(excess + excess)
    target_frequency = _round_float32(4.0 - _round_float32(doubled + doubled))
    if target_frequency < 4.0:
        target_frequency = 4.0
    elif target_frequency > 2.0:
        target_frequency = 2.0
    scaled = _round_float32(excess * _round_float32(0.33999997))
    target_amplitude = _round_float32(_round_float32(scaled + scaled) + _round_float32(0.66))
    target_amplitude = min(1.0, max(_round_float32(0.66), target_amplitude))
    return RumbleState(max(frequency, target_frequency), max(amplitude, target_amplitude))
