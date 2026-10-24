"""Pinned gameplay conversion for the measured Windows intensity setting 1.0.

This converts supplied visual channels, not contacts. Physical production,
interpolation, queue clocks and notification scheduling remain caller inputs.
"""

from math import isfinite
from struct import unpack_from

from trackmaniarl.trackmania.native_headless.full_qr.rumble_producer import RumbleState
from trackmaniarl.trackmania.native_headless.full_qr.rumble_producer import _round_float32 as f32
from trackmaniarl.trackmania.native_headless.full_qr.rumble_queue import RumblePulse


def interpolate_visual_rumble(
    previous: RumbleState, current: RumbleState, weight: float
) -> RumbleState:
    """Port 140729094..1407290dc, preserving weighted float32 operation order.

    The caller supplies actual endpoints and the render interpolation weight.
    This function does not infer either endpoint or assign a render clock.
    """
    if type(previous) is not RumbleState or type(current) is not RumbleState:
        raise ValueError("two actual typed physical rumble endpoints required")
    values = (previous.frequency, previous.amplitude, current.frequency, current.amplitude, weight)
    if any(type(v) is not float or not isfinite(v) for v in values):
        raise ValueError("finite actual interpolation inputs required")
    pf, pa, cf, ca, alpha = map(f32, values)
    if not all(isfinite(v) for v in (pf, pa, cf, ca, alpha)):
        raise ValueError("finite float32 interpolation inputs required")
    if min(pf, cf) < 0 or not all(0 <= v <= 1 for v in (pa, ca, alpha)):
        raise ValueError("bounded physical rumble endpoints and interpolation weight required")
    complement = f32(1.0 - alpha)
    return RumbleState(
        f32(f32(complement * pf) + f32(alpha * cf)),
        f32(f32(complement * pa) + f32(alpha * ca)),
    )


def decode_device_queue(raw: bytes) -> tuple[RumblePulse, ...]:
    """Decode one complete actual device+40 queue; unused slots may be stale."""
    if type(raw) is not bytes or len(raw) != 196:
        raise ValueError("complete actual device queue required")
    count = unpack_from("<I", raw)[0]
    if count > 16:
        raise ValueError("actual device queue count exceeds source capacity")
    pulses = tuple(RumblePulse(*unpack_from("<Iff", raw, 4 + i * 12)) for i in range(count))
    if any(not isfinite(v) for pulse in pulses for v in (pulse.first, pulse.second)):
        raise ValueError("finite actual queued channels required")
    return pulses


def unity_gameplay_pulse(  # noqa: PLR0913 - Explicit measured channels, setting and clock domain.
    amplitude: float,
    frequency: float,
    setting: float,
    input_clock: int,
    minimum_duration: int,
) -> RumblePulse | None:
    """Port 1402b1bb0/1402b19c0 at the independently measured unity setting.

    Other settings require the original powf path and are rejected explicitly.
    Input-clock wrap is the original unsigned queue representation.
    """
    if any(type(v) is not float or not isfinite(v) for v in (amplitude, frequency, setting)):
        raise ValueError("finite actual visual channels and input-port setting required")
    amplitude, frequency, setting = map(f32, (amplitude, frequency, setting))
    if not all(isfinite(v) for v in (amplitude, frequency, setting)):
        raise ValueError("finite float32 source required")
    if setting != 1.0:
        raise ValueError("only measured unity input-port setting is supported")
    if any(type(v) is not int or not 0 <= v <= 0xFFFFFFFF for v in (input_clock, minimum_duration)):
        raise ValueError("actual unsigned input clock and minimum duration required")
    if amplitude <= f32(0.01):
        return None
    if frequency <= 3.5:
        weight = 0.0
    elif frequency >= 70.0:
        weight = 1.0
    else:
        value = f32(f32(frequency - 3.5) / 66.5)
        first = f32(f32(value * 3.0) * value)
        second = f32(f32(f32(value + value) * value) * value)
        weight = f32(first - second)
    low = f32(1.0 - weight)
    maximum = max(weight, low)
    high_channel = f32(f32(weight * amplitude) / maximum)
    low_channel = f32(f32(low * amplitude) / maximum)
    return RumblePulse(
        (input_clock + max(80, minimum_duration)) & 0xFFFFFFFF,
        high_channel,
        low_channel,
    )
