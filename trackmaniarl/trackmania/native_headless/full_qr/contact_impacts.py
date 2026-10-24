"""Literal Windows contact accumulator; not a collision notification source.

The pinned 140848100 prefix uses abs(dot(normal, contact velocity)), not a
change in vehicle speed. Its three channels still need independently bound
contact coordinates, wheel category, normalization and notification cadence.
"""

from dataclasses import dataclass
from math import isfinite

from trackmaniarl.trackmania.native_headless.full_qr.rumble_producer import _round_float32

# File-backed constants 141d1ecc4 and 141d1f080, source SHA3fc7d8...f6edda.
MIN_CONTACT_SPEED_SQUARED = _round_float32(1e-10)
WHEEL_NORMAL_Y_THRESHOLD = 0.25


def _f32(value: float) -> float:
    if type(value) is not float or not isfinite(value):
        raise ValueError("finite literal float contact input required")
    value = _round_float32(value)
    if not isfinite(value):
        raise ValueError("finite float32 contact input required")
    return value


@dataclass(frozen=True, slots=True)
class ImpactChannels:
    main: float
    first: float
    second: float
    main_material: int
    first_material: int
    second_material: int
    flags: int

    def __post_init__(self) -> None:
        for value in (self.main, self.first, self.second):
            if _f32(value) < 0:
                raise ValueError("nonnegative actual impact accumulator required")
        for value in (self.main_material, self.first_material, self.second_material):
            if type(value) is not int or not 0 <= value <= 80:
                raise ValueError("literal impact material required")
        if type(self.flags) is not int or not 0 <= self.flags <= 0xFFFFFFFF:
            raise ValueError("actual uint32 contact flags required")


@dataclass(frozen=True, slots=True)
class ContactImpactInput:
    normal: tuple[float, float, float]
    contact_velocity: tuple[float, float, float]
    material: int
    wheel_index: int
    wheel_category: int | None
    contact_flags: int


@dataclass(frozen=True, slots=True)
class ImpactNormalizationParameters:
    speed_scale: float
    low: float
    high: float

    def __post_init__(self) -> None:
        scale, low, high = map(_f32, (self.speed_scale, self.low, self.high))
        if scale <= 0 or not 0 < low < high:
            raise ValueError("actual positive speed scale and ordered source thresholds required")


def normalize_contact_strength(raw: float, parameters: ImpactNormalizationParameters) -> float:
    """Port the input multiplication and original 1407cc500 smoothstep helper.

    Parameters are required measured source inputs. This function does not
    derive speed scale from simulation time or invent model thresholds.
    """
    raw = _f32(raw)
    if raw < 0 or type(parameters) is not ImpactNormalizationParameters:
        raise ValueError("nonnegative raw impact and actual source parameters required")
    scale, low, high = map(_f32, (parameters.speed_scale, parameters.low, parameters.high))
    raw = _f32(raw * scale)
    floor = _f32(low * _round_float32(0.2))
    if raw > low:
        value = _f32(_f32(_f32(raw - low) * 0.5) / _f32(high - low))
        value = _f32(value + 0.5)
    else:
        value = _f32(_f32(_f32(raw - floor) * 0.5) / _f32(low - floor))
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    first = _f32(_f32(value * 3.0) * value)
    second = _f32(_f32(_f32(value + value) * value) * value)
    return _f32(first - second)


def accumulate_contact_impact(
    current: ImpactChannels,
    contact: ContactImpactInput,
) -> ImpactChannels:
    """Port only the three maxima/material writes before 140848413.

    ``wheel_index=-1`` is the source's actual non-wheel sentinel. The category
    is the real wheel object +4 word; it is never inferred from material or
    normal. Missing category fails closed when that branch would read it.
    The function neither clears frame accumulators nor produces a bool reward.
    """
    if type(current) is not ImpactChannels:
        raise ValueError("typed actual accumulator required")
    if type(contact) is not ContactImpactInput:
        raise ValueError("typed literal contact input required")
    normal, contact_velocity = contact.normal, contact.contact_velocity
    material, wheel_index = contact.material, contact.wheel_index
    wheel_category, contact_flags = contact.wheel_category, contact.contact_flags
    if type(material) is not int or not 0 <= material <= 80:
        raise ValueError("literal impact material required")
    if type(contact_flags) is not int or not 0 <= contact_flags <= 0xFFFFFFFF:
        raise ValueError("actual uint32 contact flags required")
    magnitude = contact_magnitude(normal, contact_velocity, wheel_index, wheel_category)
    if magnitude is None:
        return current
    slot, strength = magnitude
    values = list(map(_f32, (current.main, current.first, current.second)))
    materials = [current.main_material, current.first_material, current.second_material]
    # COMISS/JBE preserves the previous material on equality.
    if strength > values[slot]:
        values[slot] = strength
        materials[slot] = material
    return ImpactChannels(
        values[0],
        values[1],
        values[2],
        materials[0],
        materials[1],
        materials[2],
        current.flags | contact_flags,
    )


def contact_magnitude(  # noqa: PLR0913 - four independent physical argument fields.
    normal: tuple[float, float, float],
    contact_velocity: tuple[float, float, float],
    wheel_index: int,
    wheel_category: int | None,
) -> tuple[int, float] | None:
    """Project only the source's scalar channels, without supplying absent flags.

    Flags and materials affect other accumulator writes; this function makes
    no claim about those writes or the later motor notification producer.
    None is the source's tiny-contact-speed early return, not negative event
    coverage for an interval.
    """
    if type(normal) is not tuple or type(contact_velocity) is not tuple:
        raise ValueError("literal contact vectors required")
    if len(normal) != 3 or len(contact_velocity) != 3:
        raise ValueError("three-dimensional contact vectors required")
    nx, ny, nz = map(_f32, normal)
    vx, vy, vz = map(_f32, contact_velocity)
    if type(wheel_index) is not int or not -1 <= wheel_index < 16:
        raise ValueError("actual wheel index or non-wheel sentinel required")
    if wheel_category is not None and (
        type(wheel_category) is not int or not 0 <= wheel_category <= 0xFFFFFFFF
    ):
        raise ValueError("actual uint32 wheel category required")

    # Exact SSE order: vy*vy + vx*vx, then vz*vz. The tiny-speed early return is
    # before the source ORs contact flags or reads a wheel object.
    squared = _f32(_f32(_f32(vy * vy) + _f32(vx * vx)) + _f32(vz * vz))
    if squared < MIN_CONTACT_SPEED_SQUARED:
        return None
    strength = abs(_f32(_f32(_f32(vy * ny) + _f32(vx * nx)) + _f32(vz * nz)))
    slot = 0
    if wheel_index != -1 and abs(ny) >= WHEEL_NORMAL_Y_THRESHOLD:
        if wheel_category is None:
            raise ValueError("wheel category source unavailable")
        slot = 1 if wheel_category else 2
    return slot, strength
