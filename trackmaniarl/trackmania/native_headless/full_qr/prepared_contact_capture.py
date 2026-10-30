"""Literal prepared contact arguments, distinct from wheel-entry arguments.

The measured server omits one accumulator call but retains its prepared input
for the following delivery stub. That stub's R9 is scratch, NOT contact flags.
This decoder intentionally supplies neither flags nor a collision label.
"""

from dataclasses import dataclass
from math import isfinite
from struct import unpack_from


@dataclass(frozen=True, slots=True)
class PreparedContactContext:
    pid: int
    birth: int
    module_base: int
    vehicle: int
    physical_state: int
    tracer_tid: int
    normalization_owner: int
    normalization_header: bytes
    normalization_thresholds: bytes
    wheel_models: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PreparedContactCapture:
    sequence: int
    native_time: int
    event_time: int
    wheel_index: int
    wheel_category: int | None
    material: int
    normal: tuple[float, float, float]
    point: tuple[float, float, float]
    contact_velocity: tuple[float, float, float]
    normalization_thresholds: tuple[float, float, float, float]


def decode_prepared_contact(raw: bytes, context: PreparedContactContext) -> PreparedContactCapture:
    if type(context) is not PreparedContactContext:
        raise ValueError("typed independently bound prepared contact context required")
    if (
        any(
            type(x) is not int or x <= 0
            for x in (
                context.pid,
                context.birth,
                context.module_base,
                context.vehicle,
                context.physical_state,
                context.tracer_tid,
                context.normalization_owner,
            )
        )
        or type(context.wheel_models) is not tuple
        or not 1 <= len(context.wheel_models) <= 16
        or any(type(x) is not int or x < 4096 or x % 4 for x in context.wheel_models)
        or type(context.normalization_header) is not bytes
        or len(context.normalization_header) != 32
        or type(context.normalization_thresholds) is not bytes
        or len(context.normalization_thresholds) != 16
    ):
        raise ValueError("complete independent prepared contact source binding required")
    if type(raw) is not bytes or len(raw) != 2048 or raw[:8] != b"TMIA0001":
        raise ValueError("complete prepared contact record required")

    def q(offset: int) -> int:
        _decoded_scalar: int = unpack_from("<Q", raw, offset)[0]
        return _decoded_scalar

    def u(offset: int) -> int:
        _decoded_scalar: int = unpack_from("<I", raw, offset)[0]
        return _decoded_scalar

    registers = unpack_from("<27Q", raw, 80)
    sequence, native_time = q(24), u(72)
    if (
        unpack_from("<4I", raw, 8) != (2048, 1, context.pid, context.tracer_tid)
        or q(848) != context.birth
        or q(32) != context.module_base + 0x9CE390
        or registers[16] != q(32)
        or q(304) != context.module_base + 0xAACC92
        or q(48) & 15 != 1
        or unpack_from("<2I", raw, 56) != (4, 5)
        or u(76) != 0
        or not 1 <= sequence <= 8192
        or not 2200 <= native_time <= 152200
        or native_time % 10
        or registers[13] != context.vehicle
        or q(1288) != context.physical_state
        or u(928) & 196 != 196
        or registers[12] < 4096
        or registers[12] % 8
        or u(1732) != registers[11] & 0xFFFFFFFF
        or raw[768:816] != raw[1680:1728]
        or q(1600) != context.normalization_owner
        or raw[1608:1640] != context.normalization_header
        or raw[1640:1656] != context.normalization_thresholds
    ):
        raise ValueError("prepared contact identity, caller or read provenance mismatch")
    wheel = unpack_from("<i", raw, 1732)[0]
    if wheel < -1 or wheel >= len(context.wheel_models):
        raise ValueError("prepared contact wheel index outside bound table")
    category = None
    if wheel != -1:
        if (
            raw[1440:1448] != b"TMWHEEL1"
            or not u(928) & 16
            or q(1448) != context.vehicle + 0x388 + wheel * 8
            or q(1456) != context.wheel_models[wheel]
            or u(1480) != len(context.wheel_models)
        ):
            raise ValueError("prepared contact instantaneous wheel provenance mismatch")
        category = u(1468)
    event_time = u(1740)
    if event_time != registers[14] or not 2200 <= event_time <= 152200:
        raise ValueError("actual prepared event time unavailable")
    normal = unpack_from("<3f", raw, 1684)
    point = unpack_from("<3f", raw, 1696)
    velocity = unpack_from("<3f", raw, 1708)
    thresholds = unpack_from("<4f", raw, 1640)
    if (
        raw[1680] > 80
        or not all(isfinite(v) for v in (*normal, *point, *velocity, *thresholds))
        or not 0.5 <= sum(v * v for v in normal) <= 1.5
        or not (0 < thresholds[1] < thresholds[0] and 0 < thresholds[2] < thresholds[3])
    ):
        raise ValueError("finite actual prepared vectors and ordered thresholds required")
    return PreparedContactCapture(
        sequence,
        native_time,
        event_time,
        wheel,
        category,
        raw[1680],
        normal,
        point,
        velocity,
        thresholds,
    )
