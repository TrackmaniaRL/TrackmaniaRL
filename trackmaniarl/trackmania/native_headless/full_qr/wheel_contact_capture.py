"""Decode saved, owned native wheel contacts without assigning collision rewards."""

from dataclasses import dataclass
from math import isfinite
from struct import unpack_from

WHEEL_CONTACT_RVA = 0xAAD9F0
WHEEL_CONTACT_CALL_RVA = 0xAAD38E
RECORD_BYTES = 2048


@dataclass(frozen=True, slots=True)
class WheelContactContext:
    pid: int
    birth: int
    module_base: int
    vehicle: int
    world: int
    physical_state: int
    tracer_tid: int
    wheel_count: int


@dataclass(frozen=True, slots=True)
class WheelContactCapture:
    sequence: int
    native_time: int
    wheel_index: int
    material: int
    normal: tuple[float, float, float]
    point: tuple[float, float, float]
    contact_velocity: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class WheelContactSourceContext:
    contact: WheelContactContext
    wheel_models: tuple[int, ...]
    vehicle_model: int
    normalization_owner: int
    normalization_owner_header: bytes


@dataclass(frozen=True, slots=True)
class WheelContactSourceCapture:
    contact: WheelContactCapture
    wheel_model: int
    wheel_header: bytes
    wheel_category: int
    normalization_owner: int
    wheel_low: float
    wheel_high: float
    main_low: float
    main_high: float


def decode_wheel_contact_source(
    raw: bytes, context: WheelContactSourceContext
) -> WheelContactSourceCapture:
    """Require literal instantaneous wheel/owner reads in addition to contact proof.

    The independently bound wheel table holds 44-byte structs with four-byte
    alignment. Vehicle+88 is a different model from the producer's vehicle+8
    owner: normalization thresholds must come from the latter. This decoder
    does not establish contact coordinate parity, interval coverage or rumble.
    """
    if type(context) is not WheelContactSourceContext:
        raise ValueError("typed wheel source context required")
    contact = decode_wheel_contact(raw, context.contact)
    if (
        type(context.wheel_models) is not tuple
        or len(context.wheel_models) != context.contact.wheel_count
        or any(
            type(p) is not int or not 4096 <= p < 0x800000000000 or p % 4
            for p in context.wheel_models
        )
        or any(
            type(p) is not int or not 4096 <= p < 0x800000000000 or p % 8
            for p in (context.vehicle_model, context.normalization_owner)
        )
        or type(context.normalization_owner_header) is not bytes
        or len(context.normalization_owner_header) != 32
    ):
        raise ValueError("independently bound wheel and owner sources required")
    if (
        raw[1440:1448] != b"TMWHEEL1"
        or unpack_from("<I", raw, 928)[0] & 112 != 112
        or unpack_from("<Q", raw, 1448)[0]
        != context.contact.vehicle + 0x388 + contact.wheel_index * 8
        or unpack_from("<Q", raw, 1456)[0] != context.wheel_models[contact.wheel_index]
        or unpack_from("<I", raw, 1480)[0] != context.contact.wheel_count
        or unpack_from("<Q", raw, 1520)[0] != context.vehicle_model
        or unpack_from("<Q", raw, 1600)[0] != context.normalization_owner
        or raw[1608:1640] != context.normalization_owner_header
    ):
        raise ValueError("wheel model/normalization owner read provenance mismatch")
    wheel_high, wheel_low, main_low, main_high = unpack_from("<4f", raw, 1640)
    if not all(isfinite(v) for v in (wheel_high, wheel_low, main_low, main_high)) or not (
        0 < wheel_low < wheel_high and 0 < main_low < main_high
    ):
        raise ValueError("actual ordered normalization thresholds unavailable")
    return WheelContactSourceCapture(
        contact=contact,
        wheel_model=unpack_from("<Q", raw, 1456)[0],
        wheel_header=raw[1464:1480],
        wheel_category=unpack_from("<I", raw, 1468)[0],
        normalization_owner=context.normalization_owner,
        wheel_low=wheel_low,
        wheel_high=wheel_high,
        main_low=main_low,
        main_high=main_high,
    )


def decode_wheel_contact(raw: bytes, context: WheelContactContext) -> WheelContactCapture:
    """Validate the pinned native call and return its literal contact fields.

    The caller supplies the independently bound process, body/world state and
    tracer identity. The format also contains other execution sites: they are
    deliberately rejected here. A saved contact does not establish the Windows
    rumble notification threshold, timing, or coverage of an entire interval.
    Coordinate conventions remain those of the native argument record.
    """
    integers = (
        context.pid,
        context.birth,
        context.module_base,
        context.vehicle,
        context.world,
        context.physical_state,
        context.tracer_tid,
        context.wheel_count,
    )
    if any(type(value) is not int or value <= 0 for value in integers):
        raise ValueError("positive integral wheel contact context required")
    if not 1 <= context.wheel_count <= 16:
        raise ValueError("bounded actual wheel count required")
    if type(raw) is not bytes or len(raw) != RECORD_BYTES or raw[:8] != b"TMPI0001":
        raise ValueError("complete extended wheel contact record required")
    if unpack_from("<4I", raw, 8) != (RECORD_BYTES, 1, context.pid, context.tracer_tid):
        raise ValueError("wheel contact process/tracer identity mismatch")
    if unpack_from("<Q", raw, 848)[0] != context.birth:
        raise ValueError("wheel contact process birth mismatch")
    sequence, entry = unpack_from("<2Q", raw, 24)
    dr6, code, signal = unpack_from("<QII", raw, 48)
    native_time, kind = unpack_from("<2I", raw, 72)
    registers = unpack_from("<27Q", raw, 80)
    rdx, rsi, rdi, rip = (registers[i] for i in (12, 13, 14, 16))
    if (
        entry != context.module_base + WHEEL_CONTACT_RVA
        or rip != entry
        or kind != 0
        or dr6 & 15 != 1
        or (code, signal) != (4, 5)
        or unpack_from("<Q", raw, 304)[0] != context.module_base + WHEEL_CONTACT_CALL_RVA + 5
    ):
        raise ValueError("wheel contact execution-site mismatch")
    if (rsi, rdi, unpack_from("<Q", raw, 1288)[0]) != (
        context.vehicle,
        context.world,
        context.physical_state,
    ):
        raise ValueError("wheel contact vehicle/world/state mismatch")
    if not 1 <= sequence <= 8192 or not 2200 <= native_time <= 152200 or native_time % 10:
        raise ValueError("wheel contact bounded sequence/clock required")
    if unpack_from("<I", raw, 928)[0] & 10 != 10:
        raise ValueError("wheel contact argument reads unavailable")
    if registers[11] < 4096 or registers[11] % 8 or rdx >= context.wheel_count:
        raise ValueError("wheel contact pointer/index mismatch")
    material = raw[936 + 0x18]
    normal = unpack_from("<3f", raw, 936 + 0x1C)
    point = unpack_from("<3f", raw, 936 + 0x28)
    velocity = unpack_from("<3f", raw, 936 + 0x40)
    if material > 80 or not all(isfinite(value) for value in (*normal, *point, *velocity)):
        raise ValueError("finite literal wheel contact fields required")
    if not 0.5 <= sum(value * value for value in normal) <= 1.5:
        raise ValueError("wheel contact surface normal unavailable")
    return WheelContactCapture(sequence, native_time, rdx, material, normal, point, velocity)
