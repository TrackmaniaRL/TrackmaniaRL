"""Decode the fast native delivery observer, not a motor/Finish event stream.

The enclosing owned-session manifest must bind the file to its process birth
and observer build; these records carry PID/TID, not a process-birth stamp.
The independent context comes from held engine reads outside the hook.
"""

from dataclasses import dataclass
from math import isfinite
from struct import unpack_from

from trackmaniarl.trackmania.native_headless.full_qr.prepared_contact_capture import (
    PreparedContactCapture,
)


@dataclass(frozen=True, slots=True)
class PreparedImpactContext:
    pid: int
    tid: int
    module_base: int
    vehicle: int
    participant: int
    sim: int
    controller: int
    normalization_owner: int
    normalization_header: bytes
    normalization_thresholds: bytes
    wheel_models: tuple[int, ...]
    wheel_headers: tuple[bytes, ...]


def decode_prepared_impact(raw: bytes, context: PreparedImpactContext) -> PreparedContactCapture:
    if type(context) is not PreparedImpactContext:
        raise ValueError("independently bound native delivery context required")
    if (
        any(
            type(x) is not int or x <= 0
            for x in (
                context.pid,
                context.tid,
                context.module_base,
                context.vehicle,
                context.participant,
                context.sim,
                context.controller,
                context.normalization_owner,
            )
        )
        or type(context.normalization_header) is not bytes
        or len(context.normalization_header) != 32
        or type(context.normalization_thresholds) is not bytes
        or len(context.normalization_thresholds) != 16
        or type(context.wheel_models) is not tuple
        or len(context.wheel_models) != 4
        or any(type(x) is not int or x < 4096 or x % 4 for x in context.wheel_models)
        or type(context.wheel_headers) is not tuple
        or len(context.wheel_headers) != 4
        or any(type(x) is not bytes or len(x) != 16 for x in context.wheel_headers)
    ):
        raise ValueError("complete independent native delivery binding required")
    if type(raw) is not bytes or len(raw) != 512 or raw[:8] != b"TMIH0001":
        raise ValueError("complete native delivery record required")

    def q(offset: int) -> int:
        _decoded_scalar: int = unpack_from("<Q", raw, offset)[0]
        return _decoded_scalar

    def u(offset: int) -> int:
        _decoded_scalar: int = unpack_from("<I", raw, offset)[0]
        return _decoded_scalar

    native_time, event_time = u(36), u(40)
    if (
        unpack_from("<4I", raw, 8) != (512, 1, context.pid, context.tid)
        or not 1 <= q(24) <= 100_000
        or u(32) != 0
        or not 2190 <= native_time <= 162_200
        or native_time % 10
        or not 2190 <= event_time <= 162_200
        or q(80) != event_time
        or q(48) != context.vehicle
        or q(56) != context.module_base + 0x9CE395
        or q(64) != context.module_base + 0xAACC92
        or q(72) < 4096
        or q(72) % 8
        or q(144) != context.normalization_owner
        or raw[152:184] != context.normalization_header
        or raw[184:200] != context.normalization_thresholds
        or q(224) != context.participant
        or q(232) != context.sim
        or q(240) != context.controller
        or (u(248), u(252)) != (31, 4)
        or q(256) != context.module_base
        or q(264) != context.sim + 0x48
        or raw[320:] != bytes(192)
    ):
        raise ValueError("native delivery identity, caller or read provenance mismatch")
    wheel = unpack_from("<i", raw, 44)[0]
    if not -1 <= wheel < 4:
        raise ValueError("native delivery wheel outside bound table")
    if wheel == -1:
        if raw[200:224] != bytes(24):
            raise ValueError("nonwheel contact must not contain fabricated wheel metadata")
        category = None
    else:
        if q(200) != context.wheel_models[wheel] or raw[208:224] != context.wheel_headers[wheel]:
            raise ValueError("native delivery wheel binding mismatch")
        category = u(212)
    normal = unpack_from("<3f", raw, 100)
    point = unpack_from("<3f", raw, 112)
    velocity = unpack_from("<3f", raw, 124)
    thresholds = unpack_from("<4f", raw, 184)
    physical = unpack_from("<10f", raw, 272)
    if (
        raw[96] > 80
        or not all(isfinite(x) for x in (*normal, *point, *velocity, *thresholds, *physical))
        or not 0.5 <= sum(x * x for x in normal) <= 1.5
        or not (0 < thresholds[1] < thresholds[0] and 0 < thresholds[2] < thresholds[3])
    ):
        raise ValueError("finite prepared vectors, physical state and ordered thresholds required")
    return PreparedContactCapture(
        q(24),
        native_time,
        event_time,
        wheel,
        category,
        raw[96],
        normal,
        point,
        velocity,
        thresholds,
    )
