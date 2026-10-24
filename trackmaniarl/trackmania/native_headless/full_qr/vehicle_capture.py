"""Strict decoder for experimental vendor-shadow vehicle captures.

Decoding saved bytes does not authorize a live process or qualify a full
training observation. The native producer writes its observer-owned cache;
the participant cache is preserved separately to expose stale API values.
"""

from dataclasses import dataclass
from math import isfinite
from struct import unpack_from

PRODUCER_RVA = 0x9D6AD0
RECORD_SIZE = 1280


@dataclass(frozen=True, slots=True)
class VehicleCaptureContext:
    pid: int
    participant: int
    vehicle: int
    module_base: int
    native_t: int


@dataclass(frozen=True, slots=True)
class VehicleCapture:
    native_t: int
    sequence: int
    delta_native: int
    native_cache: bytes
    visual_state: bytes
    participant_cache: bytes
    global_owner: int

    @property
    def skidding_count(self) -> int:
        _decoded_scalar: int = unpack_from("<I", self.native_cache, 0x4C)[0]
        return _decoded_scalar

    @property
    def flying_duration(self) -> int:
        _decoded_scalar: int = unpack_from("<I", self.native_cache, 0x5C)[0]
        return _decoded_scalar

    @property
    def slip_connect_order(self) -> tuple[float, ...]:
        return tuple(unpack_from("<f", self.visual_state, 0xBC + i * 0x2C)[0] for i in (0, 1, 3, 2))

    @property
    def ground_contact_materials(self) -> tuple[int, ...]:
        """Actual registered per-wheel getters, preserving native indices0..3.

        Pinned ELF9de9e0/9dea10/9dea40/9dea70 and PE728250 use the same
        visual fields.13 and80 are original getter branches, not invented
        contact defaults. This does not choose the rumble producer's material.
        """
        if len(self.visual_state) != 864:
            raise ValueError("complete actual visual source required")
        result = []
        for wheel in range(4):
            offset = wheel * 0x2C
            flags = unpack_from("<I", self.visual_state, 0xD0 + offset)[0]
            material = 13 if flags & 2 else self.visual_state[0xB8 + offset] if flags & 4 else 80
            if material > 80:
                raise ValueError("native ground material outside registered domain")
            result.append(material)
        return tuple(result)


def decode_vehicle_capture(raw: bytes, context: VehicleCaptureContext) -> VehicleCapture:
    """Bind a complete record to an independently supplied saved context."""
    if type(raw) is not bytes or len(raw) != RECORD_SIZE or raw[:8] != b"TMVH0001":
        raise ValueError("exact TMVH0001 vehicle record required")
    length, version, pid, tid = unpack_from("<4I", raw, 8)
    sequence, native_t, delta = unpack_from("<QII", raw, 24)
    participant, vehicle, global_owner = unpack_from("<3Q", raw, 40)
    if (length, version, pid, tid) != (RECORD_SIZE, 1, context.pid, context.pid):
        raise ValueError("vehicle record header/owner mismatch")
    if (participant, vehicle, native_t) != (context.participant, context.vehicle, context.native_t):
        raise ValueError("vehicle record context mismatch")
    if not 1 <= sequence <= 16200 or delta != (0 if sequence == 1 else 10):
        raise ValueError("vehicle record sequence/delta mismatch")
    if not 2190 <= native_t <= 162200 or native_t % 10:
        raise ValueError("vehicle clock outside experimental domain")
    if native_t != 2190 + (sequence - 1) * 10:
        raise ValueError("vehicle record missing native interval")
    if unpack_from("<I", raw, 204)[0] != 1:
        raise ValueError("vendor cache call proof missing")
    if raw[1212:1240] != bytes(28) or raw[1244:1248] != bytes(4):
        raise ValueError("vehicle record reserved bytes")
    if unpack_from("<I", raw, 1240)[0] != PRODUCER_RVA:
        raise ValueError("unexpected vendor cache producer")
    if unpack_from("<4I", raw, 1248) != (native_t,) * 4:
        raise ValueError("vehicle record clock changed during capture")
    if unpack_from("<2Q", raw, 1264) != (vehicle + 0x848, context.module_base + PRODUCER_RVA):
        raise ValueError("vehicle record source/producer address mismatch")
    visual = raw[208:1072]
    cache = raw[64:204]
    if unpack_from("<I", visual, 12)[0] != native_t:
        raise ValueError("stale visual state")
    slips = tuple(unpack_from("<f", visual, 0xBC + i * 0x2C)[0] for i in range(4))
    matrix = unpack_from("<9f", visual, 0x2C)
    rpm = unpack_from("<f", cache, 0x38)[0]
    if not all(isfinite(value) for value in (*slips, *matrix, rpm)):
        raise ValueError("nonfinite vehicle telemetry")
    if unpack_from("<I", cache, 0x4C)[0] > 4:
        raise ValueError("vendor skidding count outside four-wheel domain")
    if cache[:12] != visual[0x50:0x5C] or cache[0x38:0x40] != (
        visual[0x198:0x19C] + visual[0x1A4:0x1A8]
    ):
        raise ValueError("vendor cache does not match source position/rpm/gear")
    return VehicleCapture(native_t, sequence, delta, cache, visual, raw[1072:1212], global_owner)
