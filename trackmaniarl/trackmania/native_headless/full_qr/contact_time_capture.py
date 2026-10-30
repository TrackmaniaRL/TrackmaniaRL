"""Join saved native contact-time execution and append records.

This proves the witnessed calculation path, not client UI or physical crossing.
It does not authorize a live process or supply a training terminal flag.
"""

from dataclasses import dataclass
from math import isfinite, trunc
from struct import pack, unpack, unpack_from


@dataclass(frozen=True, slots=True)
class ContactTimeContext:
    pid: int
    birth: int
    module_base: int
    manager: int
    start_native: int
    stop_native: int


@dataclass(frozen=True, slots=True)
class ProjectedContact:
    native_time: int
    source_clock_ms: int
    base_ns: int
    frame_delta_seconds: float
    clamped_seconds: float
    projected_delta_seconds: float
    timestamp_ms: int
    source_pointer: int
    contact: bytes
    physical_crossing_proven: bool = False


def _f32(value: float) -> float:
    _decoded_scalar: float = unpack("<f", pack("<f", value))[0]
    return _decoded_scalar


def decode_projected_contacts(
    raw: bytes, context: ContactTimeContext
) -> tuple[ProjectedContact, ...]:
    """Require each positive registry append to follow its actual 936250 call."""
    if not raw or len(raw) % 1024 or len(raw) > 64 * 1024 * 1024:
        raise ValueError("complete bounded contact-time records required")
    pending = None
    pairs = []
    tracer_tid = None
    for sequence, offset in enumerate(range(0, len(raw), 1024), 1):
        record = raw[offset : offset + 1024]

        def u(at: int, buffer: bytes = record) -> int:
            _decoded_scalar: int = unpack_from("<I", buffer, at)[0]
            return _decoded_scalar

        def q(at: int, buffer: bytes = record) -> int:
            _decoded_scalar: int = unpack_from("<Q", buffer, at)[0]
            return _decoded_scalar

        kind = u(76)
        if (
            record[:8] != b"TMHW0004"
            or unpack_from("<3I", record, 8) != (1024, 4, context.pid)
            or q(24) != sequence
            or q(32) != context.manager + 0xC0
            or q(768) != context.birth
            or q(776) != context.module_base + 0x9364ED
            or unpack_from("<2I", record, 56) != (4, 5)
            or not context.start_native <= u(72) < context.stop_native
            or u(72) % 10
            or kind not in (1, 2)
            or q(48) & 15 != (1 if kind == 1 else 2)
            or u(20) == 0
        ):
            raise ValueError("contact-time ownership or execution mismatch")
        if tracer_tid is None:
            tracer_tid = u(20)
        if tracer_tid != u(20):
            raise ValueError("contact-time tracer thread changed")
        registers = unpack_from("<27Q", record, 80)
        rva = registers[16] - context.module_base
        if kind == 2:
            if (
                pending is not None
                or rva != 0x9364ED
                or q(760) != context.module_base + 0xA55768
                or q(600) != registers[1]
                or q(592) != registers[5]
            ):
                raise ValueError("contact-time native caller mismatch")
            pending = record
            continue
        if q(40) == 0:
            if pending is not None or rva not in (0xCBC552, 0xCBC789, 0xCBC592):
                raise ValueError("unexpected registry reset during projection")
            continue
        if pending is None or rva != 0xA557D5:
            raise ValueError("registry append without witnessed projection")
        before = pending
        pending = None
        if (
            before[72:76] != record[72:76]
            or before[592:608] != record[592:608]
            or q(40) != unpack_from("<Q", before, 40)[0] + 1
            or q(40) != u(584)
            or unpack_from("<Q", before, 40)[0] != unpack_from("<I", before, 584)[0]
            or before[608:684] != record[608:684]
            or before[688:696] != record[688:696]
            or before[848:860] != record[848:860]
            or q(600) != registers[1]
            or unpack_from("<I", before, 784)[0] & ~63 != u(784) & ~63
        ):
            raise ValueError("contact-time projection/append join mismatch")
        dt = unpack_from("<f", before, 800)[0]
        clamped = unpack_from("<f", before, 816)[0]
        delta = unpack_from("<f", record, 800)[0]
        clock_ms, step_ms = unpack_from("<2I", record, 848)
        if (
            not all(isfinite(v) for v in (dt, clamped, delta))
            or not _f32(0.001) <= clamped <= dt <= 0.05
            or delta != _f32(dt - clamped)
            or clock_ms != u(72) + 10
            or step_ms != 10
            or record[856:860] != pack("<I", 0x3C23D70B)  # Measured pinned 10ms profile encoding.
            or not clock_ms * 1000000 <= q(592) < (clock_ms + step_ms) * 1000000
        ):
            raise ValueError("contact-time native clock or floating point mismatch")
        expected = (q(592) + trunc(delta * 1_000_000_000)) // 1_000_000
        if u(684) != expected:
            raise ValueError("contact-time integer timestamp mismatch")
        pairs.append(
            ProjectedContact(
                u(72), clock_ms, q(592), dt, clamped, delta, expected, q(600), record[608:696]
            )
        )
    if pending is not None or not pairs:
        raise ValueError("unfinished contact-time execution")
    return tuple(pairs)
