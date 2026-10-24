"""Saved native contact-registry captures; no finish or collision authority."""

from dataclasses import dataclass
from struct import unpack_from

REGISTRY_RVA = 0xCBC5A0


@dataclass(frozen=True, slots=True)
class RegistryContext:
    pid: int
    sim: int
    module_base: int
    manager: int


@dataclass(frozen=True, slots=True)
class RegistryCapture:
    native_time: int
    sequence: int
    waypoint_buffer: int
    waypoint_count: int
    waypoint_capacity: int
    contact_buffer: int
    contact_count: int
    contact_capacity: int


@dataclass(frozen=True, slots=True)
class RegistryWindow:
    start_native_ms: int
    end_native_ms: int
    intervals: int
    contact_batches: int
    waypoint_batches: int


def decode_registry_window(
    raw: bytes, context: RegistryContext, native_bounds: tuple[int, int]
) -> RegistryWindow:
    """Require a consumer snapshot for every 10 ms interval in a held window.

    Counts describe the vectors at the observed consumer entry. Even a fully
    covered empty window does not prove absence of the historical UI finish.
    """
    if type(native_bounds) is not tuple or len(native_bounds) != 2:
        raise ValueError("bounded registry start/end pair required")
    start_native_ms, end_native_ms = native_bounds
    if (
        type(start_native_ms) is not int
        or type(end_native_ms) is not int
        or not 2200 <= start_native_ms < end_native_ms <= 152200
        or start_native_ms % 10
        or end_native_ms % 10
        or type(raw) is not bytes
        or len(raw) != (end_native_ms - start_native_ms) // 10 * 128
    ):
        raise ValueError("complete bounded registry window required")
    captures = [
        decode_registry_capture(raw[offset : offset + 128], context)
        for offset in range(0, len(raw), 128)
    ]
    if any(
        capture.native_time != start_native_ms + index * 10
        for index, capture in enumerate(captures)
    ):
        raise ValueError("registry window interval coverage mismatch")
    return RegistryWindow(
        start_native_ms,
        end_native_ms,
        len(captures),
        sum(capture.contact_count > 0 for capture in captures),
        sum(capture.waypoint_count > 0 for capture in captures),
    )


def decode_registry_capture(raw: bytes, context: RegistryContext) -> RegistryCapture:
    """Check a saved observer record against a supplied context.

    The native consumer supplies manager identity. This decoder does not authorize
    ptrace, a live process, or infer no collision from a zero contact count.
    """
    if type(raw) is not bytes or len(raw) != 128 or raw[:8] != b"TMCR0001":
        raise ValueError("complete TMCR0001 registry record required")
    if unpack_from("<4I", raw, 8) != (128, 1, context.pid, context.pid):
        raise ValueError("registry owner/header mismatch")
    sequence, clock, rva, sim, manager = unpack_from("<QIIQQ", raw, 24)
    base, _stack, site = unpack_from("<3Q", raw, 88)
    if (rva, sim, manager, base, site) != (
        REGISTRY_RVA,
        context.sim,
        context.manager,
        context.module_base,
        context.module_base + REGISTRY_RVA,
    ):
        raise ValueError("registry native context mismatch")
    if manager == 0 or manager % 8 or raw[112:] != bytes(16):
        raise ValueError("registry address/reserved bytes mismatch")
    if not 1 <= sequence <= 16200 or clock != 2190 + (sequence - 1) * 10:
        raise ValueError("registry sequence/clock mismatch")
    waypoint, wc, wcap, contact, cc, ccap = unpack_from("<QIIQII", raw, 56)
    if not (0 <= wc <= wcap <= 10000 and 0 <= cc <= ccap <= 10000):
        raise ValueError("registry vector count/capacity mismatch")
    if (wcap and (not waypoint or waypoint % 8)) or (ccap and (not contact or contact % 8)):
        raise ValueError("registry vector storage mismatch")
    return RegistryCapture(clock, sequence, waypoint, wc, wcap, contact, cc, ccap)
