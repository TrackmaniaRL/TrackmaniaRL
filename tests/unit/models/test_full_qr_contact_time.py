"""Synthetic record guards; native execution is checked with saved actual probes."""

from struct import pack, pack_into, unpack

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.contact_time_capture import (
    ContactTimeContext,
    decode_projected_contacts,
)


def _records() -> tuple[bytes, ContactTimeContext]:
    context = ContactTimeContext(123, 456, 0x10000000, 0x20000, 19850, 19990)
    records = []
    dt = unpack("<f", pack("<f", 0.0033333336468786))[0]
    tau = unpack("<f", pack("<f", 0.001))[0]
    delta = unpack("<f", pack("<f", dt - tau))[0]
    for index, kind in enumerate((2, 1)):
        raw = bytearray(1024)
        raw[:8] = b"TMHW0004"
        pack_into("<4I", raw, 8, 1024, 4, 123, 1000)
        pack_into("<4Q", raw, 24, index + 1, context.manager + 0xC0, index, kind)
        pack_into("<2I", raw, 56, 4, 5)
        pack_into("<2I", raw, 72, 19900, kind)
        registers = [0] * 27
        registers[1], registers[5] = 0x30000, 19913333333
        registers[16] = context.module_base + (0x9364ED if kind == 2 else 0xA557D5)
        pack_into("<27Q", raw, 80, *registers)
        pack_into("<QII", raw, 576, 0x40000, index, 6)
        pack_into("<2Q", raw, 592, 19913333333, 0x30000)
        pack_into("<2Q", raw, 608, 0x50000, 0x60000)
        pack_into("<I", raw, 684, 0 if kind == 2 else 19915)
        pack_into(
            "<4Q",
            raw,
            752,
            0x70000,
            context.module_base + 0xA55768,
            456,
            context.module_base + 0x9364ED,
        )
        pack_into("<I", raw, 784, 0x1F80)
        pack_into("<f", raw, 800, dt if kind == 2 else delta)
        pack_into("<f", raw, 816, tau)
        pack_into("<3I", raw, 848, 19910, 10, 0x3C23D70B)
        records.append(bytes(raw))
    return b"".join(records), context


def test_native_substep_projection_is_not_relabelled_as_physical_crossing() -> None:
    raw, context = _records()
    (pair,) = decode_projected_contacts(raw, context)
    assert pair.timestamp_ms == 19915
    assert pair.native_time == 19900
    assert pair.source_clock_ms == 19910
    assert pair.physical_crossing_proven is False


@pytest.mark.parametrize(
    ("offset", "fmt", "value"),
    [
        (760, "Q", 0),
        (800, "f", 0.004),
        (1024 + 684, "I", 19916),
        (768, "Q", 0),
        (1024 + 600, "Q", 0),
        (1024 + 856, "I", 0x3C23D70A),
    ],
)
def test_changed_native_execution_or_clock_is_rejected(
    offset: int, fmt: str, value: int | float
) -> None:
    raw, context = _records()
    changed = bytearray(raw)
    pack_into("<" + fmt, changed, offset, value)
    with pytest.raises(ValueError, match="contact-time"):
        decode_projected_contacts(bytes(changed), context)


def test_append_without_execution_is_rejected() -> None:
    raw, context = _records()
    changed = bytearray(raw[1024:])
    pack_into("<Q", changed, 24, 1)
    with pytest.raises(ValueError, match="without witnessed projection"):
        decode_projected_contacts(bytes(changed), context)
