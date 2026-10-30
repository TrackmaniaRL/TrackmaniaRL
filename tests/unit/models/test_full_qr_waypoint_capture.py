"""Synthetic format fixtures, never gameplay or physical-crossing evidence."""

import struct

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.waypoint_capture import decode_writer_pairs


def fixture() -> tuple[bytes, dict[str, int]]:
    meta = {
        "module_slide": 0x10000000,
        "participant": 0x40000000,
        "sim": 0x50000000,
        "controller": 0x60000000,
    }
    rows = [bytearray(512), bytearray(512)]
    for index, d in enumerate(rows):
        d[:8] = b"TMWP0001"
        struct.pack_into("<4I", d, 8, 512, 1, 123, 123)
        struct.pack_into("<Q", d, 24, index + 1)
        struct.pack_into(
            "<IIQQII",
            d,
            32,
            index,
            (0x1216411, 0x12167D6)[index],
            meta["sim"],
            meta["controller"],
            19900,
            (1231, 799)[index],
        )
        struct.pack_into(
            "<QQ4I", d, 248, meta["participant"], meta["participant"], 10, 11, 2200, 2200
        )
    a, b = rows
    regs = [0] * 23
    for index, value in {
        9: meta["participant"],
        12: 19915,
        14: 19910,
        10: 0x70000000,
        1: 1,
    }.items():
        regs[index] = value
    struct.pack_into("<23Q", a, 64, *regs)
    struct.pack_into("<QQ", a, 360, regs[10], meta["module_slide"] + 0x11FBA27)
    struct.pack_into("<QQ", a, 392, 0x71000000, meta["module_slide"] + 0x120AC95)
    struct.pack_into("<II", a, 416, 0, 19910)
    struct.pack_into("<I", a, 440, 0x327D000)
    struct.pack_into("<I", a, 456, 19915)
    struct.pack_into("<4Q", a, 464, 0x72000000, regs[10], 0x120AC95, 0x71000000)
    struct.pack_into("<III", b, 288, 0, 19915, 1)
    for offset, selector in ((416, 10), (448, 11)):
        struct.pack_into("<6I", b, offset, 12, selector, 0, 1, 19915, 0)
    return bytes(a + b), meta


def test_written_result_does_not_authorize_physical_finish() -> None:
    raw, meta = fixture()
    (evidence,) = decode_writer_pairs(raw, pid=123, meta=meta)
    assert evidence.raw_waypoint_time == 19915
    assert evidence.endpoint_write_requested
    assert evidence.physical_crossing_proven is False


@pytest.mark.parametrize(
    ("offset", "fmt", "value"),
    [
        (368, "Q", 0x12389A0),  # scripting caller must not acquire event provenance
        (456, "I", 19914),  # dispatcher event and writer argument disagree
        (512 + 432, "I", 19914),  # actual destination differs from the entry argument
        (512 + 24, "Q", 1),  # replayed/out-of-order store
    ],
)
def test_unrelated_or_corrupted_write_is_rejected(offset: int, fmt: str, value: int) -> None:
    raw, meta = fixture()
    corrupt = bytearray(raw)
    struct.pack_into("<" + fmt, corrupt, offset, value)
    with pytest.raises(
        ValueError, match=r"binding mismatch|destination mismatch|ownership mismatch"
    ):
        decode_writer_pairs(bytes(corrupt), pid=123, meta=meta)
