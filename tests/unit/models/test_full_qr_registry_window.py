"""Coverage guards on an invented 50 ms finish-control snapshot."""

from dataclasses import asdict
from struct import pack_into

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.contact_registry_capture import (
    RegistryContext,
    decode_registry_window,
)


def evidence() -> tuple[bytes, RegistryContext]:
    context = RegistryContext(pid=123, sim=0x20000, module_base=0x10000000, manager=0x30000)
    records = []
    for index in range(5):
        raw = bytearray(128)
        raw[:8] = b"TMCR0001"
        pack_into("<4I", raw, 8, 128, 1, context.pid, context.pid)
        pack_into(
            "<QIIQQ",
            raw,
            24,
            1772 + index,
            19900 + index * 10,
            0xCBC5A0,
            context.sim,
            context.manager,
        )
        pack_into("<QIIQII", raw, 56, 0, 0, 0, 0x40000, int(index < 4), 4)
        pack_into("<3Q", raw, 88, context.module_base, 0, context.module_base + 0xCBC5A0)
        records.append(raw)
    return b"".join(records), context


def test_synthetic_window_has_every_interval_despite_zero_waypoint_vector() -> None:
    raw, context = evidence()
    result = decode_registry_window(raw, context, (19900, 19950))
    assert result.intervals == 5
    assert result.contact_batches == 4
    assert result.waypoint_batches == 0
    assert asdict(result) == {
        "start_native_ms": 19900,
        "end_native_ms": 19950,
        "intervals": 5,
        "contact_batches": 4,
        "waypoint_batches": 0,
    }
    assert not hasattr(result, "finished")


@pytest.mark.parametrize("kind", ["missing", "partial", "duplicate", "swap"])
def test_missing_or_reordered_interval_rejected(kind: str) -> None:
    raw, context = evidence()
    damaged = {
        "missing": raw[:-128],
        "partial": raw[:-1],
        "duplicate": raw[:128] + raw[:128] + raw[256:],
        "swap": raw[128:256] + raw[:128] + raw[256:],
    }[kind]
    with pytest.raises(ValueError, match="registry"):
        decode_registry_window(damaged, context, (19900, 19950))


@pytest.mark.parametrize("offset", [16, 24, 32, 48, 88])
def test_redirected_window_record_rejected(offset: int) -> None:
    raw, context = evidence()
    damaged = bytearray(raw)
    pack_into("<I", damaged, 128 + offset, 0)
    with pytest.raises(ValueError, match="registry"):
        decode_registry_window(bytes(damaged), context, (19900, 19950))


@pytest.mark.parametrize("bounds", [(19910, 19960), (19900, 19900), (19901, 19951)])
def test_misbound_window_rejected(bounds: tuple[int, int]) -> None:
    raw, context = evidence()
    with pytest.raises(ValueError, match="registry"):
        decode_registry_window(raw, context, bounds)
