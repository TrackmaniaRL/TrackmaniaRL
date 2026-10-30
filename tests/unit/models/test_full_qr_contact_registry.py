"""Parser guards only; native execution is verified in separate saved probes."""

from struct import pack_into

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.contact_registry_capture import (
    REGISTRY_RVA,
    RegistryContext,
    decode_registry_capture,
)


def fixture_record() -> tuple[bytearray, RegistryContext]:
    context = RegistryContext(123, 0x10000, 0x10000000, 0x20000)
    record = bytearray(128)
    record[:8] = b"TMCR0001"
    pack_into("<4I", record, 8, 128, 1, 123, 123)
    pack_into("<QIIQQ", record, 24, 1, 2190, REGISTRY_RVA, context.sim, context.manager)
    pack_into("<QIIQII", record, 56, 0x30000, 0, 250, 0x40000, 3, 6)
    pack_into("<3Q", record, 88, context.module_base, 0x50000, context.module_base + REGISTRY_RVA)
    return record, context


def test_registry_preserves_actual_counts_without_collision_classification() -> None:
    raw, context = fixture_record()
    capture = decode_registry_capture(bytes(raw), context)
    assert capture.contact_count == 3
    assert capture.waypoint_count == 0
    pack_into("<I", raw, 80, 0)
    assert decode_registry_capture(bytes(raw), context).contact_count == 0


@pytest.mark.parametrize(
    ("offset", "value"), [(16, 124), (24, 2), (32, 2200), (48, 0), (80, 7), (88, 0), (112, 1)]
)
def test_registry_rejects_redirected_or_inconsistent_evidence(offset: int, value: int) -> None:
    raw, context = fixture_record()
    pack_into("<I", raw, offset, value)
    with pytest.raises(ValueError, match="registry"):
        decode_registry_capture(bytes(raw), context)
