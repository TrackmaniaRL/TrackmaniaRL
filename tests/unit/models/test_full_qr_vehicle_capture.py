"""Synthetic parser fixtures; native execution evidence is stored separately."""

from struct import pack_into

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.vehicle_capture import (
    PRODUCER_RVA,
    VehicleCaptureContext,
    decode_vehicle_capture,
)


def _record() -> tuple[bytes, VehicleCaptureContext]:
    raw = bytearray(1280)
    raw[:8] = b"TMVH0001"
    context = VehicleCaptureContext(123, 0x10000, 0x20000, 0x10000000, 2190)
    pack_into("<4I", raw, 8, 1280, 1, 123, 123)
    pack_into("<QII", raw, 24, 1, 2190, 0)
    pack_into("<3Q", raw, 40, context.participant, context.vehicle, 0)
    pack_into("<I", raw, 204, 1)
    pack_into("<I", raw, 220, 2190)
    pack_into("<I", raw, 1240, PRODUCER_RVA)
    pack_into("<4I", raw, 1248, 2190, 2190, 2190, 2190)
    pack_into("<2Q", raw, 1264, context.vehicle + 0x848, context.module_base + PRODUCER_RVA)
    # Different current vendor result and captured participant cache must survive.
    pack_into("<I", raw, 64 + 0x4C, 3)
    pack_into("<I", raw, 64 + 0x5C, 120)
    for i, slip in enumerate((0.1, 0.2, 0.3, 0.4)):
        pack_into("<f", raw, 208 + 0xBC + i * 0x2C, slip)
    return bytes(raw), context


def test_vendor_result_is_not_replaced_by_stale_participant_cache() -> None:
    raw, context = _record()
    captured = decode_vehicle_capture(raw, context)
    assert captured.skidding_count == 3
    assert captured.flying_duration == 120
    assert captured.participant_cache == bytes(140)
    assert captured.slip_connect_order == pytest.approx((0.1, 0.2, 0.4, 0.3))


@pytest.mark.parametrize(
    ("offset", "value"), [(16, 124), (24, 2), (32, 2200), (204, 0), (1240, 0), (1264, 0)]
)
def test_redirected_or_stale_context_is_rejected(offset: int, value: int) -> None:
    raw, context = _record()
    changed = bytearray(raw)
    pack_into("<I", changed, offset, value)
    with pytest.raises(ValueError, match=r"vehicle|producer|proof"):
        decode_vehicle_capture(bytes(changed), context)


def test_fixture_stream_and_nonfinite_coefficients_are_rejected() -> None:
    raw, context = _record()
    with pytest.raises(ValueError, match="exact"):
        decode_vehicle_capture(b"TMVF0001" + bytes(56), context)
    changed = bytearray(raw)
    pack_into("<f", changed, 208 + 0xBC, float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        decode_vehicle_capture(bytes(changed), context)
