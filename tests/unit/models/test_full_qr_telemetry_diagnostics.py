"""Source equation tests; fixtures are synthetic, not live engine evidence."""

from struct import pack, pack_into

import pytest

from experiments.full_qr.telemetry_diagnostics import diagnose_vehicle


def test_skidding_and_material_branches_preserve_raw_coefficients() -> None:
    visual = bytearray(864)
    for i, (slip, flags, material) in enumerate(
        [(0.5, 4, 16), (0.0, 4, 3), (1.0, 0, 2), (-0.5, 6, 7)]
    ):
        pack_into("<f", visual, 0xBC + i * 0x2C, slip)
        pack_into("<I", visual, 0xD0 + i * 0x2C, flags)
        visual[0xB8 + i * 0x2C] = material
    cache = bytearray(0x8C)
    pack_into("<I", cache, 0x4C, 2)
    result = diagnose_vehicle(bytes(visual), bytes(cache), bytes(8))
    assert result.slip_native_order == (0.5, 0.0, 1.0, -0.5)
    assert result.material_native_order == (16, 3, 80, 13)
    assert result.modeled_skidding_count == 1
    assert result.cached_skidding_count == 2
    pack_into("<I", visual, 0x88, 32)
    burning = diagnose_vehicle(bytes(visual), bytes(cache), pack("<Q", 4096))
    assert burning.modeled_skidding_count == 3
    assert burning.modeled_adherence_default is None
    assert result.modeled_adherence_default == 1.0


@pytest.mark.parametrize(
    ("flags", "extra", "expected"),
    [(0x10000, 0, True), (0x10008, 0, False), (0, 0, False), (0x10000, 1, False)],
)
def test_flying_predicate_is_not_a_ground_contact_timer(
    flags: int, extra: int, *, expected: bool
) -> None:
    visual = bytearray(864)
    pack_into("<I", visual, 0x88, flags)
    pack_into("<I", visual, 0x1BC, extra)
    result = diagnose_vehicle(bytes(visual), bytes(0x8C), bytes(8))
    assert result.flying_accumulator_branch is expected


def test_missing_bytes_and_nonfinite_slip_are_rejected() -> None:
    with pytest.raises(ValueError, match="864"):
        diagnose_vehicle(bytes(863), bytes(0x8C), bytes(8))
    visual = bytearray(864)
    pack_into("<f", visual, 0xBC, float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        diagnose_vehicle(bytes(visual), bytes(0x8C), bytes(8))
