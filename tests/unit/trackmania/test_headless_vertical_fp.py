"""FP evidence must bind interpretations to the saved interrupted bytes."""

import struct

import pytest

from scripts.headless_vertical_fp_decode import decode_saved_fp


def record(mxcsr: int = 0x1FA5, control: int = 0x037F) -> bytes:
    raw = bytearray(256)
    raw[:8] = b"TMFPU001"
    struct.pack_into("<4I", raw, 8, 256, 1, 1234, 1234)
    struct.pack_into("<5I", raw, 40, 3820, mxcsr, control, 0x20, 0xFFFF)
    struct.pack_into("<HH", raw, 96, control, 0x20)
    struct.pack_into("<II", raw, 120, mxcsr, 0xFFFF)
    return bytes(raw)


def test_saved_fp_capture_decodes_real_engine_control_bits() -> None:
    result = decode_saved_fp(record())
    assert result["mxcsr_rounding"] == result["x87_rounding"] == "nearest-even"
    assert result["x87_precision_bits"] == 64
    assert result["mxcsr_exception_status"] == 0x25
    assert result["mxcsr_exception_masks"] == 63
    assert result["mxcsr_flush_to_zero"] is result["mxcsr_denormals_are_zero"] is False


def test_alternative_rounding_and_precision_are_reported_without_normalization() -> None:
    result = decode_saved_fp(record(0xDFC0, 0x0A7F))
    assert result["mxcsr_rounding"] == "toward-positive"
    assert result["mxcsr_flush_to_zero"] is result["mxcsr_denormals_are_zero"] is True
    assert result["x87_rounding"] == "toward-positive"
    assert result["x87_precision_bits"] == 53


def test_control_fields_cannot_disagree_with_the_copied_saved_context() -> None:
    raw = bytearray(record())
    struct.pack_into("<I", raw, 44, 0x9FA5)
    with pytest.raises(ValueError, match="saved interrupted bytes"):
        decode_saved_fp(bytes(raw))


def test_reserved_mxcsr_upper_bits_are_not_accepted_as_a_valid_control_word() -> None:
    with pytest.raises(ValueError, match="saved interrupted bytes"):
        decode_saved_fp(record(0x11FA5))


@pytest.mark.parametrize("raw", [b"", record()[:-1], b"TMPOST03" + record()[8:]])
def test_only_the_independent_complete_fp_sidecar_is_accepted(raw: bytes) -> None:
    with pytest.raises(ValueError, match="sidecar"):
        decode_saved_fp(raw)
