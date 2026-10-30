"""Pure decoder for copied interrupted FXSAVE control words."""

from __future__ import annotations

import struct


def u32(raw: bytes, offset: int) -> int:
    return struct.unpack_from("<I", raw, offset)[0]


ROUNDING = {0: "nearest-even", 1: "toward-negative", 2: "toward-positive", 3: "toward-zero"}


def decode_saved_fp(raw: bytes) -> dict:
    if len(raw) != 256 or raw[:8] != b"TMFPU001" or u32(raw, 8) != 256 or u32(raw, 12) != 1:
        raise ValueError("exact independent FP sidecar required")
    saved = raw[96:128]
    mxcsr, cw, sw, mask = (u32(raw, offset) for offset in (44, 48, 52, 56))
    if (
        mxcsr != u32(saved, 24)
        or mask != u32(saved, 28)
        or cw != struct.unpack_from("<H", saved)[0]
        or sw != struct.unpack_from("<H", saved, 2)[0]
        or mxcsr & ~0xFFFF
    ):
        raise ValueError("FP controls disagree with copied saved interrupted bytes")
    return {
        "native_t": u32(raw, 40),
        "pid": u32(raw, 16),
        "tid": u32(raw, 20),
        "mxcsr_raw": mxcsr,
        "mxcsr_mask_raw": mask,
        "mxcsr_rounding": ROUNDING[(mxcsr >> 13) & 3],
        "mxcsr_flush_to_zero": bool(mxcsr & (1 << 15)),
        "mxcsr_denormals_are_zero": bool(mxcsr & (1 << 6)),
        "mxcsr_exception_status": mxcsr & 63,
        "mxcsr_exception_masks": (mxcsr >> 7) & 63,
        "x87_control_raw": cw,
        "x87_status_raw": sw,
        "x87_rounding": ROUNDING[(cw >> 10) & 3],
        "x87_precision_bits": {0: 24, 1: "reserved", 2: 53, 3: 64}[(cw >> 8) & 3],
        "saved_interrupted_fxsave32_hex": saved.hex(),
    }
