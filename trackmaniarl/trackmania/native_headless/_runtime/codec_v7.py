"""Experimental immediate-input wire; the pinned V6 decoder stays unchanged."""

from __future__ import annotations

import math
import struct
from typing import Any

from trackmaniarl.trackmania.native_headless._runtime import codec_v6

PROFILE_ID = "tmre.long4152-immediate/v7"


def decode_v7_response(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or len(raw) != 448 or raw[:5] != b"TMR7\x07":
        raise ValueError("V7 exact frame/header required")
    kind = raw[5]
    view = bytearray(raw)
    view[:5] = b"TMR6\x06"
    if kind == 6:
        n = struct.unpack_from("<I", raw, 12)[0]
        p, k, length, mode = struct.unpack_from("<4I", raw, 288)
        if not (1 <= n <= 5 and p == k and length == n and mode == 2):
            raise ValueError("V7 immediate current-record receipt required")
        if struct.unpack_from("<I", raw, 28)[0] != 4:
            raise ValueError("V7 immediate flag required")
        count = struct.unpack_from("<I", raw, 32 + 116)[0]
        selected = struct.unpack_from("<I", raw, 32 + 112)[0]
        if p != selected or p + n > count or any(raw[304 + n * 24 :]):
            raise ValueError("V7 current-record range/unused receipt bytes mismatch")
        if raw[304:316] != raw[32 + 156 : 32 + 168]:
            raise ValueError("V7 first before channels must match captured current reader")
        for ordinal in range(n):
            at = 316 + ordinal * 24
            steer, gas, brake = struct.unpack_from("<3f", raw, at)
            if (
                not math.isfinite(steer)
                or not -1 <= steer <= 1
                or gas not in (0, 1)
                or brake not in (0, 1)
            ):
                raise ValueError("V7 after channels outside native domain")
        # Reuse the strict read-only reader decoder, then restore V7 semantics.
        # A V6 commit decoder intentionally cannot authorize k == p.
        view[5] = 3
        struct.pack_into("<I", view, 12, 0)
        struct.pack_into("<I", view, 28, 0)
        view[288:] = bytes(160)
    decoded: dict[str, Any] = codec_v6.decode_v6_response(bytes(view))
    decoded.update(
        version=7,
        kind=kind,
        wireProfile="TMR7/v7; op6 pre-load current input",
        profileId=PROFILE_ID,
        immediateCurrentRecord=kind == 6,
    )
    if kind == 6:
        decoded.update(
            iterations=n,
            receiptHex=raw[288:].hex(),
            receipt={"p": p, "first": k, "length": length, "mode": mode},
            samePreWriteReaderEvent=True,
        )
    return decoded


def encode_request(  # noqa: PLR0913 -- independently validated literal protocol fields
    *,
    op: int,
    sequence: int,
    n: int,
    epoch: int,
    p: int = 0,
    k: int = 0,
    befores: tuple[bytes, ...] = (),
    afters: tuple[bytes, ...] = (),
) -> bytes:
    if len(befores) != len(afters) or len(afters) > 5:
        raise ValueError("V7 paired bounded input records required")
    if op == 6 and not (1 <= n <= 5 and p == k and len(afters) == n):
        raise ValueError("V7 immediate request writes exactly consumed current records")
    result = bytearray(192)
    result[:8] = b"TMA7\x07" + bytes([op]) + b"\x00\x00"
    struct.pack_into("<IIQIII", result, 8, sequence, n, epoch, p, k, len(afters))
    for ordinal, (before, after) in enumerate(zip(befores, afters, strict=True)):
        if len(before) != 12 or len(after) != 12:
            raise ValueError("V7 input channel record must have 12 bytes")
        offset = 40 + ordinal * 24
        result[offset : offset + 12] = before
        result[offset + 12 : offset + 24] = after
    return bytes(result)
