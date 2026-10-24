from __future__ import annotations

import struct
from typing import Any

from trackmaniarl.trackmania.native_headless._runtime import codec_v5 as V5

PROFILE_ID = "tmre.long4152-declared40000/v6"
INPUT_COUNT = 4152
MAX_NATIVE = 50000
TAPE_SHA256 = "1cb2dfe4749c9743c8aa7cc710c5442f82c1193157e2c15b8e973f5bb5d6979e"
DRIVER_SHA256 = "669486302fd66cdd8833128e06879089d1c4073ceed0dc11bcc0963fea06e1c4"


def u(b: Any, o: Any) -> Any:
    return struct.unpack_from("<I", b, o)[0]


def q(b: Any, o: Any) -> Any:
    return struct.unpack_from("<Q", b, o)[0]


def decode_v6_response(b: Any) -> Any:
    if type(b) is not bytes or len(b) != 448 or b[:5] != b"TMR6\x06":
        raise ValueError("V6 exact frame/header required")
    view = bytearray(b)
    view[:5] = b"TMR5\x05"
    d = V5.decode_v5_response(bytes(view))
    kind = d["kind"]
    n = d["iterations"]
    raw = b[32:288]
    if n > 1000 or (kind in (2, 3) and n != 0) or (kind == 4 and n == 0):
        raise ValueError("V6 iteration bounds")
    if kind in (3, 4, 5) and u(raw, 116) != INPUT_COUNT:
        raise ValueError("V6 reader count profile")
    anchor = u(raw, 112) if kind in (1, 2) else u(raw, 124)
    if anchor > MAX_NATIVE:
        raise ValueError("V6 observed native cap")
    d.update(
        version=6,
        wireProfile="TMR6/v6; unchanged V5 ops1..5 schemas",
        profileId=PROFILE_ID,
        profileInputCount=INPUT_COUNT,
        profileMaxNative=MAX_NATIVE,
        sourceTapeIdentityProvenByPayload=False,
    )
    return d
