from __future__ import annotations

import struct
from typing import Any

from trackmaniarl.trackmania.native_headless._runtime import codec_v4 as SDK


def u(b: Any, o: Any) -> Any:
    return struct.unpack_from("<I", b, o)[0]


def q(b: Any, o: Any) -> Any:
    return struct.unpack_from("<Q", b, o)[0]


def decode_v5_response(b: Any) -> Any:
    if not isinstance(b, bytes) or len(b) != 448 or b[:5] != b"TMR5\x05" or (b[6:8] != bytes(2)):
        raise ValueError("V5 exact frame/header required")
    kind = b[5]
    if kind not in (1, 2, 3, 4, 5) or u(b, 24) != 416 or q(b, 16) == 0:
        raise ValueError("V5 kind/payload/epoch")
    if kind != 5:
        view = bytearray(b)
        view[:5] = b"TMR4\x04"
        result = SDK.decode_v4_response(bytes(view))
        result["version"] = 5
        result["wireProfile"] = "TMR5/v5; unchanged ops1..4 event+receipt schema"
        return result
    if u(b, 12) != 1 or u(b, 28) != 2:
        raise ValueError("release ACK requires N1 and flags2")
    event = SDK.decode_event(b[32:288])
    raw = b[32:288]
    receipt = b[288:]
    if (
        event.get("eventType") != "TMREAD01"
        or event.get("readerRole") != 1
        or event.get("validityMask") != 447
    ):
        raise ValueError("release ACK reader domain")
    p, count = (u(raw, 112), u(raw, 116))
    if not 0 < count <= 200000 or p + 1 != count:
        raise ValueError("release ACK last-reader domain")
    if (u(receipt, 0), u(receipt, 4), u(receipt, 8), u(receipt, 12)) != (p, count, 1, 1) or receipt[
        52:
    ] != bytes(108):
        raise ValueError("release receipt metadata/reserved")
    if (
        receipt[16:48] != raw[152:184]
        or receipt[48:52] != raw[184:188]
        or receipt[48:52] != receipt[20:24]
    ):
        raise ValueError("release receipt current/shadow mismatch")
    return {
        "version": 5,
        "kind": 5,
        "sequence": u(b, 8),
        "iterations": 1,
        "epoch": q(b, 16),
        "event": event,
        "receiptHex": receipt.hex(),
        "releaseReceipt": {
            "ownP": p,
            "inputCount": count,
            "noFuturePlan": True,
            "currentRecordHex": receipt[16:48].hex(),
            "shadowHex": receipt[48:52].hex(),
            "stage": "authorization/readback before returning to original reader",
        },
        "physicalTransitionCandidate": False,
        "releaseConsumptionConfirmed": False,
        "completedIntervalsInThisAck": 0,
        "runAckIsNewState": False,
        "heldNow": None,
        "requiresLiveOwnerAndEpochCheck": True,
    }
