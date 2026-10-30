from __future__ import annotations

import math
import struct
from typing import Any

from trackmaniarl.trackmania.native_headless._runtime.shapes import NumericGroupSpec


def u32(b: Any, o: Any) -> Any:
    return struct.unpack_from("<I", b, o)[0]


def u64(b: Any, o: Any) -> Any:
    return struct.unpack_from("<Q", b, o)[0]


def floats(b: Any, o: Any, n: Any) -> Any:
    return list(struct.unpack_from("<" + "f" * n, b, o))


def numeric_group(spec: NumericGroupSpec) -> Any:
    b, o, n, available = (spec.b, spec.o, spec.n, spec.available)
    if not available:
        return {"value": None, "status": "unavailable", "bits": None}
    words = [u32(b, o + i * 4) for i in range(n)]
    v = floats(b, o, n)
    return {
        "value": v if all(math.isfinite(x) for x in v) else None,
        "status": "available" if all(math.isfinite(x) for x in v) else "invalid_nonfinite",
        "bits": words,
    }


def decode_event(b: Any) -> Any:
    if len(b) != 256:
        raise ValueError("event must be exactly256B")
    tag = b[:8]
    if tag not in (b"TMPOST03", b"TMREAD01"):
        raise ValueError("unsupported event magic")
    post = tag == b"TMPOST03"
    version = 3 if post else 1
    kind = 10 if post else 11
    if (u32(b, 8), u32(b, 12), u32(b, 212)) != (256, version, kind):
        raise ValueError("event size/version/kind mismatch")
    mask = u32(b, 148)
    if mask & ~(203 if post else 511):
        raise ValueError("unknown pinned validity bits")
    result = {
        "eventType": tag.decode(),
        "phase": "BOUNDARY_NATIVE_1219774" if post else "READER_PRE_READ_119F163",
        "pid": u32(b, 16),
        "tid": u32(b, 20),
        "hit": u64(b, 24),
        "lroundfCalls": u64(b, 32),
        "validityMask": mask,
        "heldNow": None,
        "atomicSnapshotProven": False,
        "physical": None,
        "vehicleTelemetry": None,
        "selectedInput": None,
        "rawRecord": None,
        "readBoundaryProfileReady": False,
        "timestamps": {},
        "rawEventHex": b.hex(),
    }
    if post:
        if (
            not mask & 8
            or bool(mask & 1) != bool(mask & 64)
            or (mask & 2 and (not mask & 1))
            or (mask & 128 and (not mask & 1))
        ):
            raise ValueError("mask impossible for pinned boundary writer")
        if not (u32(b, 112) == u32(b, 116) == u32(b, 144) and u32(b, 136) == u32(b, 140) == 10):
            raise ValueError("boundary guarded clock/dt mismatch")
        result["timestamps"] = {
            "localT": {"value": u32(b, 112), "unit": "raw native u32", "status": "guarded"},
            "simClock": {"value": u32(b, 116), "unit": "raw native u32", "status": "guarded"},
            "frameCurrent": {"value": u32(b, 144), "unit": "raw native u32", "status": "guarded"},
            "gClock": {
                "value": u32(b, 128) if mask & 128 else None,
                "unit": "separate raw u32",
                "status": "read" if mask & 128 else "unavailable",
            },
            "nativeGlobal": {
                "rawValue": u32(b, 132),
                "unit": "separate raw u32",
                "status": "mandatory producer read; not a step counter",
            },
            "dt": u32(b, 136),
            "frameLimitRaw": u32(b, 120),
        }
        available = mask & 65 == 65
        physical = numeric_group(NumericGroupSpec(b=b, o=152, n=10, available=available))
        if physical["value"] is not None:
            v = physical["value"]
            physical["value"] = {
                "orientationWXYZ": v[:4],
                "positionXYZ": v[4:7],
                "linearVelocityXYZ": v[7:10],
            }
            physical["quaternionNorm"] = math.sqrt(sum(x * x for x in v[:4]))
            physical["unitQuaternionWithinTolerance"] = abs(physical["quaternionNorm"] - 1) <= 0.01
        else:
            physical["unitQuaternionWithinTolerance"] = None
        inp = numeric_group(NumericGroupSpec(b=b, o=88, n=4, available=available))
        rpm = numeric_group(NumericGroupSpec(b=b, o=192, n=1, available=available))
        mirror = numeric_group(NumericGroupSpec(b=b, o=200, n=1, available=available))
        result["physical"] = physical
        result["vehicleTelemetry"] = {
            "inputsGasBrakeSteerFiltered": inp,
            "rpmRaw": rpm,
            "gearRawU32": u32(b, 196) if available else None,
            "rpmMirrorRaw": mirror,
            "gearMirrorRawU32": u32(b, 204) if available else None,
            "gearChangeMarkerRawU32": u32(b, 208) if available else None,
            "rawBitsAvailable": available,
            "repeatedReadsEqual": bool(mask & 2),
            "markerIsFreshnessClock": False,
            "inputIsBrakingBoolean": None,
        }
        result["readBoundaryProfileReady"] = (
            mask == 203
            and all(x["status"] == "available" for x in (physical, inp, rpm, mirror))
            and (physical["unitQuaternionWithinTolerance"] is True)
        )
    else:
        role = u32(b, 208)
        expected_role = 1 if mask & 447 == 447 else 0
        if role != expected_role:
            raise ValueError("reader role inconsistent with validity mask")
        raw_available = bool(mask & 128)
        if raw_available and (not u32(b, 112) < u32(b, 116) <= 200000):
            raise ValueError("record bounds inconsistent with raw32 bit")
        result["readerRole"] = role
        result["timestamps"] = {
            "callerIteration": {
                "rawValue": u32(b, 120),
                "value": u32(b, 120) if mask & 1 else None,
                "status": "guard-correlated read" if mask & 1 else "unavailable",
            },
            "simClock": {
                "rawValue": u32(b, 124),
                "value": None,
                "status": "valueOrZero; no dedicated success bit, ownership bit4 is separate",
            },
            "nativeGlobal": {
                "rawValue": u32(b, 132),
                "value": None,
                "status": "valueOrZero; no dedicated success bit",
            },
            "gClock": {"value": None, "status": "not present in TMREAD01"},
            "headerWordRaw": {
                "value": u32(b, 128) if raw_available else None,
                "status": "record header, not timestamp",
            },
        }
        result["readerIndex"] = {
            "rawValue": u32(b, 112),
            "qualifiedValue": u32(b, 112) if role == 1 else None,
            "countRaw": u32(b, 116),
            "preparedCountRaw": u64(b, 104),
            "contextCountRaw": u32(b, 188),
            "gLayoutHint": bool(mask & 64),
            "gLayoutHintIsNotClockValidity": True,
        }
        if raw_available:
            result["rawRecord"] = {
                "hex": b[152:184].hex(),
                "headerWord": u32(b, 152),
                "channelsSteerGasBrake": numeric_group(
                    NumericGroupSpec(b=b, o=156, n=3, available=True)
                ),
                "sourceOwnershipQualified": role == 1,
            }
        if role == 1:
            result["selectedInput"] = {
                "index": u32(b, 112),
                "channelsSteerGasBrake": numeric_group(
                    NumericGroupSpec(b=b, o=156, n=3, available=True)
                ),
                "stage": "selected for upcoming native read; not completed physical transition",
                "shadowSteerBits": u32(b, 184) if mask & 32 else None,
            }
    return result


def decode_response(b: Any) -> Any:
    """Decode an already captured fixed response; never establish current live ownership."""
    if len(b) == 448:
        return decode_v4_response(b) if b[:4] == b"TMR4" else decode_v3_response(b)
    if len(b) not in (272, 320):
        raise ValueError("unsupported response size")
    v2 = len(b) == 320
    if b[:4] != (b"TMR2" if v2 else b"TMR1") or b[4] != (2 if v2 else 1) or b[6:8] != b"\x00\x00":
        raise ValueError("response header mismatch")
    kind = b[5]
    if kind not in ((1, 2, 3, 4) if v2 else (1, 2)):
        raise ValueError("unknown response kind")
    offset = 32 if v2 else 16
    if v2 and u32(b, 24) != 288:
        raise ValueError("response payload size mismatch")
    event = decode_event(b[offset : offset + 256])
    if (kind in (1, 2)) != (event["eventType"] == "TMPOST03"):
        raise ValueError("response kind/event union mismatch")
    if kind in (1, 2) and event["validityMask"] != 203:
        raise ValueError("wire boundary requires mask203")
    if kind in (3, 4) and event["readerRole"] != 1:
        raise ValueError("wire reader requires role1")
    receipt = b[288:320] if v2 else bytes(32)
    if v2 and (u32(b, 28) != (1 if kind == 4 else 0) or (kind != 4 and any(receipt))):
        raise ValueError("response flags/receipt mismatch")
    return {
        "version": 2 if v2 else 1,
        "kind": kind,
        "sequence": u32(b, 8),
        "iterations": u32(b, 12),
        "epoch": u64(b, 16) if v2 else None,
        "event": event,
        "receiptHex": receipt.hex() if kind == 4 else None,
        "physicalTransitionCandidate": kind == 1,
        "heldNow": None,
        "requiresLiveOwnerAndEpochCheck": True,
        "samePreWriteReaderEvent": kind == 4,
        "runAckIsNewState": False,
    }


def _decode_plan_response(b: Any, version: Any) -> Any:
    """Pinned V3 receipt structure/domain checks; not evidence of current live hold."""
    if len(b) != 448:
        raise ValueError("plan response must be exactly448B")
    if (
        version not in (3, 4)
        or b[:4] != (b"TMR3" if version == 3 else b"TMR4")
        or b[4] != version
        or (b[6:8] != b"\x00\x00")
    ):
        raise ValueError("plan response header mismatch")
    kind = b[5]
    if kind not in (1, 2, 3, 4):
        raise ValueError("plan unknown response kind")
    if u32(b, 24) != 416:
        raise ValueError("plan payload must be416B")
    if u32(b, 28) != (1 if kind == 4 else 0):
        raise ValueError("plan flags inconsistent with kind")
    event = decode_event(b[32:288])
    receipt = b[288:448]
    if (kind in (1, 2)) != (event["eventType"] == "TMPOST03"):
        raise ValueError("plan response kind/event union mismatch")
    required_mask = 203 if version == 3 else 75
    if kind in (1, 2) and event["validityMask"] != required_mask:
        raise ValueError("plan boundary mask does not match selected version")
    event["readNativeSimBoundaryProfileReady"] = False
    if version == 4 and kind in (1, 2):
        raw = b[32:288]
        if u32(raw, 112) != u32(raw, 116) or u32(raw, 128) != 0:
            raise ValueError("V4 requires direct nativeSIM clock and absent legacy Gclock")
        event["timestamps"]["nativeSimClock"] = {
            "value": u32(raw, 116),
            "source": "capturedSim+0x48",
            "unit": "raw native u32; no race conversion",
            "status": "pinnedV4 boundary source",
        }
        groups = [
            event["physical"],
            event["vehicleTelemetry"]["inputsGasBrakeSteerFiltered"],
            event["vehicleTelemetry"]["rpmRaw"],
            event["vehicleTelemetry"]["rpmMirrorRaw"],
        ]
        event["readNativeSimBoundaryProfileReady"] = (
            all(g["status"] == "available" for g in groups)
            and event["physical"]["unitQuaternionWithinTolerance"] is True
        )
        assert event["readBoundaryProfileReady"] is False
        assert event["timestamps"]["gClock"]["value"] is None
    if kind in (3, 4) and event["readerRole"] != 1:
        raise ValueError("plan reader requires role1")
    result = {
        "version": version,
        "kind": kind,
        "sequence": u32(b, 8),
        "iterations": u32(b, 12),
        "epoch": u64(b, 16),
        "event": event,
        "receiptHex": receipt.hex() if kind == 4 else None,
        "receipt": None,
        "physicalTransitionCandidate": kind == 1,
        "heldNow": None,
        "requiresLiveOwnerAndEpochCheck": True,
        "samePreWriteReaderEvent": kind == 4,
        "runAckIsNewState": False,
    }
    if (
        result["epoch"] == 0
        or result["iterations"] > 1000
        or (kind in (2, 3) and result["iterations"] != 0)
    ):
        raise ValueError("plan epoch/iterations outside pinned response domain")
    if kind != 4:
        if any(receipt):
            raise ValueError("plan noncommit receipt must be zero")
        return result
    p, k, length = (u32(receipt, 0), u32(receipt, 4), u32(receipt, 8))
    if not 1 <= length <= 5:
        raise ValueError("plan commit length outside1..5")
    if any(receipt[12:16]) or any(receipt[136:160]):
        raise ValueError("plan reserved receipt bytes nonzero")
    if any(receipt[16 + length * 24 : 136]):
        raise ValueError("plan unused receipt slot nonzero")
    if not 1 <= result["iterations"] <= 1000:
        raise ValueError("plan commit iterations outside1..1000")
    if (
        p != event["selectedInput"]["index"]
        or k <= p
        or k + length > event["readerIndex"]["countRaw"]
    ):
        raise ValueError("plan commit P/k/range inconsistent with reader")
    slots = []
    for j in range(length):
        at = 16 + j * 24
        before = receipt[at : at + 12]
        after = receipt[at + 12 : at + 24]
        steer = struct.unpack_from("<f", after)[0]
        gas = u32(after, 4)
        brake = u32(after, 8)
        if (
            not math.isfinite(steer)
            or not -1.0 <= steer <= 1.0
            or gas not in (0, 1065353216)
            or (brake not in (0, 1065353216))
        ):
            raise ValueError("plan after channels violate pinned native domain")
        slots.append(
            {
                "ordinal": j,
                "index": k + j,
                "before12Hex": before.hex(),
                "after12Hex": after.hex(),
                "beforeSteerGasBrake": numeric_group(
                    NumericGroupSpec(b=before, o=0, n=3, available=True)
                ),
                "afterSteerGasBrake": numeric_group(
                    NumericGroupSpec(b=after, o=0, n=3, available=True)
                ),
            }
        )
    result["receipt"] = {
        "ownP": p,
        "firstIndex": k,
        "length": length,
        "slots": slots,
        "stage": "memory write/readback receipt, not complete physical transition",
        "iterationsAreIndependentOfPlanLength": True,
    }
    return result


def decode_v3_response(b: Any) -> Any:
    return _decode_plan_response(b, 3)


def decode_v4_response(b: Any) -> Any:
    return _decode_plan_response(b, 4)
