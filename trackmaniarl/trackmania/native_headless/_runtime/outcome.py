from __future__ import annotations

import struct
from typing import Any

from trackmaniarl.trackmania.native_headless._runtime import validator
from trackmaniarl.trackmania.native_headless._runtime.shapes import ParseRecordsSpec


def u(b: Any, o: Any) -> Any:
    return struct.unpack_from("<I", b, o)[0]


def q(b: Any, o: Any) -> Any:
    return struct.unpack_from("<Q", b, o)[0]


MAP_UID = "oqIJ5rQDRrNwLPTh9H2p_W4tLof"


def classify(packet: Any, env_module: Any, root: Any) -> Any:
    result: dict[str, Any] = {
        "status": "inconclusive",
        "terminated": None,
        "truncated": None,
        "computedResult": None,
        "validationAcceptance": None,
        "physicalFinishNativeT": None,
    }
    try:
        checks = env_module
        evidence = packet["resultEvidence"]
        checks.require(
            packet["profile"] == checks.RUNTIME_PROFILE
            and evidence["profile"] == checks.RUNTIME_PROFILE,
            "terminal runtime profile mismatch",
        )
        checks.require(evidence["pins"] == checks.PINS, "terminal producer identity mismatch")
        reaped = evidence["reaped"]
        checks.require(
            reaped["pid"] == packet["pid"]
            and reaped["kind"] == "reaped"
            and (reaped["process_gone"] is True)
            and (reaped["parent_unchanged"] is True)
            and (reaped["exit_status"] is None)
            and (reaped["truncated_by_driver"] is False),
            "normal reap proof",
        )
        raw_json = bytes.fromhex(evidence["validatorArrayHex"])
        record = validator.parse_records(
            ParseRecordsSpec(
                text=raw_json.decode("utf-8"),
                expected_files={"g.Ghost.Gbx"},
                expected_map_uid=MAP_UID,
                mode="parent",
                allow_inherited_array_close=False,
            )
        )[0]
        result["validationAcceptance"] = record["IsValid"]
        value = record["ValidatedResult"]
        checks.require(value is not None, "computed result unavailable; not DNF")
        checks.require(
            all(type(v) is int and 0 <= v < 2**32 for v in value.values())
            and value["Time"] != 4294967295
            and (value["NbCheckpoints"] == 1),
            "computed finish not established",
        )
        raw = bytes.fromhex(evidence["collectorHex"])
        checks.require(
            raw and len(raw) % 768 == 0 and (len(raw) <= 4096 * 768),
            "complete result collector768 required",
        )
        endpoint = bytes.fromhex(packet["boundaryHex"])
        _, end_t = checks.qualify_post(endpoint, packet["pid"], packet["meta"])
        records = [raw[i : i + 768] for i in range(0, len(raw), 768)]
        for hit, b in enumerate(records, 1):
            checks.require(
                b[:8] == b"TMRESULT"
                and (
                    u(b, 8),
                    u(b, 12),
                    u(b, 16),
                    u(b, 20),
                    q(b, 24),
                    u(b, 148),
                    u(b, 240),
                    u(b, 248),
                )
                == (768, 1, packet["pid"], packet["pid"], hit, 24575, 13, 1),
                "complete own TMRESULT profile",
            )
            checks.require(
                q(b, 40) == packet["meta"]["controller"] and q(b, 48) == packet["meta"]["sim"],
                "result owner chain",
            )
        last = records[-1]
        checks.require(u(last, 128) == end_t, "result endpoint clock binding")
        checks.require(
            (
                u(last, 152),
                u(last, 156),
                u(last, 160),
                u(last, 184),
                u(last, 192),
                u(last, 256),
                u(last, 260),
            )
            == (value["Time"], value["Score"], value["NbRespawns"], 1, 8, 0, value["Time"]),
            "validator/TMRESULT mismatch",
        )
        result.update(
            status="terminated",
            terminated=True,
            truncated=False,
            computedResult=value,
            qualifiedBy="env002.full-original-validator+TMRESULT",
        )
    except (KeyError, TypeError, ValueError, UnicodeError, struct.error, AssertionError) as error:
        result["reason"] = str(error)
    return result
