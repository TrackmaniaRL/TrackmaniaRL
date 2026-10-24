"""Release EOF cannot substitute for a full, correlated original finish result."""

import json
import struct

import pytest

from scripts.headless_finish_drain import inspect_native_drain
from trackmaniarl.trackmania.native_headless._runtime.outcome import MAP_UID

PID = 420
META = {"controller": 0x10000, "sim": 0x20000, "start_offset_ms": -1510}


def post(native_t: int) -> bytes:
    raw = bytearray(256)
    raw[:8] = b"TMPOST03"
    for offset, value in (
        (8, 256),
        (12, 3),
        (16, PID),
        (20, PID),
        (112, native_t),
        (116, native_t),
        (136, 10),
        (140, 10),
        (144, native_t),
        (148, 75),
        (212, 10),
    ):
        struct.pack_into("<I", raw, offset, value)
    struct.pack_into("<QQ", raw, 40, META["controller"], META["sim"])
    return bytes(raw)


def result(*, finish_time: int = 17715, checkpoints: int = 1) -> bytes:
    raw = bytearray(768)
    raw[:8] = b"TMRESULT"
    for offset, value in (
        (8, 768),
        (12, 1),
        (16, PID),
        (20, PID),
        (128, 20000),
        (148, 24575),
        (152, finish_time),
        (184, checkpoints),
        (192, checkpoints * 8),
        (240, 13),
        (248, 1),
    ):
        struct.pack_into("<I", raw, offset, value)
    struct.pack_into("<Q", raw, 24, 1)
    struct.pack_into("<QQ", raw, 40, META["controller"], META["sim"])
    if checkpoints:
        struct.pack_into("<II", raw, 256, 0, finish_time)
    return bytes(raw)


def validator(*, value: dict | None = None) -> bytes:
    if value is None:
        value = {"Time": 17715, "Score": 0, "NbRespawns": 0, "NbCheckpoints": 1}
    return json.dumps(
        [
            {
                "MapUid": MAP_UID,
                "FileName": "g.Ghost.Gbx",
                "IsValid": False,
                "Desc": "declared40000 differs from computed17715",
                "DeclaredResult": {"Time": 40000, "Score": 0, "NbRespawns": 0, "NbCheckpoints": 1},
                "ValidatedResult": value,
            }
        ]
    ).encode()


def test_full_computed_finish_is_distinct_from_ghost_validation_acceptance() -> None:
    report = inspect_native_drain(validator(), post(20000), result(), PID, META)
    assert report["computed_finish"]
    assert report["computed_result_correlated"]
    assert report["validation_acceptance"] is False
    assert report["computed_dnf"] is False
    assert report["outcome"] == "physical-finish"


def test_complete_null_computed_result_stays_inconclusive() -> None:
    record = json.loads(validator())
    record[0]["ValidatedResult"] = None
    record[0]["Desc"] = "wrong simu"
    report = inspect_native_drain(json.dumps(record).encode(), post(20000), result(), PID, META)
    assert report["computed_finish"] is False
    assert report["computed_dnf"] is False
    assert report["computed_result_correlated"] is False


def test_actual_complete_dnf_is_not_a_finish() -> None:
    value = {"Time": 4294967295, "Score": 0, "NbRespawns": 0, "NbCheckpoints": 0}
    report = inspect_native_drain(
        validator(value=value),
        post(20000),
        result(finish_time=4294967295, checkpoints=0),
        PID,
        META,
    )
    assert report["computed_dnf"]
    assert report["computed_finish"] is False


@pytest.mark.parametrize("raw", [b"", b"[\n", validator()[:-1]])
def test_socket_or_incomplete_json_cannot_qualify_a_finish(raw: bytes) -> None:
    with pytest.raises(json.JSONDecodeError):
        inspect_native_drain(raw, post(20000), result(), PID, META)


def test_collector_result_must_match_exact_validator_time() -> None:
    with pytest.raises(ValueError, match="disagree"):
        inspect_native_drain(validator(), post(20000), result(finish_time=17716), PID, META)


def test_result_clock_must_have_a_measured_native_post() -> None:
    with pytest.raises(ValueError, match="no actual native POST"):
        inspect_native_drain(validator(), post(19990), result(), PID, META)


def test_missing_native_clock_tick_is_rejected() -> None:
    with pytest.raises(ValueError, match="consecutive"):
        inspect_native_drain(validator(), post(19980) + post(20000), result(), PID, META)


def test_truncated_result_or_unqualified_input_write_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"partial768|partial 768"):
        inspect_native_drain(validator(), post(20000), result()[:-1], PID, META)
    audit = bytearray(post(20000))
    audit[:8] = b"TMAUD003"
    with pytest.raises(ValueError, match="write audits"):
        inspect_native_drain(validator(), post(20000) + bytes(audit), result(), PID, META)


def test_foreign_result_owner_is_rejected() -> None:
    raw = bytearray(result())
    struct.pack_into("<Q", raw, 48, META["sim"] + 8)
    with pytest.raises(ValueError, match="owner"):
        inspect_native_drain(validator(), post(20000), bytes(raw), PID, META)
