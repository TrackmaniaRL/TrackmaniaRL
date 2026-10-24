"""Real collector55000/Goal55007 source; no relaxed old held-reader contract."""

import hashlib
import json
from pathlib import Path
from struct import pack_into

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.final_capture import (
    decode_final_collector_boundary,
)
from trackmaniarl.trackmania.native_headless.full_qr.goal_delivery import NativeGoalContext

FIXTURE = Path(__file__).parents[2] / "fixtures/full_qr/final_collector580"


def _sources() -> tuple[bytes, bytes, NativeGoalContext]:
    context = json.loads((FIXTURE / "context.json").read_text())
    for name, digest in context["files"].items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == digest
    return (
        (FIXTURE / "collector.bin").read_bytes(),
        (FIXTURE / "goals.bin").read_bytes(),
        NativeGoalContext(context["pid"], context["meta"]),
    )


def test_actual_final_capture_proves_an_additional_native_tick() -> None:
    raw, goals, context = _sources()
    result = decode_final_collector_boundary(raw, goals, context=context)
    assert result.previous_native_t == 54990
    assert result.native_t == 55000
    assert result.goal_native_t == 55007


@pytest.mark.parametrize(("offset", "value"), [(32, 55010), (36, 54980), (76, 55008), (108, 0)])
def test_changed_clock_or_unfinished_capture_is_rejected(offset: int, value: int) -> None:
    raw, goals, context = _sources()
    changed = bytearray(raw)
    pack_into("<I", changed, offset, value)
    with pytest.raises(ValueError, match=r"mismatch|required"):
        decode_final_collector_boundary(bytes(changed), goals, context=context)


def test_collector_primary_record_cannot_be_replaced_by_marker_alone() -> None:
    raw, goals, context = _sources()
    with pytest.raises(ValueError, match="primary record"):
        decode_final_collector_boundary(raw[:128] + bytes(768), goals, context=context)
    with pytest.raises(ValueError, match="matching final Goal"):
        decode_final_collector_boundary(raw, b"", context=context)


def test_foreign_process_and_bare_eof_cannot_supply_final_capture() -> None:
    raw, goals, context = _sources()
    with pytest.raises(ValueError, match="identity"):
        decode_final_collector_boundary(
            raw, goals, context=NativeGoalContext(context.pid + 1, context.meta)
        )
    with pytest.raises(ValueError, match="exact completed"):
        decode_final_collector_boundary(b"", goals, context=context)
