"""Actual pilot524 partial final capture; ordinary 50 ms guards stay strict."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.goal_delivery import NativeGoalContext
from trackmaniarl.trackmania.native_headless.full_qr.native_features import NativeGraphFrame
from trackmaniarl.trackmania.native_headless.full_qr.terminal_capture import (
    PartialGoalCapture,
    decode_partial_goal_capture,
)

FIXTURE = Path(__file__).parents[2] / "fixtures/full_qr/terminal_partial537"


def _decode(**overrides: Any) -> PartialGoalCapture:
    context = json.loads((FIXTURE / "context.json").read_text())
    for name, digest in context["files"].items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == digest
    arguments = {
        "vehicle_raw": (FIXTURE / "vehicle.bin").read_bytes(),
        "goal_facts_raw": (FIXTURE / "goals.bin").read_bytes(),
        "goal_receipt_raw": (FIXTURE / "goal-receipt.bin").read_bytes(),
        "context": NativeGoalContext(context["pid"], context["meta"]),
        "previous_native_t": context["previous_native_t"],
        "held_native_t": context["held_native_t"],
    }
    arguments.update(overrides)
    return decode_partial_goal_capture(**arguments)


def test_actual_final_40ms_capture_does_not_manufacture_next_boundary() -> None:
    result = _decode()
    assert result.observed_interval_ms == 40
    assert result.vehicle.native_t == 71990
    assert result.goal.last_goal_native_t == 71951
    assert result.goal.client_ui_finish_proven is False
    with pytest.raises(ValueError, match="50ms"):
        NativeGraphFrame((69790.0, *([0.0] * 22)))


@pytest.mark.parametrize("held_native_t", [71940, 72000])
def test_wrong_or_invented_held_boundary_is_rejected(held_native_t: int) -> None:
    with pytest.raises(ValueError, match="actual partial interval"):
        _decode(held_native_t=held_native_t)


def test_empty_goal_stream_is_not_negative_or_terminal_coverage() -> None:
    with pytest.raises(ValueError, match="prefix"):
        _decode(goal_facts_raw=b"")


def test_foreign_episode_binding_is_rejected() -> None:
    context = json.loads((FIXTURE / "context.json").read_text())
    with pytest.raises(ValueError, match="owner mismatch"):
        _decode(context=NativeGoalContext(context["pid"] + 1, context["meta"]))
