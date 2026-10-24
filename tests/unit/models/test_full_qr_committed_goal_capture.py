"""Real pilot547 failure: sampled physics and committed score have distinct clocks."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.goal_delivery import NativeGoalContext
from trackmaniarl.trackmania.native_headless.full_qr.terminal_capture import (
    decode_committed_goal_capture,
    decode_partial_goal_capture,
)

FIXTURE = Path(__file__).parents[2] / "fixtures/full_qr/terminal_committed554"


def _arguments() -> dict[str, Any]:
    context = json.loads((FIXTURE / "context.json").read_text())
    for name, digest in context["files"].items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == digest
    return {
        "vehicle_raw": (FIXTURE / "vehicle.bin").read_bytes(),
        "goal_facts_raw": (FIXTURE / "goals.bin").read_bytes(),
        "goal_receipt_raw": (FIXTURE / "goal-receipt.bin").read_bytes(),
        "context": NativeGoalContext(context["pid"], context["meta"]),
        "previous_native_t": context["previous_native_t"],
        "held_native_t": context["held_native_t"],
    }


def test_actual_committed_score_does_not_change_the_physical_sample_clock() -> None:
    result = decode_committed_goal_capture(**_arguments())
    assert result.vehicle.native_t == 54990
    assert result.fact.event_native_t == 54992
    assert result.fact.dispatch_native_t == 54990
    assert result.fact.callback_native_t == 54980
    assert result.goal.goal_active is False
    assert result.goal.client_ui_finish_proven is False


def test_existing_occurred_goal_guard_still_rejects_this_real_capture() -> None:
    with pytest.raises(ValueError, match="actual occurred Goal"):
        decode_partial_goal_capture(**_arguments())


@pytest.mark.parametrize("delta", [0, 10, 100])
def test_arbitrary_future_goal_is_not_a_committed_subtick(delta: int) -> None:
    result = decode_committed_goal_capture(**_arguments())
    event = result.vehicle.native_t + delta
    with pytest.raises(ValueError, match="matching committed Goal"):
        replace(
            result,
            fact=replace(result.fact, event_native_t=event),
            goal=replace(result.goal, last_goal_native_t=event),
        )


def test_wrong_dispatch_and_callback_are_rejected() -> None:
    result = decode_committed_goal_capture(**_arguments())
    for field in ("dispatch_native_t", "callback_native_t"):
        with pytest.raises(ValueError, match="matching committed Goal"):
            replace(result, fact=replace(result.fact, **{field: 54970}))


def test_entry_without_matching_store_is_not_a_committed_goal() -> None:
    arguments = _arguments()
    arguments["goal_facts_raw"] = arguments["goal_facts_raw"][:768]
    with pytest.raises(ValueError, match="entry/store"):
        decode_committed_goal_capture(**arguments)
