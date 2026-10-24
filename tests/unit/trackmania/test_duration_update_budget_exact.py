"""Long accounting uses recorded durations, without accumulated binary drift."""

from __future__ import annotations

import copy
import math
from fractions import Fraction
from typing import Any

import pytest

from tests.unit.trackmania.test_common_training import contract, source
from trackmaniarl.core.data import Transition
from trackmaniarl.trackmania.common_training import DurationUpdateBudget, make_transition, replay

NUMERATOR = "elapsed_seconds_exact_numerator"
DENOMINATOR = "elapsed_seconds_exact_denominator"


def fixture_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[DurationUpdateBudget, dict[int, Transition]]:
    selected = contract()
    rows = {n: make_transition(source(elapsed_ms=n * 10), selected) for n in range(1, 6)}
    # Validate each fixture with the production validator before installing the
    # narrow test-only stub. The long loops still call actual budget.credit,
    # while avoiding millions of duplicate tiny Torch validation operations.
    validated = {id(row): replay.validate_transition(row, selected) for row in rows.values()}

    def already_validated(row: Transition, spec: Any) -> float:
        assert spec is selected
        return validated[id(row)]

    monkeypatch.setattr(replay, "validate_transition", already_validated)
    return DurationUpdateBudget(selected), rows


@pytest.mark.parametrize("count", [100_000, 1_000_000])
@pytest.mark.parametrize("pattern", [(1,), (5,), (1, 5, 2, 4, 3)])
def test_long_credit_sequence_matches_integer_oracle_at_every_transition(
    monkeypatch: pytest.MonkeyPatch, count: int, pattern: tuple[int, ...]
) -> None:
    budget, rows = fixture_rows(monkeypatch)
    intervals = 0
    for index in range(count):
        n = pattern[index % len(pattern)]
        budget.credit(rows[n])
        intervals += n
        # Each recorded .01 second earns .05 updates; twenty intervals earn one.
        assert budget.integral_available == intervals // 20
    assert budget.elapsed_seconds == float(Fraction(intervals, 100))
    assert budget.transitions == count
    state = budget.state_dict()
    assert Fraction(state[NUMERATOR], state[DENOMINATOR]) == Fraction(intervals, 100)


def test_n5_transition_12488_does_not_delay_a_duration_earned_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget, rows = fixture_rows(monkeypatch)
    legacy = 0.0
    for _ in range(12_488):
        legacy += 0.05
        budget.credit(rows[5])
    assert math.floor(legacy / 0.05 * 0.25 + 1e-10) == 3121
    assert budget.integral_available == 3122
    assert budget.elapsed_seconds == 624.4


def test_first_2048_n5_update_steps_match_legacy_schedule_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget, rows = fixture_rows(monkeypatch)
    legacy_elapsed, legacy_updates = 0.0, 0
    old_steps, new_steps = [], []
    for step in range(1, 2049):
        legacy_elapsed += 0.05
        if math.floor(max(0, legacy_elapsed / 0.05 * 0.25 - legacy_updates) + 1e-10):
            legacy_updates += 1
            old_steps.append(step)
        budget.credit(rows[5])
        if budget.integral_available:
            budget.consume()
            new_steps.append(step)
    assert new_steps == old_steps == list(range(4, 2049, 4))
    assert budget.updates == legacy_updates == 512


def test_exact_credit_and_entitlement_survive_repeated_checkpoint_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uninterrupted, rows = fixture_rows(monkeypatch)
    resumed = DurationUpdateBudget(uninterrupted.contract)
    for index in range(30_000):
        row = rows[(index % 5) + 1]
        uninterrupted.credit(row)
        resumed.credit(row)
        if uninterrupted.integral_available:
            uninterrupted.consume()
            resumed.consume()
        if index % 997 == 0:
            saved = resumed.state_dict()
            resumed = DurationUpdateBudget(uninterrupted.contract)
            resumed.load_state_dict(saved)
        assert resumed.integral_available == uninterrupted.integral_available
        assert resumed.credit_available == uninterrupted.credit_available
    assert resumed.state_dict() == uninterrupted.state_dict()


def test_legacy_checkpoint_preserves_saved_approximation_without_reconstructing_past(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget, rows = fixture_rows(monkeypatch)
    # Match the original repeated += operation. Python 3.12's compensated sum()
    # intentionally does not reproduce that historical accumulation error.
    legacy_elapsed = 0.0
    for _ in range(2048):
        legacy_elapsed += 0.05
    legacy = {
        "common_contract": budget.contract.envelope(),
        "elapsed_seconds": legacy_elapsed,
        "transitions": 2048,
        "updates": 512,
    }
    budget.load_state_dict(legacy)
    saved = budget.state_dict()
    imported = Fraction(str(legacy_elapsed))
    assert imported != Fraction("102.4")
    assert Fraction(saved[NUMERATOR], saved[DENOMINATOR]) == imported
    assert saved["elapsed_seconds"] == legacy_elapsed
    assert budget.integral_available == 0
    for _ in range(4):
        budget.credit(rows[5])
    assert budget.integral_available == 1
    assert Fraction(budget.state_dict()[NUMERATOR], budget.state_dict()[DENOMINATOR]) == (
        imported + Fraction("0.2")
    )


@pytest.mark.parametrize(
    "change",
    [
        {NUMERATOR: None},
        {NUMERATOR: True, DENOMINATOR: 2},
        {NUMERATOR: -1, DENOMINATOR: 2},
        {NUMERATOR: 1, DENOMINATOR: 0},
        {NUMERATOR: 1, DENOMINATOR: -2},
        {NUMERATOR: 1, DENOMINATOR: False},
        {NUMERATOR: 1, DENOMINATOR: 2.0},
        {NUMERATOR: 2, DENOMINATOR: 40},
        {NUMERATOR: 1, DENOMINATOR: 10},
        {NUMERATOR: 1 << 4096, DENOMINATOR: 1},
        {NUMERATOR: 1, DENOMINATOR: 1 << 4096},
        {NUMERATOR: 1 << 4095, DENOMINATOR: 1},
    ],
)
def test_malformed_exact_checkpoint_is_rejected_without_mutating_live_budget(
    monkeypatch: pytest.MonkeyPatch, change: dict[str, Any]
) -> None:
    budget, rows = fixture_rows(monkeypatch)
    budget.credit(rows[5])
    before = copy.deepcopy(budget.state_dict())
    invalid = {**before, **change}
    with pytest.raises(ValueError, match=r"exact elapsed|exact elapsed integers"):
        budget.load_state_dict(invalid)
    assert budget.state_dict() == before


@pytest.mark.parametrize("missing", [NUMERATOR, DENOMINATOR])
def test_partial_exact_state_cannot_silently_fall_back_to_legacy(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    budget, rows = fixture_rows(monkeypatch)
    budget.credit(rows[5])
    invalid = budget.state_dict()
    del invalid[missing]
    with pytest.raises(ValueError, match="both exact elapsed"):
        budget.load_state_dict(invalid)


def test_zero_exact_state_and_entitlement_keep_existing_checkpoint_invariants() -> None:
    budget = DurationUpdateBudget(contract())
    initial = budget.state_dict()
    budget.load_state_dict(initial)
    assert budget.state_dict() == initial
    for change in (
        {"transitions": 1},
        {"updates": 1},
        {NUMERATOR: 1, DENOMINATOR: 10**400},
    ):
        with pytest.raises(ValueError, match="invalid duration budget"):
            budget.load_state_dict({**initial, **change})
        assert budget.state_dict() == initial


def test_production_credit_still_validates_actual_row_before_any_state_change() -> None:
    selected = contract()
    budget = DurationUpdateBudget(selected)
    row = make_transition(source(), selected)
    before = budget.state_dict()
    with pytest.raises(ValueError, match="transition reward differs"):
        budget.credit(
            Transition(
                row.observation,
                row.action,
                row.reward + 1,
                row.next_observation,
                row.terminated,
                row.truncated,
                row.info,
            )
        )
    assert budget.state_dict() == before
