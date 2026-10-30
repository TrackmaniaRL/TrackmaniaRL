"""Per-call validation sharing preserves all row evidence and batch bytes."""

from __future__ import annotations

import copy
import random
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, asdict, replace
from typing import Any, cast

import numpy as np
import pytest
import torch

from tests.unit.trackmania.test_common_training import contract, source
from tests.unit.trackmania.test_common_training_performance import _assert_exact
from trackmaniarl.core.data import TrainingBatch, Transition
from trackmaniarl.trackmania.common_training import (
    COMMON6_PROFILE,
    CommonTrainingContract,
    features,
    make_transition,
    replay,
)
from trackmaniarl.trackmania.common_training import contract as contract_module
from trackmaniarl.trackmania.common_training.contract import SCHEMA


def _legacy_collate(
    rows: list[Transition], identities: list[int], selected: CommonTrainingContract
) -> TrainingBatch:
    """Pre-context collator: independently validate each row, then build tensors."""
    durations = [features.validate_transition(row, selected) for row in rows]
    indices = [index for index, row in enumerate(rows) if not row.terminated]
    next_observations = (
        torch.stack([cast(torch.Tensor, rows[index].next_observation) for index in indices])
        if indices
        else torch.empty((0, selected.feature_dim), dtype=torch.float32)
    )
    return TrainingBatch(
        data={"bootstrap_row_indices": torch.tensor(indices, dtype=torch.long)},
        observations=torch.stack([cast(torch.Tensor, row.observation) for row in rows]),
        actions=torch.tensor([row.action for row in rows], dtype=torch.long),
        rewards=torch.tensor([row.reward for row in rows], dtype=torch.float32),
        next_observations=next_observations,
        terminated=torch.tensor([row.terminated for row in rows], dtype=torch.bool),
        truncated=torch.tensor([row.truncated for row in rows], dtype=torch.bool),
        bootstrap_discounts=torch.tensor(
            [
                0.0
                if row.terminated
                else selected.gamma ** (elapsed / selected.discount_reference_seconds)
                for row, elapsed in zip(rows, durations, strict=True)
            ],
            dtype=torch.float32,
        ),
        transition_ids=identities,
        metadata={
            "common_ragged_contract": SCHEMA,
            "contract_sha256": selected.digest,
            "n_step": 1,
            "sequence_length": 1,
            "gamma": selected.gamma,
            "elapsed_seconds": durations,
            "reward_measurements": [
                row.info[features.COMMON_INTERVAL_KEY]["reward_measurement"] for row in rows
            ],
        },
    )


@pytest.mark.parametrize("profile", [None, COMMON6_PROFILE])
@pytest.mark.parametrize("all_terminal", [False, True])
def test_batch_matches_independent_row_collation_without_changing_rng(
    *, profile: str | None, all_terminal: bool
) -> None:
    selected = contract(**({"observation_profile": profile} if profile else {}))
    rows = [
        make_transition(
            replace(
                source(
                    backend="native-headless" if index % 2 else "windows-connect33",
                    terminated=all_terminal or index % 3 == 0,
                    truncated=not all_terminal and index % 3 == 1,
                    elapsed_ms=10 * (index % 5 + 1),
                ),
                action=(index * 17) % 78,
            ),
            selected,
        )
        for index in range(32)
    ]
    identities = list(range(100, 132))
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    expected = _legacy_collate(rows, identities, selected)
    actual = replay.collate_rows(rows, identities, selected)
    for name in TrainingBatch.__dataclass_fields__:
        _assert_exact(getattr(actual, name), getattr(expected, name))
    after = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    _assert_exact(before, after)


def test_context_is_immutable_per_call_and_serializes_expected_contract_twice_per_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = contract()
    rows = [make_transition(source(), selected)] * 32
    original = CommonTrainingContract.envelope
    calls = []

    def counted(self: CommonTrainingContract) -> dict[str, Any]:
        calls.append(self)
        return original(self)

    monkeypatch.setattr(CommonTrainingContract, "envelope", counted)
    replay.collate_rows(rows, list(range(32)), selected)
    assert len(calls) == 2  # Initial snapshot and final full contract recheck.
    calls.clear()
    _legacy_collate(rows, list(range(32)), selected)
    assert len(calls) == 33
    first, second = features._validation_context(selected), features._validation_context(selected)
    assert first is not second
    with pytest.raises(FrozenInstanceError):
        first.digest = "changed"  # type: ignore[misc]
    assert asdict(first) == asdict(second)


@pytest.mark.parametrize("policy", ["MODEL_SPEC", "COMMON_SCALES", "COMMON_FIELDS"])
def test_persistent_policy_mutation_during_collation_is_rejected_before_return(
    monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    selected = contract()
    rows = [make_transition(source(), selected)] * 3
    original = replay._validate_transition_record
    calls = []

    def changed(
        row: Transition, expected: features._TransitionValidation
    ) -> tuple[Mapping[str, Any], float]:
        validated = original(row, expected)
        calls.append(row)
        if len(calls) == 1:
            if policy == "MODEL_SPEC":
                monkeypatch.setattr(
                    contract_module, policy, {**contract_module.MODEL_SPEC, "action_count": 79}
                )
            elif policy == "COMMON_SCALES":
                monkeypatch.setattr(
                    contract_module, policy, (999.0, *contract_module.COMMON_SCALES[1:])
                )
            else:
                monkeypatch.setattr(
                    contract_module, policy, ("changed", *contract_module.COMMON_FIELDS[1:])
                )
        return validated

    monkeypatch.setattr(replay, "_validate_transition_record", changed)
    with pytest.raises(ValueError, match="contract changed during batch collation"):
        replay.collate_rows(rows, [0, 1, 2], selected)
    assert len(calls) == 3
    # Public calls always obtain fresh policy, independent of the earlier batch.
    with pytest.raises(ValueError, match="common interval contract mismatch"):
        features.validate_transition(rows[0], selected)
    with pytest.raises(ValueError, match="common interval contract mismatch"):
        features.validate_interval(rows[0].info[features.COMMON_INTERVAL_KEY], selected)


def _corrupt(row: Transition, name: str) -> Transition:
    record = copy.deepcopy(row.info[features.COMMON_INTERVAL_KEY])
    altered = replace(row, info={features.COMMON_INTERVAL_KEY: record})
    if name in ("schema", "contract_sha256", "backend", "raw_transition_sha256"):
        record[name] = "invalid"
    elif name == "clock":
        record["clock"]["seconds_per_tick"] = 1.0
    elif name in ("elapsed_seconds", "after_timestamp"):
        record[name] = 0
    elif name == "action_type":
        record["action"] = True
    elif name == "action_value":
        record["action"] = 78
    elif name == "flag_type":
        record["terminated"] = 0
    elif name == "flags_contradict":
        record["terminated"] = record["truncated"] = True
    elif name in ("observation_sha256", "next_observation_sha256"):
        record[name] = "0" * 64
    elif name == "measurement":
        record["reward_measurement"]["deviation_after_units"] = -1
    elif name == "finish_without_terminal":
        record["reward_measurement"]["finished"] = True
    elif name == "outer_action":
        altered = replace(altered, action=40)
    elif name == "outer_flag":
        altered = replace(altered, truncated=True)
    elif name == "tensor_dtype":
        altered = replace(altered, observation=row.observation.double())
    elif name == "tensor_nonfinite":
        altered = replace(altered, observation=torch.full_like(row.observation, float("nan")))
    elif name == "missing_next":
        altered = replace(altered, next_observation=None)
    elif name == "reward":
        altered = replace(altered, reward=row.reward + 1)
    else:
        raise AssertionError(name)
    return altered


@pytest.mark.parametrize(
    "name",
    [
        "schema",
        "contract_sha256",
        "backend",
        "raw_transition_sha256",
        "clock",
        "elapsed_seconds",
        "after_timestamp",
        "action_type",
        "action_value",
        "flag_type",
        "flags_contradict",
        "observation_sha256",
        "next_observation_sha256",
        "measurement",
        "finish_without_terminal",
        "outer_action",
        "outer_flag",
        "tensor_dtype",
        "tensor_nonfinite",
        "missing_next",
        "reward",
    ],
)
def test_each_row_rejection_remains_independent_inside_shared_batch(name: str) -> None:
    selected = contract()
    row = make_transition(source(), selected)
    invalid = _corrupt(row, name)
    with pytest.raises(ValueError, match=r".+") as independent:
        features.validate_transition(invalid, selected)
    with pytest.raises(ValueError, match=r".+") as batched:
        replay.collate_rows([row, invalid, row], [0, 1, 2], selected)
    assert str(batched.value) == str(independent.value)


def test_terminal_next_remains_absent_in_every_row() -> None:
    selected = contract()
    terminal = make_transition(source(terminated=True), selected)
    invalid = replace(terminal, next_observation=torch.zeros(selected.feature_dim))
    with pytest.raises(ValueError, match="terminal NEXT must be absent"):
        replay.collate_rows([terminal, invalid], [0, 1], selected)
