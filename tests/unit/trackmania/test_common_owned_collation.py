"""Owned bulk validation preserves real row evidence and every learner byte."""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
from typing import Any

import pytest
import torch

from tests.unit.trackmania.test_common_batch_validation import _legacy_collate
from tests.unit.trackmania.test_common_training import contract, source
from tests.unit.trackmania.test_common_training_performance import _assert_exact
from trackmaniarl.core.data import BatchRequest, TrainingBatch
from trackmaniarl.trackmania.common_training import (
    COMMON6_PROFILE,
    CommonReplayStore,
    CommonSampler,
    make_learner,
    make_transition,
    replay,
)
from trackmaniarl.trackmania.common_training.features import (
    COMMON_INTERVAL_KEY,
    tensor_digest,
    validate_transition,
)


@pytest.mark.parametrize("profile", [None, COMMON6_PROFILE])
def test_sampler_updates_preserve_legacy_batch_optimizer_and_every_rng(
    monkeypatch: pytest.MonkeyPatch, profile: str | None
) -> None:
    selected = contract(**({"observation_profile": profile} if profile else {}))
    store = CommonReplayStore(selected, 67)
    for index in range(80):
        store.append(
            make_transition(
                replace(
                    source(
                        terminated=index % 3 == 0,
                        truncated=index % 3 == 1,
                        elapsed_ms=10 * (index % 5 + 1),
                    ),
                    action=index % 78,
                ),
                selected,
            )
        )
    left, right = CommonSampler(selected, seed=983), CommonSampler(selected, seed=983)
    old_learner, new_learner = make_learner(selected), make_learner(selected)
    optimized = replay.collate_rows
    for _ in range(12):
        monkeypatch.setattr(replay, "collate_rows", _legacy_collate)
        old_batch = left.sample(store, BatchRequest(32, gamma=selected.gamma))
        monkeypatch.setattr(replay, "collate_rows", optimized)
        new_batch = right.sample(store, BatchRequest(32, gamma=selected.gamma))
        for name in TrainingBatch.__dataclass_fields__:
            _assert_exact(getattr(old_batch, name), getattr(new_batch, name))
        _assert_exact(left.state_dict(), right.state_dict())
        before_rng = torch.get_rng_state().clone()
        old_metrics, old_priorities = old_learner.update(old_batch)
        after_rng = torch.get_rng_state().clone()
        old_state = old_learner.state_dict()
        torch.set_rng_state(before_rng)
        new_metrics, new_priorities = new_learner.update(new_batch)
        assert torch.equal(after_rng, torch.get_rng_state())
        _assert_exact(old_state, new_learner.state_dict())
        assert old_priorities == new_priorities
        assert {k: v for k, v in old_metrics.items() if k != "timing/update_s"} == {
            k: v for k, v in new_metrics.items() if k != "timing/update_s"
        }


def test_owned_copy_is_validated_even_if_original_alias_changes_during_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = contract()
    row = make_transition(source(), selected)
    original_observation = row.observation.clone()
    original_stack = torch.stack
    calls = 0

    def aliased(values: list[torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal calls
        copied = original_stack(values, *args, **kwargs)
        calls += 1
        if calls == 1:
            row.observation.fill_(987)
        return copied

    monkeypatch.setattr(torch, "stack", aliased)
    batch = replay.collate_rows([row], [0], selected)
    assert torch.equal(batch.observations[0], original_observation)
    assert not torch.equal(batch.observations[0], row.observation)
    with pytest.raises(ValueError, match="features differ"):
        validate_transition(row, selected)


@pytest.mark.parametrize("mutation", ["nonfinite", "hash"])
def test_owned_tensor_corruption_is_detected_after_copy(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    selected = contract()
    row = make_transition(source(), selected)
    original_stack = torch.stack

    def changed(values: list[torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:
        copied = original_stack(values, *args, **kwargs)
        copied[0, 0] = float("nan") if mutation == "nonfinite" else 123
        return copied

    monkeypatch.setattr(torch, "stack", changed)
    with pytest.raises(ValueError, match=r"actual finite|features differ"):
        replay.collate_rows([row], [0], selected)


def test_metadata_mutation_during_copy_is_rechecked_before_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = contract()
    row = make_transition(source(), selected)
    original_stack = torch.stack

    def changed(values: list[torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:
        copied = original_stack(values, *args, **kwargs)
        row.info[COMMON_INTERVAL_KEY]["action"] = 7
        return copied

    monkeypatch.setattr(torch, "stack", changed)
    with pytest.raises(ValueError, match="action/flags differ"):
        replay.collate_rows([row], [0], selected)


def test_reward_measurements_are_owned_after_validation() -> None:
    selected = contract()
    row = make_transition(source(), selected)
    batch = replay.collate_rows([row], [0], selected)
    expected = copy.deepcopy(batch.metadata["reward_measurements"])
    row.info[COMMON_INTERVAL_KEY]["reward_measurement"]["finished"] = True
    assert batch.metadata["reward_measurements"] == expected


def test_noncontiguous_and_gradient_tracking_inputs_keep_exact_values() -> None:
    selected = contract()
    row = make_transition(source(), selected)
    current = row.observation.repeat_interleave(2)[::2].requires_grad_()
    following = row.next_observation.repeat_interleave(2)[::2].requires_grad_()
    assert not current.is_contiguous()
    altered = replace(row, observation=current, next_observation=following)
    old = _legacy_collate([altered], [0], selected)
    new = replay.collate_rows([altered], [0], selected)
    for name in TrainingBatch.__dataclass_fields__:
        _assert_exact(getattr(old, name), getattr(new, name))
    assert new.observations.requires_grad
    assert new.next_observations.requires_grad


def test_forged_finite_hash_does_not_authorize_nonfinite_next() -> None:
    selected = contract()
    row = make_transition(source(), selected)
    following = row.next_observation.clone()
    following[0] = float("inf")
    record = copy.deepcopy(row.info[COMMON_INTERVAL_KEY])
    record["next_observation_sha256"] = tensor_digest(following)
    altered = replace(row, next_observation=following, info={COMMON_INTERVAL_KEY: record})
    with pytest.raises(ValueError, match="actual finite"):
        replay.collate_rows([row, altered], [0, 1], selected)


def test_valid_shared_reward_that_overflows_float32_is_rejected() -> None:
    selected = contract()
    row = make_transition(source(), selected)
    huge = replace(selected, reward=replace(selected.reward, progress_per_unit=1e40))
    record = copy.deepcopy(row.info[COMMON_INTERVAL_KEY])
    record["contract_sha256"] = huge.digest
    record["clock"] = asdict(huge.clock)
    reward = huge.reward.score(source().reward_measurement, 0.05)
    altered = replace(row, reward=reward, info={COMMON_INTERVAL_KEY: record})
    for operation in (
        validate_transition,
        lambda value, spec: replay.collate_rows([value], [0], spec),
    ):
        with pytest.raises(ValueError, match="reward differs"):
            operation(altered, huge)


def test_finite_reductions_are_per_owned_stack_not_per_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = contract()
    row = make_transition(source(), selected)
    original = torch.isfinite
    calls: list[tuple[int, ...]] = []

    def counted(value: torch.Tensor) -> torch.Tensor:
        calls.append(tuple(value.shape))
        return original(value)

    monkeypatch.setattr(torch, "isfinite", counted)
    replay.collate_rows([row] * 32, list(range(32)), selected)
    assert calls == [(32, selected.feature_dim), (32, selected.feature_dim), (32,)]
