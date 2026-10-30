"""Exact equivalence regressions for allocation-only common training changes."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch

from tests.unit.trackmania.test_common_training import contract, source
from trackmaniarl.core.data import BatchRequest, TrainingBatch
from trackmaniarl.trackmania.common_training import (
    COMMON6_PROFILE,
    CommonReplayStore,
    CommonSampler,
    learning,
    make_learner,
    make_transition,
)
from trackmaniarl.trackmania.common_training.replay import collate_rows


def _assert_exact(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert left.detach().cpu().contiguous().numpy().tobytes() == (
            right.detach().cpu().contiguous().numpy().tobytes()
        )
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert left.tobytes() == right.tobytes()
    elif isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        assert left.keys() == right.keys()
        for name in left:
            _assert_exact(left[name], right[name])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_exact(a, b)
    else:
        assert type(left) is type(right)
        assert left == right


@pytest.mark.parametrize("capacity", [3, 67, 311])
def test_range_sampling_preserves_legacy_ids_rng_and_batches_through_eviction(
    capacity: int,
) -> None:
    selected = contract(observation_profile=COMMON6_PROFILE)
    store = CommonReplayStore(selected, capacity)
    sampler, legacy = CommonSampler(selected, seed=879), random.Random(879)
    row = make_transition(source(), selected)
    for index in range(capacity + 11):
        store.append(row)
        if index + 1 not in (capacity, capacity + 11):
            continue
        assert isinstance(store.available_id_range(), range)
        assert list(store.available_id_range()) == store.available_ids()
        for size in (1, min(3, capacity), min(64, capacity)):
            for _ in range(3):
                expected_ids = legacy.sample(store.available_ids(), size)
                expected = collate_rows(store.get(expected_ids), expected_ids, selected)
                actual = sampler.sample(store, BatchRequest(size, gamma=selected.gamma))
                assert actual.transition_ids == expected_ids
                assert sampler.state_dict()["rng"] == legacy.getstate()
                for name in TrainingBatch.__dataclass_fields__:
                    _assert_exact(getattr(actual, name), getattr(expected, name))
        restored = CommonSampler(selected)
        restored.load_state_dict(sampler.state_dict())
        request = BatchRequest(min(3, capacity), gamma=selected.gamma)
        assert (
            restored.sample(store, request).transition_ids
            == sampler.sample(store, request).transition_ids
        )
        # The restored comparison above consumes the same single sample as the
        # current sampler, so advance the independent legacy RNG identically.
        legacy.sample(store.available_ids(), request.batch_size)
        assert sampler.state_dict()["rng"] == legacy.getstate()


def _legacy_targets(
    batch: TrainingBatch,
    selected: Any,
    evaluate_actual_next: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Literal pre-optimization public target implementation for comparison."""
    indices = learning.validate_batch(batch, selected)
    rewards, discounts = batch.rewards, batch.bootstrap_discounts
    targets = rewards[:, None].expand(-1, 32).clone()
    if indices.numel():
        values = evaluate_actual_next(batch.next_observations)
        if tuple(values.shape) != (len(indices), 32) or not bool(torch.isfinite(values).all()):
            raise ValueError("actual finite NEXT fixed32 quantiles required")
        targets[indices] = rewards[indices, None] + discounts[indices, None] * values
    return targets


@pytest.mark.parametrize("profile", [None, COMMON6_PROFILE])
def test_updates_preserve_legacy_gradients_weights_optimizer_and_rng_exactly(
    monkeypatch: pytest.MonkeyPatch,
    profile: str | None,
) -> None:
    selected = contract(**({"observation_profile": profile} if profile else {}))
    original = learning._validated_targets
    baseline, optimized = make_learner(selected), make_learner(selected)
    for flags in ((False, False, True), (True, True, True), (False, True, False)):
        rows = [
            make_transition(
                source(
                    terminated=terminal,
                    truncated=not terminal and index == 0,
                    elapsed_ms=(index + 1) * 10,
                ),
                selected,
            )
            for index, terminal in enumerate(flags)
        ]
        batch = collate_rows(rows, list(range(len(rows))), selected)

        def old(
            value: TrainingBatch,
            unused_indices: torch.Tensor,
            evaluate: Callable[[torch.Tensor], torch.Tensor],
        ) -> torch.Tensor:
            del unused_indices
            return _legacy_targets(value, selected, evaluate)

        monkeypatch.setattr(learning, "_validated_targets", old)
        before_rng = torch.get_rng_state().clone()
        old_metrics, old_priorities = baseline.update(batch)
        old_state = baseline.state_dict()
        old_rng = torch.get_rng_state().clone()
        torch.set_rng_state(before_rng)
        monkeypatch.setattr(learning, "_validated_targets", original)
        new_metrics, new_priorities = optimized.update(batch)
        assert torch.equal(old_rng, torch.get_rng_state())
        _assert_exact(old_state, optimized.state_dict())
        for a, b in zip(baseline.model.parameters(), optimized.model.parameters(), strict=True):
            _assert_exact(a.grad, b.grad)
        assert old_priorities == new_priorities
        assert {k: v for k, v in old_metrics.items() if k != "timing/update_s"} == {
            k: v for k, v in new_metrics.items() if k != "timing/update_s"
        }


def test_update_validates_once_and_public_targets_keep_independent_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = contract()
    row = make_transition(source(terminated=True), selected)
    batch = collate_rows([row], [0], selected)
    learner = make_learner(selected)
    original, calls = learning.validate_batch, []

    def counted(value: TrainingBatch, spec: Any) -> torch.Tensor:
        calls.append(value)
        return original(value, spec)

    monkeypatch.setattr(learning, "validate_batch", counted)
    learner.update(batch)
    assert len(calls) == 1
    learning.lazy_targets(batch, selected, lambda _: pytest.fail("terminal NEXT encoded"))
    assert len(calls) == 2
    invalid = replace(batch, bootstrap_discounts=torch.ones_like(batch.bootstrap_discounts))
    before = learner.update_count
    with pytest.raises(ValueError, match="bootstrap discount mismatch"):
        learner.update(invalid)
    assert learner.update_count == before
    with pytest.raises(ValueError, match="bootstrap discount mismatch"):
        learning.lazy_targets(invalid, selected, lambda _: pytest.fail("invalid NEXT encoded"))
