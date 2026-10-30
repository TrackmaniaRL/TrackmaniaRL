"""A discarded lookahead must not consume RNG or retain aliases into replay."""

import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from tests.unit._composite_value_fixtures import _scalar_model
from trackmaniarl.core.builtins import IdentityFeaturePipeline
from trackmaniarl.core.data import BatchRequest, TrainingBatch, Transition
from trackmaniarl.core.replay import InMemoryReplayStore, UniformSampler
from trackmaniarl.distributed.coordinator_support import _BatchPrefetcher
from trackmaniarl.experiments.full_qr.training_performance import (
    CompiledEncoderQrLearner,
    PrefetchedUniformSampler,
)


def _store() -> InMemoryReplayStore:
    store = InMemoryReplayStore(capacity=16)
    for step in range(16):
        store.append(
            Transition(
                observation=torch.tensor([float(step)]),
                action=step % 3,
                reward=float(step),
                next_observation=torch.tensor([float(step + 1)]),
                terminated=step == 15,
                truncated=False,
                episode_id="real-boundaries",
                step=step,
            )
        )
    return store


def test_lookahead_checkpoint_and_restore_preserve_uniform_draws() -> None:
    store = _store()
    baseline = UniformSampler(IdentityFeaturePipeline(), seed=17)
    sampler = PrefetchedUniformSampler(IdentityFeaturePipeline(), seed=17)
    restored = PrefetchedUniformSampler(IdentityFeaturePipeline(), seed=99)
    request = BatchRequest(8, n_step=3, gamma=0.995)
    try:
        for _ in range(4):
            actual, expected = sampler.sample(store, request), baseline.sample(store, request)
            assert actual.transition_ids == expected.transition_ids
            torch.testing.assert_close(actual.rewards, expected.rewards, rtol=0, atol=0)
            torch.testing.assert_close(actual.bootstrap_discounts, expected.bootstrap_discounts)
        state = sampler.state_dict()
        assert state == baseline.state_dict()
        restored.load_state_dict(state)
        for _ in range(3):
            assert (
                restored.sample(store, request).transition_ids
                == baseline.sample(store, request).transition_ids
            )
    finally:
        sampler.close_prefetch()
        restored.close_prefetch()


def test_changed_request_discards_pending_draw_without_rng_loss() -> None:
    store = _store()
    sampler = PrefetchedUniformSampler(IdentityFeaturePipeline(), seed=4)
    baseline = UniformSampler(IdentityFeaturePipeline(), seed=4)
    try:
        for request in (BatchRequest(8), BatchRequest(3, n_step=2), BatchRequest(5, gamma=0.9)):
            assert (
                sampler.sample(store, request).transition_ids
                == baseline.sample(store, request).transition_ids
            )
    finally:
        sampler.close_prefetch()
    assert sampler.state_dict() == baseline.state_dict()


def test_prefetched_rows_survive_ring_overwrite_and_coordinator_closes_worker() -> None:
    store = _store()
    sampler = PrefetchedUniformSampler(IdentityFeaturePipeline(), seed=2)
    source = _BatchPrefetcher(SimpleNamespace(sampler=sampler))
    try:
        first = sampler.sample(store, BatchRequest(4))
        assert sampler._pending is not None
        ready = sampler._pending.result().batch
        expected = ready.observations.clone()
        for step in range(16):
            store.append(
                Transition(
                    observation=torch.tensor([-100.0]),
                    action=0,
                    reward=0.0,
                    next_observation=torch.tensor([-100.0]),
                    terminated=step == 15,
                    truncated=False,
                    episode_id="overwritten",
                    step=step,
                )
            )
        second = sampler.sample(store, BatchRequest(4))
        torch.testing.assert_close(second.observations, expected, rtol=0, atol=0)
        assert first.observations.min() >= 0
    finally:
        source.close()
    with pytest.raises(RuntimeError, match="closed"):
        sampler.sample(store, BatchRequest(4))


def test_background_pipeline_failure_is_not_silently_retried() -> None:
    class FailingPipeline(IdentityFeaturePipeline):
        calls = 0

        def collate(self, transitions: Sequence[Transition]) -> Any:
            self.calls += 1
            if self.calls > 1:
                raise ValueError("actual malformed observation")
            return super().collate(transitions)

    sampler = PrefetchedUniformSampler(FailingPipeline(), seed=17)
    store = _store()
    baseline = UniformSampler(IdentityFeaturePipeline(), seed=17)
    sampler.sample(store, BatchRequest(4))
    baseline.sample(store, BatchRequest(4))
    with pytest.raises(ValueError, match="malformed"):
        sampler.sample(store, BatchRequest(4))
    with pytest.raises(ValueError, match="malformed"):
        sampler.close_prefetch()
    assert sampler.state_dict() == baseline.state_dict()


def test_cpu_actor_keeps_eager_policy_and_checkpoint_parameter_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_call = Mock(side_effect=AssertionError("CPU actors must not compile"))
    monkeypatch.setattr(torch.nn.Module, "compile", compile_call)
    learner = CompiledEncoderQrLearner(
        model=_scalar_model(), execution={"device": "cpu", "precision": "float32"}
    )
    learner.setup({"seed": 17})
    policy = learner.policy()
    compile_call.assert_not_called()
    assert policy.model.encoder._compiled_call_impl is None
    assert policy.model.architecture_fingerprint() == learner.model.architecture_fingerprint()
    assert tuple(policy.model.state_dict()) == tuple(learner.model.state_dict())
    for name, value in learner.model.state_dict().items():
        torch.testing.assert_close(policy.model.state_dict()[name], value, rtol=0, atol=0)


def test_transfer_prefetch_is_bound_by_coordinator_and_runs_outside_replay_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store()
    original = store.sampling_transaction
    local = threading.local()

    @contextmanager
    def sampling_transaction() -> Iterator[None]:
        with original():
            local.in_transaction = True
            try:
                yield
            finally:
                local.in_transaction = False

    monkeypatch.setattr(store, "sampling_transaction", sampling_transaction)
    calls: list[str] = []

    def prepare(batch: TrainingBatch) -> TrainingBatch:
        assert not getattr(local, "in_transaction", False)
        calls.append(threading.current_thread().name)
        return replace(batch, metadata={**batch.metadata, "prepared_probe": True})

    sampler = PrefetchedUniformSampler(IdentityFeaturePipeline(), seed=17, transfer_prefetch=True)
    baseline = UniformSampler(IdentityFeaturePipeline(), seed=17)
    source = _BatchPrefetcher(
        SimpleNamespace(
            sampler=sampler, replay_store=store, learner=SimpleNamespace(prepare_batch=prepare)
        )
    )
    request = BatchRequest(4)
    try:
        for _ in range(3):
            batch, _, _ = source.next(request)
            expected = baseline.sample(store, request)
            assert batch.metadata["prepared_probe"] is True
            assert batch.transition_ids == expected.transition_ids
            torch.testing.assert_close(batch.observations, expected.observations, rtol=0, atol=0)
        assert any(name.startswith("qr-batch") for name in calls)
    finally:
        source.close()
    assert sampler.state_dict() == baseline.state_dict()


def test_opted_in_transfer_requires_binding_before_rng_is_consumed() -> None:
    sampler = PrefetchedUniformSampler(IdentityFeaturePipeline(), seed=17, transfer_prefetch=True)
    baseline = UniformSampler(IdentityFeaturePipeline(), seed=17)
    with pytest.raises(RuntimeError, match="bound"):
        sampler.sample(_store(), BatchRequest(4))
    assert sampler.state_dict() == baseline.state_dict()
    sampler.close_prefetch()


def test_background_transfer_failure_is_fatal_and_checkpoint_rewinds_unused_draw() -> None:
    calls = 0

    def prepare(batch: TrainingBatch) -> TrainingBatch:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise ValueError("transfer preparation failed")
        return batch

    sampler = PrefetchedUniformSampler(IdentityFeaturePipeline(), seed=17, transfer_prefetch=True)
    store, request = _store(), BatchRequest(4)
    source = _BatchPrefetcher(
        SimpleNamespace(
            sampler=sampler, replay_store=store, learner=SimpleNamespace(prepare_batch=prepare)
        )
    )
    baseline = UniformSampler(IdentityFeaturePipeline(), seed=17)
    source.next(request)
    baseline.sample(store, request)
    with pytest.raises(ValueError, match="transfer preparation failed"):
        source.next(request)
    with pytest.raises(ValueError, match="transfer preparation failed"):
        source.close()
    assert sampler.state_dict() == baseline.state_dict()
