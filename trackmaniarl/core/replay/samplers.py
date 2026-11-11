"""Uniform, sequential, on-policy, and demonstration replay sampling."""

from __future__ import annotations

import random
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any, cast

from trackmaniarl.core.contracts import ReplayStore
from trackmaniarl.core.data import BatchRequest, PriorityUpdate, TrainingBatch
from trackmaniarl.core.replay.batches import (
    _BatchBuild,
    _collate_materialized_batch,
    _eligible_n_step_ids,
    _make_batch,
    _materialize_batch,
)
from trackmaniarl.core.replay.demo_sampler import DemoMixSampler as DemoMixSampler
from trackmaniarl.core.replay.sequence_batches import (
    _reshape_training_batch,
    _SequenceBatchShape,
)
from trackmaniarl.core.replay.sequence_sampler import SequenceSampler as SequenceSampler
from trackmaniarl.core.replay.sequence_validation import _is_contiguous_rollout
from trackmaniarl.core.replay.store import InMemoryReplayStore


class UniformSampler:
    """Reference sampler suitable for custom project templates and smoke tests."""

    supports_sequence_sampling = False

    def __init__(self, pipeline: Any, seed: int = 0) -> None:
        self.pipeline = pipeline
        self._rng = random.Random(seed)

    def sample(self, store: ReplayStore, request: BatchRequest) -> TrainingBatch:
        if request.sequence_length != 1:
            raise ValueError("UniformSampler supports sequence_length=1; use a sequence sampler")
        # Only the built-in store guarantees a reentrant transaction around its
        # selection and materialization APIs. Keep user collation outside it.
        transaction = (
            store.sampling_transaction() if type(store) is InMemoryReplayStore else nullcontext()
        )
        with transaction:
            transition_ids = self._select_ids(store, request)
            build = _BatchBuild(store, self.pipeline, transition_ids, request)
            materialized = _materialize_batch(build)
        return _collate_materialized_batch(build, materialized)

    def _select_ids(self, store: ReplayStore, request: BatchRequest) -> list[int]:
        fast_sample = getattr(store, "sample_eligible_ids", None)
        if callable(fast_sample):
            return cast(list[int], fast_sample(request.n_step, request.batch_size, self._rng))
        transition_ids = _eligible_n_step_ids(store, request)
        if len(transition_ids) < request.batch_size:
            raise RuntimeError(
                f"Need {request.batch_size} complete n-step transitions, replay has "
                f"{len(transition_ids)}"
            )
        return self._rng.sample(transition_ids, request.batch_size)

    def update_priorities(self, update: PriorityUpdate) -> None:
        del update  # Uniform sampling intentionally ignores priority feedback.

    def state_dict(self) -> dict[str, Any]:
        return {"rng": self._rng.getstate()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._rng.setstate(state["rng"])


class OnPolicySequenceSampler:
    """Collate the latest fixed-length on-policy rollout."""

    on_policy_rollouts = True
    supports_sequence_sampling = True

    def __init__(self, pipeline: Any, seed: int = 0) -> None:
        self.pipeline = pipeline
        self.seed = seed

    def sample(self, store: ReplayStore, request: BatchRequest) -> TrainingBatch:
        transition_ids = _latest_rollout_ids(store, request)
        batch = _make_batch(
            _BatchBuild(
                store,
                self.pipeline,
                transition_ids,
                request,
                metadata={"sampling": "on_policy", "sequence_length": request.sequence_length},
            )
        )
        shape = _SequenceBatchShape(batch, 1, request.sequence_length, batch.next_observations)
        return _reshape_training_batch(shape)

    def update_priorities(self, update: PriorityUpdate) -> None:
        del update

    def state_dict(self) -> dict[str, int]:
        return {"seed": self.seed}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.seed = int(state["seed"])


def _latest_rollout_ids(store: ReplayStore, request: BatchRequest) -> list[int]:
    if request.batch_size != 1 or request.n_step != 1:
        raise ValueError("OnPolicySequenceSampler requires batch_size=1 and n_step=1")
    ordered = store.available_ids()
    if not ordered:
        raise RuntimeError("On-policy replay is empty")
    if len(ordered) < request.sequence_length:
        raise RuntimeError(
            f"On-policy replay has {len(ordered)} transitions, need {request.sequence_length}"
        )
    transition_ids = ordered[-request.sequence_length :]
    if not _is_contiguous_rollout(transition_ids, store.get(transition_ids)):
        raise RuntimeError("Latest on-policy rollout is not contiguous")
    return transition_ids
