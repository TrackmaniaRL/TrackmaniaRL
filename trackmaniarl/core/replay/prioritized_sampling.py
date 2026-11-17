"""Batch assembly for incremental prioritized replay selections."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import torch

from trackmaniarl.core.data import BatchRequest, TrainingBatch, TransitionId
from trackmaniarl.core.replay.batches import (
    _batch_data,
    _BatchAssembly,
    _BatchBuild,
    _history_padding_masks,
    _materialize_batch,
    _reshape_batch_sequences,
    _reshape_sequence_batch,
    _SequenceBatchLayout,
    _standard_collation,
    _training_batch,
)
from trackmaniarl.core.replay.ownership import owned_transition
from trackmaniarl.core.replay.sequence_sampler import (
    _collate_target_snapshot,
    _observation_collator,
    _snapshot_target_histories,
    _TargetSnapshot,
)
from trackmaniarl.core.replay.store import _IncrementalReplayStore

if TYPE_CHECKING:
    from trackmaniarl.core.replay.prioritized import PrioritizedSampler


@dataclass(frozen=True, slots=True)
class _IncrementalSample:
    sampler: PrioritizedSampler
    store: _IncrementalReplayStore
    request: BatchRequest
    transition_ids: list[TransitionId]
    weights: tuple[float, ...]
    beta: float
    sampling: dict[str, float]
    demo_flags: tuple[bool, ...]
    expert_flags: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class _SequenceMaterialization:
    histories: list[list[TransitionId]]
    flattened: list[TransitionId]
    targets: _TargetSnapshot
    next_masks: torch.Tensor


def _sample_incremental_snapshot(
    sampler: PrioritizedSampler,
    store: _IncrementalReplayStore,
    request: BatchRequest,
) -> TrainingBatch:
    hook = _observation_collator(sampler.pipeline) if request.sequence_length > 1 else None
    with store.sampling_transaction():
        transition_ids, weights, beta, sampling = sampler._sample_incremental_ids(store, request)
        sample = _IncrementalSample(
            sampler,
            store,
            request,
            transition_ids,
            weights,
            beta,
            sampling,
            tuple(store.demo_flags(transition_ids)),
            _expert_flags(sampler, transition_ids),
        )
        sequence = _sequence_materialization(sample, hook) if request.sequence_length > 1 else None
        build = _batch_build(sample, sequence)
        materialized = _materialize_batch(build)
        materialized = replace(
            materialized,
            transitions=[owned_transition(item) for item in materialized.transitions],
        )
    # Keep the existing target-before-transition collation order while running
    # both user hooks outside the producer's replay lock.
    targets = (
        _collate_target_snapshot(sample.sampler.pipeline, sequence.targets)
        if sequence is not None
        else None
    )
    collated = build.pipeline.collate(materialized.transitions)
    standard = _standard_collation(collated)
    batch = _training_batch(
        _BatchAssembly(build, materialized, standard, _batch_data(collated, standard))
    )
    return (
        _reshape_sequence_output(sample, sequence, batch, targets)
        if sequence is not None
        else batch
    )


def _expert_flags(
    sampler: PrioritizedSampler, transition_ids: list[TransitionId]
) -> tuple[bool, ...]:
    if sampler._tree is None:
        raise RuntimeError("prioritized replay index is not initialized")
    return tuple(bool(sampler._expert_slots[item % sampler._tree.size]) for item in transition_ids)


def _sequence_materialization(sample: _IncrementalSample, hook: Any) -> _SequenceMaterialization:
    length = sample.request.sequence_length
    histories = [sample.store.history_ids(item, length) for item in sample.transition_ids]
    if any(len(history) != length for history in histories):
        raise RuntimeError("prioritized sequence index is out of sync with replay")
    horizons = [
        sample.store.n_step_ids(item, sample.request.n_step) for item in sample.transition_ids
    ]
    if any(not horizon for horizon in horizons):
        raise RuntimeError("prioritized n-step index is out of sync with replay")
    history_values = [sample.store.get(history) for history in histories]
    horizon_values = [sample.store.get(horizon) for horizon in horizons]
    targets = _snapshot_target_histories(hook, history_values, horizon_values, length)
    flattened = [item for history in histories for item in history]
    masks = _history_padding_masks(histories)
    next_masks = torch.stack(
        [
            torch.cat((row, torch.ones(len(horizon), dtype=torch.bool)))[-length:]
            for row, horizon in zip(masks, horizons, strict=True)
        ]
    )
    return _SequenceMaterialization(histories, flattened, targets, next_masks)


def _reshape_sequence_output(  # noqa: PLR0913
    sample: _IncrementalSample,
    materialized: _SequenceMaterialization,
    batch: TrainingBatch,
    targets: Any,
) -> TrainingBatch:
    request = sample.request
    layout = _SequenceBatchLayout(
        request.batch_size,
        request.sequence_length,
        _history_padding_masks(materialized.histories),
    )
    reshaped = _reshape_batch_sequences(batch, layout)
    next_observations = _reshape_sequence_batch(
        targets,
        request.batch_size,
        request.sequence_length,
    )
    return replace(
        reshaped,
        next_observations=next_observations,
        metadata={**reshaped.metadata, "next_observation_masks": materialized.next_masks},
    )


def _batch_build(
    sample: _IncrementalSample, sequence: _SequenceMaterialization | None
) -> _BatchBuild:
    return _BatchBuild(
        sample.store,
        sample.sampler.pipeline,
        sample.transition_ids if sequence is None else sequence.flattened,
        sample.request,
        importance_weights=sample.weights,
        metadata=_sample_metadata(
            sample, "prioritized" if sequence is None else "prioritized_sequence"
        ),
        bootstrap_stride=1 if sequence is None else sample.request.sequence_length,
    )


def _sample_metadata(
    sample: _IncrementalSample, kind: Literal["prioritized", "prioritized_sequence"]
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "sampling": kind,
        "beta": sample.beta,
        "demo_flags": sample.demo_flags,
        **sample.sampling,
    }
    if kind == "prioritized_sequence":
        metadata.update(_sequence_metadata(sample.request, sample.transition_ids))
    if sample.sampler.expert_demo_time_s is not None:
        metadata["expert_demo_flags"] = sample.expert_flags
    return metadata


def _sequence_metadata(
    request: BatchRequest, transition_ids: list[TransitionId]
) -> dict[str, object]:
    return {
        "sequence_length": request.sequence_length,
        "n_step": request.n_step,
        "gamma": request.gamma,
        "priority_transition_ids": tuple(transition_ids),
    }
