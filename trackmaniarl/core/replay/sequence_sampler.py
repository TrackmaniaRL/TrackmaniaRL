"""Contiguous recurrent replay sampling."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any

from trackmaniarl.core.contracts import ReplayStore
from trackmaniarl.core.data import BatchRequest, PriorityUpdate, TrainingBatch, Transition
from trackmaniarl.core.pytree import tree_collate, tree_snapshot
from trackmaniarl.core.replay.batches import (
    _batch_data,
    _BatchAssembly,
    _BatchBuild,
    _has_complete_n_step,
    _materialize_batch,
    _n_step_horizon,
    _sequence_target_observation_histories,
    _standard_collation,
    _training_batch,
)
from trackmaniarl.core.replay.sequence_batches import (
    _reshape_training_batch,
    _SequenceBatchShape,
)
from trackmaniarl.core.replay.sequence_index import SequenceIndex
from trackmaniarl.core.replay.store import InMemoryReplayStore


@dataclass(frozen=True, slots=True)
class _WindowScan:
    ordered: list[int]
    transitions: list[Transition]
    available: Mapping[int, Transition]
    request: BatchRequest
    length: int


@dataclass(frozen=True, slots=True)
class _WindowCandidates:
    store: ReplayStore
    ordered: list[int]
    request: BatchRequest
    length: int


@dataclass(frozen=True, slots=True)
class _SequenceEdge:
    previous_id: int
    transition_id: int
    previous: Transition
    transition: Transition


@dataclass(frozen=True, slots=True)
class _LinkedHistory:
    history_ids: Callable[[int, int], list[int]]
    is_n_step_eligible: Callable[[int, int], bool]


@dataclass(frozen=True, slots=True)
class _SelectedSequences:
    store: ReplayStore
    windows: list[list[int]]
    request: BatchRequest


@dataclass(frozen=True, slots=True)
class _ReplayIndex:
    ordered: list[int] | None
    revision: int | None
    identifier: tuple[int, ...] | None
    cached_starts: list[int] | None


@dataclass(frozen=True, slots=True)
class _CacheUpdate:
    store: ReplayStore
    request_key: tuple[int, int]
    index: _ReplayIndex
    starts: list[int]


@dataclass(frozen=True, slots=True)
class _WindowPosition:
    scan: _WindowScan
    index: int
    transition: Transition
    current: int


class SequenceSampler:
    """Samples only contiguous transitions from one identified episode."""

    supports_sequence_sampling = True
    _cached_store: ReplayStore | None
    _cached_request: tuple[int, int] | None
    _cached_revision: int | None
    _cached_ids: tuple[int, ...] | None
    _cached_starts: list[int]

    def __init__(self, pipeline: Any, sequence_length: int, seed: int = 0) -> None:
        if sequence_length < 2:
            raise ValueError("SequenceSampler requires sequence_length >= 2")
        self.pipeline = pipeline
        self.sequence_length = sequence_length
        self._rng = random.Random(seed)
        self._reset_cache()

    def sample(self, store: ReplayStore, request: BatchRequest) -> TrainingBatch:
        length = request.sequence_length if request.sequence_length > 1 else self.sequence_length
        # The built-in store guarantees reentrant queries during this
        # transaction. External implementations retain the prior behavior.
        owned_store = type(store) is InMemoryReplayStore
        hook = _observation_collator(self.pipeline) if owned_store else None
        transaction = store.sampling_transaction() if owned_store else nullcontext()
        with transaction:
            windows = self._sample_windows(store, request, length)
            selection = _SelectedSequences(store, windows, request)
            histories, horizons = _selected_sequence_transitions(selection)
            build = self._sequence_batch_build(selection, length)
            materialized = _materialize_batch(build)
            if owned_store:
                targets = _snapshot_target_histories(hook, histories, horizons, length)
                materialized = replace(
                    materialized,
                    transitions=[_owned_transition(item) for item in materialized.transitions],
                )
        # User collators can perform arbitrary work. Materialize the entire
        # replay snapshot first, then release the store lock before invoking them.
        next_observations = (
            _collate_target_snapshot(self.pipeline, targets)
            if owned_store
            else _collate_target_histories(self.pipeline, histories, horizons, length)
        )
        collated = self.pipeline.collate(materialized.transitions)
        standard = _standard_collation(collated)
        batch = _training_batch(
            _BatchAssembly(build, materialized, standard, _batch_data(collated, standard))
        )
        shape = _SequenceBatchShape(batch, request.batch_size, length, next_observations)
        return _reshape_training_batch(shape)

    def _sample_windows(
        self, store: ReplayStore, request: BatchRequest, length: int
    ) -> list[list[int]]:
        if type(store) is InMemoryReplayStore:
            if (
                self._incremental_index is None
                or self._incremental_index.store is not store
                or self._incremental_index.length != length
                or self._incremental_index.n_step != request.n_step
            ):
                self._incremental_index = SequenceIndex(store, length, request.n_step)
            index = self._incremental_index
            index.synchronize()
            chosen = index.draw(self._rng, request.batch_size)
            return [store.history_ids(final_id, length) for final_id in chosen]
        starts = self._window_starts(store, request, length)
        if len(starts) < request.batch_size:
            raise RuntimeError(
                f"Need {request.batch_size} valid sequences, replay has {len(starts)}"
            )
        chosen = self._rng.sample(starts, request.batch_size)
        linked = _linked_history(store)
        if linked is not None:
            return [linked.history_ids(final_id, length) for final_id in chosen]
        return [list(range(start, start + length)) for start in chosen]

    def _sequence_batch_build(self, selection: _SelectedSequences, length: int) -> _BatchBuild:
        flattened = [transition_id for window in selection.windows for transition_id in window]
        metadata = _sequence_metadata(selection, length)
        return _BatchBuild(
            selection.store,
            self.pipeline,
            flattened,
            selection.request,
            metadata=metadata,
            bootstrap_stride=length,
        )

    def _window_starts(self, store: ReplayStore, request: BatchRequest, length: int) -> list[int]:
        request_key = (length, request.n_step)
        index = self._replay_index(store, request_key)
        if index.cached_starts is not None:
            return index.cached_starts
        ordered = store.available_ids() if index.ordered is None else index.ordered
        candidates = _WindowCandidates(store, ordered, request, length)
        starts = _uncached_window_starts(candidates)
        self._cache_starts(_CacheUpdate(store, request_key, index, starts))
        return starts

    def _replay_index(self, store: ReplayStore, request_key: tuple[int, int]) -> _ReplayIndex:
        same_index = self._cached_store is store and self._cached_request == request_key
        changes_since = getattr(store, "changes_since", None)
        if callable(changes_since):
            revision, changes = changes_since(self._cached_revision if same_index else None)
            cached = self._cached_starts if same_index and changes == [] else None
            return _ReplayIndex(None, revision, None, cached)
        ordered = store.available_ids()
        identifier = tuple(ordered)
        cached = self._cached_starts if same_index and identifier == self._cached_ids else None
        return _ReplayIndex(ordered, None, identifier, cached)

    def _cache_starts(self, update: _CacheUpdate) -> None:
        self._cached_store = update.store
        self._cached_request = update.request_key
        self._cached_revision = update.index.revision
        self._cached_ids = update.index.identifier
        self._cached_starts = update.starts

    def _reset_cache(self) -> None:
        self._incremental_index: SequenceIndex | None = None
        self._cached_store = None
        self._cached_request = None
        self._cached_revision = None
        self._cached_ids = None
        self._cached_starts = []

    def update_priorities(self, update: PriorityUpdate) -> None:
        del update

    def state_dict(self) -> dict[str, Any]:
        return {"rng": self._rng.getstate()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._rng.setstate(state["rng"])
        self._reset_cache()


def _owned_transition(item: Transition) -> Transition:
    return replace(
        item,
        observation=tree_snapshot(item.observation),
        action=tree_snapshot(item.action),
        next_observation=tree_snapshot(item.next_observation),
        info=tree_snapshot(item.info),
    )


def _linked_history(store: ReplayStore) -> _LinkedHistory | None:
    history_ids = getattr(store, "history_ids", None)
    is_n_step_eligible = getattr(store, "is_n_step_eligible", None)
    if not callable(history_ids) or not callable(is_n_step_eligible):
        return None
    return _LinkedHistory(history_ids, is_n_step_eligible)


def _uncached_window_starts(candidates: _WindowCandidates) -> list[int]:
    if len(candidates.ordered) < candidates.length:
        raise RuntimeError(f"Need at least {candidates.length} transitions for sequence sampling")
    linked = _linked_history(candidates.store)
    if linked is not None:
        return _linked_window_starts(candidates, linked)
    transitions = candidates.store.get(candidates.ordered)
    available = dict(zip(candidates.ordered, transitions, strict=True))
    return _sequence_window_starts(
        _WindowScan(
            candidates.ordered,
            transitions,
            available,
            candidates.request,
            candidates.length,
        )
    )


def _linked_window_starts(candidates: _WindowCandidates, linked: _LinkedHistory) -> list[int]:
    return [
        transition_id
        for transition_id in candidates.ordered
        if linked.is_n_step_eligible(transition_id, candidates.request.n_step)
        and len(set(linked.history_ids(transition_id, candidates.length))) == candidates.length
    ]


def _sequence_window_starts(scan: _WindowScan) -> list[int]:
    eligible = [_has_complete_n_step(item, scan.available, scan.request) for item in scan.ordered]
    starts: list[int] = []
    contiguous = 1
    ineligible = 0
    for index, transition in enumerate(scan.transitions):
        contiguous = _contiguous_length(_WindowPosition(scan, index, transition, contiguous))
        ineligible += int(not eligible[index])
        if index >= scan.length:
            ineligible -= int(not eligible[index - scan.length])
        if index >= scan.length - 1 and contiguous >= scan.length and not ineligible:
            starts.append(scan.ordered[index - scan.length + 1])
    return starts


def _contiguous_length(position: _WindowPosition) -> int:
    if not position.index:
        return position.current
    scan = position.scan
    index = position.index
    edge = _SequenceEdge(
        scan.ordered[index - 1],
        scan.ordered[index],
        scan.transitions[index - 1],
        position.transition,
    )
    return position.current + 1 if _extends_episode(edge) else 1


def _extends_episode(edge: _SequenceEdge) -> bool:
    return bool(
        edge.transition.episode_id is not None
        and edge.transition.episode_id == edge.previous.episode_id
        and edge.transition_id == edge.previous_id + 1
        and (edge.previous.step is None or edge.transition.step == edge.previous.step + 1)
        and not edge.previous.terminated
        and not edge.previous.truncated
    )


def _selected_sequence_transitions(
    selection: _SelectedSequences,
) -> tuple[list[list[Transition]], list[list[Transition]]]:
    final_ids = [window[-1] for window in selection.windows]
    resolver = getattr(selection.store, "n_step_ids", None)
    horizon_ids = _selected_horizon_ids(selection, final_ids, resolver)
    requested = list(dict.fromkeys(_flatten_windows([*selection.windows, *horizon_ids])))
    available = dict(zip(requested, selection.store.get(requested), strict=True))
    histories = [[available[item] for item in window] for window in selection.windows]
    if callable(resolver):
        horizons = [[available[item] for item in horizon] for horizon in horizon_ids]
    else:
        horizons = [_n_step_horizon(item, available, selection.request) for item in final_ids]
    return histories, horizons


def _selected_horizon_ids(
    selection: _SelectedSequences, final_ids: list[int], resolver: Any
) -> list[list[int]]:
    if callable(resolver):
        return [resolver(item, selection.request.n_step) for item in final_ids]
    return [
        [
            candidate
            for candidate in range(item, item + selection.request.n_step)
            if selection.store.contains(candidate)
        ]
        for item in final_ids
    ]


def _sequence_metadata(selection: _SelectedSequences, length: int) -> dict[str, Any]:
    return {
        "sampling": "sequence",
        "sequence_length": length,
        "n_step": selection.request.n_step,
        "gamma": selection.request.gamma,
        "priority_transition_ids": tuple(window[-1] for window in selection.windows),
    }


def _flatten_windows(windows: list[list[int]]) -> list[int]:
    return [item for window in windows for item in window]


def _flatten_histories(histories: list[list[Any]]) -> list[Any]:
    return [item for history in histories for item in history]


def _observation_collator(pipeline: Any) -> Callable[..., Any] | None:
    """Avoid an inherited hook after a subclass changes transition collation."""
    hook = getattr(pipeline, "collate_observations", None)
    if not callable(hook):
        return None
    # Instance hooks and dynamically supplied methods are explicit opt-ins.
    instance_methods = getattr(pipeline, "__dict__", {})
    if "collate_observations" in instance_methods:
        return hook
    if "collate" in instance_methods:
        return None
    for cls in type(pipeline).__mro__:
        if "collate_observations" in cls.__dict__:
            return hook
        if "collate" in cls.__dict__:
            return None
    return hook


def _collate_target_histories(  # noqa: PLR0913
    pipeline: Any,
    histories: list[list[Transition]],
    horizons: list[list[Transition]],
    length: int,
) -> Any:
    """Apply the same observation collation to bootstrap histories as inputs.

    A bootstrap state can be the final next_observation of an episode, with
    no transition starting there. Use its predecessor for the remaining
    fields in that case. A collator must not assume those fields describe a
    real action taken from that final bootstrap state.
    """
    observations = _sequence_target_observation_histories(histories, horizons, length)
    collate_observations = _observation_collator(pipeline)
    if collate_observations is not None:
        return collate_observations(_flatten_histories(observations))
    targets = [
        replace(transition, observation=observation)
        for history, horizon, row in zip(histories, horizons, observations, strict=True)
        for transition, observation in zip(
            (history + horizon[1:] + horizon[-1:])[-length:], row, strict=True
        )
    ]
    standard = _standard_collation(pipeline.collate(targets))
    # Opaque collators do not supply TrainingBatch.observations either; match
    # the fallback used for inputs by _collated_fields in batches.py.
    return (
        standard["observations"]
        if standard is not None
        else tree_collate(_flatten_histories(observations))
    )


@dataclass(frozen=True, slots=True)
class _TargetSnapshot:
    observations: list[Any]
    transitions: list[Transition] | None
    collate_observations: Callable[..., Any] | None


def _snapshot_target_histories(  # noqa: PLR0913
    hook: Callable[..., Any] | None,
    histories: list[list[Transition]],
    horizons: list[list[Transition]],
    length: int,
) -> _TargetSnapshot:
    """Own only target inputs, keeping each occurrence isolated from user mutations."""
    rows = _sequence_target_observation_histories(histories, horizons, length)
    if hook is not None:
        return _TargetSnapshot(
            [tree_snapshot(item) for item in _flatten_histories(rows)], None, hook
        )
    transitions = [
        _owned_transition(replace(transition, observation=observation))
        for history, horizon, row in zip(histories, horizons, rows, strict=True)
        for transition, observation in zip(
            (history + horizon[1:] + horizon[-1:])[-length:], row, strict=True
        )
    ]
    return _TargetSnapshot([item.observation for item in transitions], transitions, None)


def _collate_target_snapshot(pipeline: Any, snapshot: _TargetSnapshot) -> Any:
    if snapshot.collate_observations is not None:
        return snapshot.collate_observations(snapshot.observations)
    assert snapshot.transitions is not None
    standard = _standard_collation(pipeline.collate(snapshot.transitions))
    return standard["observations"] if standard is not None else tree_collate(snapshot.observations)
