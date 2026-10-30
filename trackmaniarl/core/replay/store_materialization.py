"""Transition materialization for in-memory replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from trackmaniarl.core.data import BatchRequest, Transition, TransitionId
from trackmaniarl.core.pytree import tree_snapshot

if TYPE_CHECKING:
    from trackmaniarl.core.replay.store import InMemoryReplayStore


@dataclass(frozen=True, slots=True)
class _StoredHorizon:
    store: InMemoryReplayStore
    request: BatchRequest
    transition_id: TransitionId
    first_slot: int
    episode_code: int
    first_step: int


@dataclass(slots=True)
class _StoredReturn:
    final_id: TransitionId
    reward: float = 0.0
    discount: float = 1.0
    steps: int = 0
    terminated: bool = False
    truncated: bool = False

    def add(self, store: InMemoryReplayStore, current_id: TransitionId, gamma: float) -> None:
        slot = current_id % store.capacity
        self.final_id = current_id
        self.reward += self.discount * float(store._rewards[slot])
        self.steps += 1
        self.terminated = bool(store._terminated[slot])
        self.truncated = bool(store._truncated[slot])
        if not (self.terminated or self.truncated):
            self.discount *= gamma


def get(store: InMemoryReplayStore, transition_ids: list[TransitionId]) -> list[Transition]:
    with store._lock:
        missing = [
            transition_id for transition_id in transition_ids if not store.contains(transition_id)
        ]
        if missing:
            raise KeyError(f"Replay transitions no longer available: {missing[:3]}")
        return [store._transition(transition_id) for transition_id in transition_ids]


def materialize_n_step(
    store: InMemoryReplayStore, transition_ids: list[TransitionId], request: BatchRequest
) -> tuple[list[Transition], list[float]]:
    """Aggregate n-step targets while reading only their endpoint observations."""

    with store._lock:
        missing = [
            transition_id for transition_id in transition_ids if not store.contains(transition_id)
        ]
        if missing:
            raise KeyError(f"Replay transitions no longer available: {missing[:3]}")
        if request.n_step == 1 and len(transition_ids) > 1:
            return _materialize_single_step_batch(store, transition_ids, request.gamma)
        materialized = [
            store._materialize_n_step_locked(transition_id, request)
            for transition_id in transition_ids
        ]
    return [item[0] for item in materialized], [item[1] for item in materialized]


def _materialize_single_step_batch(
    store: InMemoryReplayStore, transition_ids: list[TransitionId], gamma: float
) -> tuple[list[Transition], list[float]]:
    """Gather the one-step horizon without per-row traversal/context objects.

    Caller holds the replay lock and has checked every selected ID. Column
    gathers copy into owned arrays, including duplicate IDs. Terminal overrides
    and nonterminal missing endpoints retain the strict endpoint checks.
    """
    assert store._observations is not None
    assert store._actions is not None
    slots = np.asarray(transition_ids, dtype=np.int64) % store.capacity
    observations = store._observations.read_many_owned(slots)
    actions = store._actions.read_many_owned(slots)
    following = _bulk_next_observations(store, transition_ids, slots)
    rows, discounts = [], []
    for index, (transition_id, slot) in enumerate(zip(transition_ids, slots, strict=True)):
        terminated = bool(store._terminated[slot])
        step, code = int(store._steps[slot]), int(store._episode_codes[slot])
        rows.append(
            Transition(
                observation=observations[index],
                action=actions[index],
                # Match the reference horizon's initial +0.0, including signed zero.
                reward=0.0 + float(store._rewards[slot]),
                next_observation=following[index],
                terminated=terminated,
                truncated=bool(store._truncated[slot]),
                info=store._info.get(transition_id, {}),
                episode_id=store._episode_names.get(code),
                step=step if step >= 0 else None,
            )
        )
        discounts.append(0.0 if terminated else gamma**1)
    return rows, discounts


def _bulk_next_observations(
    store: InMemoryReplayStore,
    transition_ids: list[TransitionId],
    slots: np.ndarray[Any, np.dtype[np.int64]],
) -> list[Any]:
    assert store._observations is not None
    following: list[Any] = [None] * len(transition_ids)
    linked = []
    for index, (transition_id, slot) in enumerate(zip(transition_ids, slots, strict=True)):
        if transition_id in store._next_overrides:
            following[index] = _owned_next_observation(store, transition_id, int(slot))
        else:
            next_id = int(store._next_ids[slot])
            if not store.contains(next_id):
                raise RuntimeError(f"Transition {transition_id} has no next observation")
            linked.append((index, next_id % store.capacity))
    if linked:
        next_slots = np.asarray([slot for _, slot in linked], dtype=np.int64)
        values = store._observations.read_many_owned(next_slots)
        for (index, _), value in zip(linked, values, strict=True):
            following[index] = value
    return following


def _materialize_n_step_locked(
    store: InMemoryReplayStore, transition_id: TransitionId, request: BatchRequest
) -> tuple[Transition, float]:
    assert store._observations is not None
    assert store._actions is not None
    context = _stored_horizon(store, transition_id, request)
    result = _accumulate_horizon(context)
    if result.steps == 0:
        raise RuntimeError(f"Transition {transition_id} is no longer available")
    transition = _materialized_transition(context, result)
    discount = 0.0 if result.terminated else request.gamma**result.steps
    return transition, discount


def _stored_horizon(
    store: InMemoryReplayStore, transition_id: TransitionId, request: BatchRequest
) -> _StoredHorizon:
    first_slot = transition_id % store.capacity
    return _StoredHorizon(
        store,
        request,
        transition_id,
        first_slot,
        int(store._episode_codes[first_slot]),
        int(store._steps[first_slot]),
    )


def _accumulate_horizon(context: _StoredHorizon) -> _StoredReturn:
    current_id = context.transition_id
    result = _StoredReturn(context.transition_id)
    for offset in range(context.request.n_step):
        candidate = _stored_candidate(context, current_id, offset)
        if candidate is None:
            break
        result.add(context.store, candidate, context.request.gamma)
        if result.terminated or result.truncated:
            break
        current_id = int(context.store._next_ids[candidate % context.store.capacity])
    return result


def _stored_candidate(
    context: _StoredHorizon, current_id: TransitionId, offset: int
) -> TransitionId | None:
    store = context.store
    if not store.contains(current_id):
        return None
    slot = current_id % store.capacity
    if int(store._episode_codes[slot]) != context.episode_code:
        return None
    current_step = int(store._steps[slot])
    continuous = (
        context.first_step < 0 or current_step < 0 or current_step == context.first_step + offset
    )
    return current_id if continuous else None


def _materialized_transition(context: _StoredHorizon, result: _StoredReturn) -> Transition:
    store = context.store
    assert store._observations is not None
    assert store._actions is not None
    final_slot = result.final_id % store.capacity
    return Transition(
        observation=store._observations.read_owned(context.first_slot),
        action=store._actions.read_owned(context.first_slot),
        reward=result.reward,
        next_observation=_owned_next_observation(store, result.final_id, final_slot),
        terminated=result.terminated,
        truncated=result.truncated,
        info=store._info.get(context.transition_id, {}),
        episode_id=store._episode_names.get(context.episode_code),
        step=context.first_step if context.first_step >= 0 else None,
    )


def _transition(store: InMemoryReplayStore, transition_id: TransitionId) -> Transition:
    assert store._observations is not None
    assert store._actions is not None
    slot = transition_id % store.capacity
    episode_code = int(store._episode_codes[slot])
    step = int(store._steps[slot])
    return Transition(
        observation=store._observations.read(slot),
        action=store._actions.read(slot),
        reward=float(store._rewards[slot]),
        next_observation=store._next_observation(transition_id, slot),
        terminated=bool(store._terminated[slot]),
        truncated=bool(store._truncated[slot]),
        info=store._info.get(transition_id, {}),
        episode_id=store._episode_names.get(episode_code),
        step=step if step >= 0 else None,
    )


def _owned_next_observation(
    store: InMemoryReplayStore, transition_id: TransitionId, slot: int
) -> Any:
    assert store._observations is not None
    if transition_id in store._next_overrides:
        return tree_snapshot(_next_observation(store, transition_id, slot))
    next_id = int(store._next_ids[slot])
    if not store.contains(next_id):
        raise RuntimeError(f"Transition {transition_id} has no next observation")
    return store._observations.read_owned(next_id % store.capacity)


def _next_observation(store: InMemoryReplayStore, transition_id: TransitionId, slot: int) -> Any:
    assert store._observations is not None
    if transition_id in store._next_overrides:
        next_observation = store._next_overrides[transition_id]
        if next_observation is None and not bool(store._terminated[slot]):
            raise RuntimeError(
                f"Transition {transition_id} requires a nonterminal next observation"
            )
        return next_observation
    next_id = int(store._next_ids[slot])
    if not store.contains(next_id):
        raise RuntimeError(f"Transition {transition_id} has no next observation")
    return store._observations.read(next_id % store.capacity)
