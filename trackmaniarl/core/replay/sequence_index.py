"""Incremental uniform index of complete recurrent replay windows."""

from __future__ import annotations

from itertools import pairwise
from random import Random

from trackmaniarl.core.replay.store_support import _IncrementalReplayStore, _ReplayChange
from trackmaniarl.core.replay.structures import _FenwickTree


class SequenceIndex:
    """Track eligible final transition IDs in stable replay-slot order."""

    def __init__(self, store: _IncrementalReplayStore, length: int, n_step: int) -> None:
        self.store = store
        self.length = length
        self.n_step = n_step
        self.revision: int | None = None
        self.tree = _FenwickTree(store.capacity)
        self.ids: list[int | None] = [None] * store.capacity

    def synchronize(self) -> None:
        revision, changes = self.store.changes_since(self.revision)
        if changes is None:
            self.tree = _FenwickTree(self.store.capacity)
            self.ids = [None] * self.store.capacity
            for candidate in self.store.available_ids():
                self._refresh(candidate)
        else:
            candidates: set[int] = set()
            for change in changes:
                if change.evicted is not None:
                    self._remove(change.evicted)
                candidates.update(self._affected(change))
            for candidate in candidates:
                self._refresh(candidate)
        self.revision = revision

    def draw(self, rng: Random, count: int) -> list[int]:
        available = int(self.tree.total)
        if available < count:
            raise RuntimeError(f"Need {count} valid sequences, replay has {available}")
        result: list[int] = []
        for rank in rng.sample(range(available), count):
            candidate = self.ids[self.tree.find(float(rank))]
            if candidate is None:
                raise RuntimeError("sequence index is inconsistent with replay")
            result.append(candidate)
        return result

    def _affected(self, change: _ReplayChange) -> set[int]:
        candidates: set[int] = set()
        if change.appended is not None:
            candidates.update(self.store.affected_n_step_starts(change.appended, self.n_step))
            candidates.update(self.store.n_step_ids(change.appended, self.length))
        if change.evicted_previous is not None and self.n_step > 1:
            candidates.update(
                self.store.affected_n_step_starts(change.evicted_previous, self.n_step - 1)
            )
        if change.evicted_next is not None and self.length > 1:
            candidates.update(self.store.n_step_ids(change.evicted_next, self.length - 1))
        return candidates

    def _refresh(self, candidate: int) -> None:
        history = self.store.history_ids(candidate, self.length)
        if (
            self.store.contains(candidate)
            and self.store.is_n_step_eligible(candidate, self.n_step)
            and len(set(history)) == self.length
            # Out-of-order episode data can leave a predecessor pointer to a
            # terminal transition without a matching forward link. Such a
            # history crosses an episode boundary and must never be indexed.
            and all(
                self.store.n_step_ids(previous, 2) == [previous, following]
                for previous, following in pairwise(history)
            )
        ):
            slot = candidate % self.store.capacity
            self.ids[slot] = candidate
            self.tree.set(slot, 1.0)
        else:
            self._remove(candidate)

    def _remove(self, candidate: int) -> None:
        slot = candidate % self.store.capacity
        if self.ids[slot] == candidate:
            self.ids[slot] = None
            self.tree.set(slot, 0.0)
