from __future__ import annotations

from collections.abc import Sequence
from threading import Event, Thread
from typing import Any

import numpy as np
import pytest

from trackmaniarl.builtins.features import TransitionFeaturePipeline
from trackmaniarl.core.data import BatchRequest, Transition
from trackmaniarl.core.replay.prioritized import PrioritizedSampler
from trackmaniarl.core.replay.store import InMemoryReplayStore


@pytest.mark.parametrize(
    ("sequence_length", "wait_in"), [(1, "transitions"), (2, "transitions"), (2, "observations")]
)
def test_prioritized_collator_does_not_hold_replay_lock(sequence_length: int, wait_in: str) -> None:
    store = InMemoryReplayStore(capacity=3)

    def add(step: int) -> None:
        store.append(
            Transition(
                observation=np.array([step], dtype=np.float32),
                action=0,
                reward=1.0,
                next_observation=np.array([step + 1], dtype=np.float32),
                terminated=False,
                truncated=False,
                episode_id="episode",
                step=step,
            )
        )

    for step in range(3):
        add(step)
    entered, finished = Event(), Event()

    class WaitingPipeline(TransitionFeaturePipeline):
        def collate(self, transitions: list[Transition]) -> dict:
            if wait_in == "transitions":
                entered.set()
                if not finished.wait(3):
                    raise TimeoutError(
                        "producer blocked by replay lock during transition collation"
                    )
            return super().collate(transitions)

        def collate_observations(self, observations: Sequence[Any]) -> Any:
            if wait_in == "observations":
                entered.set()
                if not finished.wait(3):
                    raise TimeoutError(
                        "producer blocked by replay lock during observation collation"
                    )
            return super().collate_observations(observations)

    def producer() -> None:
        if entered.wait(3):
            for step in range(3, 6):
                add(step)
            finished.set()

    thread = Thread(target=producer)
    thread.start()
    try:
        sampler = PrioritizedSampler(WaitingPipeline())
        batch = sampler.sample(store, BatchRequest(batch_size=1, sequence_length=sequence_length))
        assert max(batch.observations.flatten().tolist()) < 3
        assert max(batch.next_observations.flatten().tolist()) <= 3
    finally:
        thread.join(timeout=4)
    assert not thread.is_alive()
