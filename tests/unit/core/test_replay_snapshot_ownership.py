from __future__ import annotations

import numpy as np

from trackmaniarl.core.data import Transition
from trackmaniarl.core.replay import InMemoryReplayStore


def test_snapshot_and_restored_replay_own_nested_metadata_and_overrides() -> None:
    store = InMemoryReplayStore(capacity=4)
    store.append(
        Transition(
            observation=np.array([1.0]),
            action=0,
            reward=1.0,
            next_observation={"pixels": np.array([2.0])},
            terminated=True,
            truncated=False,
            episode_id="episode",
            step=0,
            info={"nested": {"values": [np.array([3.0])]}},
        )
    )
    snapshot = store.state_dict()
    restored = InMemoryReplayStore(capacity=4)
    restored.load_state_dict(snapshot)

    snapshot["next_overrides"][0]["pixels"][0] = 20.0
    snapshot["info"][0]["nested"]["values"][0][0] = 30.0
    for replay in (store, restored):
        transition = replay.get([0])[0]
        assert transition.next_observation["pixels"][0] == 2.0
        assert transition.info["nested"]["values"][0][0] == 3.0

    fresh = restored.state_dict()
    fresh["info"][0]["nested"]["values"].append(4)
    assert len(restored.get([0])[0].info["nested"]["values"]) == 1
