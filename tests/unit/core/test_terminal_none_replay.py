from dataclasses import replace

import pytest
import torch

from trackmaniarl.core.data import BatchRequest, Transition
from trackmaniarl.core.replay.store import InMemoryReplayStore


def terminal() -> Transition:
    return Transition(torch.ones(6), 39, 1.0, None, True, False, episode_id="end", step=0)


def test_explicit_terminal_none_survives_store_and_checkpoint_without_synthetic_next() -> None:
    store = InMemoryReplayStore(2)
    identity = store.append(terminal())
    assert store.get([identity])[0].next_observation is None
    restored = InMemoryReplayStore(2)
    restored.load_state_dict(store.state_dict())
    actual = restored.get([identity])[0]
    assert actual.terminated
    assert actual.next_observation is None
    materialized, discounts = restored.materialize_n_step([identity], BatchRequest(1, n_step=3))
    assert materialized[0].next_observation is None
    assert discounts == [0.0]


def test_terminal_none_stays_explicit_when_fifo_slots_wrap() -> None:
    store = InMemoryReplayStore(2)
    store.append(replace(terminal(), episode_id="old"))
    identity = store.append(terminal())
    store.append(replace(terminal(), episode_id="new"))
    assert store.get([identity])[0].next_observation is None
    assert len(store) == 2


@pytest.mark.parametrize("truncated", [False, True])
def test_absent_nonterminal_endpoint_is_not_accepted_as_optional(*, truncated: bool) -> None:
    store = InMemoryReplayStore(2)
    identity = store.append(replace(terminal(), terminated=False, truncated=truncated))
    with pytest.raises(RuntimeError, match="requires a nonterminal"):
        store.get([identity])
