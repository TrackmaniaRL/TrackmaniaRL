"""Owned replay reads must survive mutation, duplicate selection and ring overwrite."""

import numpy as np
import pytest
import torch

from trackmaniarl.core.data import BatchRequest, Transition
from trackmaniarl.core.pytree import tree_snapshot
from trackmaniarl.core.replay import InMemoryReplayStore
from trackmaniarl.core.replay.store_support import _TreeColumns


@pytest.mark.parametrize("scalar", [False, True])
def test_owned_columns_preserve_types_dtypes_and_independence(*, scalar: bool) -> None:
    value = {
        "tensor": torch.tensor(3, dtype=torch.int64) if scalar else torch.arange(4.0),
        "array": np.array(2, dtype=np.float32) if scalar else np.arange(4, dtype=np.int16),
        "nested": ([torch.tensor(True)], (7.0,)),
    }
    columns = _TreeColumns(2, value)
    columns.write(0, value)
    expected = tree_snapshot(columns.read(0))
    owned = columns.read_owned(0)
    assert type(owned["array"]) is type(expected["array"])
    assert owned["array"].dtype == expected["array"].dtype
    np.testing.assert_array_equal(owned["array"], expected["array"])
    torch.testing.assert_close(owned["tensor"], expected["tensor"], rtol=0, atol=0)
    owned["tensor"].fill_(99)
    owned["nested"][0][0].fill_(False)
    fresh = columns.read_owned(0)
    torch.testing.assert_close(fresh["tensor"], expected["tensor"], rtol=0, atol=0)
    assert fresh["nested"][0][0].item()
    if not scalar:
        owned["array"].fill(99)
        np.testing.assert_array_equal(columns.read(0)["array"], expected["array"])


@pytest.mark.parametrize(
    ("terminated", "truncated"), [(False, False), (True, False), (False, True)]
)
def test_owned_materialization_preserves_boundaries_and_duplicate_rows(
    *, terminated: bool, truncated: bool
) -> None:
    store = InMemoryReplayStore(capacity=3)
    for step in range(3):
        store.append(
            Transition(
                observation={"x": torch.tensor([float(step)])},
                action=torch.tensor(step),
                reward=float(step + 1),
                next_observation={"x": torch.tensor([float(step + 1)])},
                terminated=terminated and step == 1,
                truncated=truncated and step == 1,
                episode_id="first",
                step=step,
            )
        )
    rows, discounts = store.materialize_n_step([0, 0], BatchRequest(2, n_step=2, gamma=0.5))
    assert rows[0].reward == rows[1].reward == 2.0
    assert discounts == ([0.0, 0.0] if terminated else [0.25, 0.25])
    assert rows[0].terminated == terminated
    assert rows[0].truncated == truncated
    rows[0].observation["x"].fill_(99)
    rows[0].action.fill_(99)
    rows[0].next_observation["x"].fill_(99)
    assert rows[1].observation["x"].item() == 0.0
    assert rows[1].action.item() == 0
    assert rows[1].next_observation["x"].item() == 2.0
    assert store.get([0])[0].observation["x"].item() == 0.0
    for step in range(3):
        store.append(
            Transition(
                observation={"x": torch.tensor([100.0 + step])},
                action=torch.tensor(8),
                reward=0.0,
                next_observation={"x": torch.tensor([101.0 + step])},
                terminated=False,
                truncated=False,
                episode_id="overwrite",
                step=step,
            )
        )
    assert rows[1].observation["x"].item() == 0.0
    assert rows[1].next_observation["x"].item() == 2.0
