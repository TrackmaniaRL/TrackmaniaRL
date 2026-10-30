"""Owned replay reads must survive mutation, duplicate selection and ring overwrite."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import numpy as np
import pytest
import torch

from trackmaniarl.core.builtins import TorchCheckpointCodec
from trackmaniarl.core.data import BatchRequest, Transition
from trackmaniarl.core.pytree import tree_snapshot
from trackmaniarl.core.replay import InMemoryReplayStore
from trackmaniarl.core.replay.store_support import _TreeColumns
from trackmaniarl.distributed.coordinator_support import snapshot_state_dict

_CUDA_AVAILABLE = torch.cuda.is_available()


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

    many = columns.read_many_owned(np.array([0, 0], dtype=np.int64))
    torch.testing.assert_close(many[0]["tensor"], expected["tensor"], rtol=0, atol=0)
    many[0]["tensor"].fill_(44)
    many[0]["nested"][0][0].fill_(False)
    torch.testing.assert_close(many[1]["tensor"], expected["tensor"], rtol=0, atol=0)
    assert many[1]["nested"][0][0].item()
    np.testing.assert_array_equal(many[1]["array"], expected["array"])
    if not scalar:
        many[0]["array"].fill(44)
        np.testing.assert_array_equal(many[1]["array"], expected["array"])
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


@pytest.mark.parametrize("ending", ["terminated", "truncated", "continuing"])
def test_bulk_one_step_matches_reference_after_ring_overwrite(ending: str) -> None:
    store = InMemoryReplayStore(capacity=5)
    for index in range(7):
        ended = index % 3 == 2
        terminated = ended and ending == "terminated"
        store.append(
            Transition(
                observation={"x": torch.tensor([float(index), -0.0])},
                action=torch.tensor(index % 3),
                reward=-0.0 if index % 2 else float(index),
                next_observation=None
                if terminated
                else {"x": torch.tensor([float(index + 1), -0.0])},
                terminated=terminated,
                truncated=ended and ending == "truncated",
                info={"evidence": index},
                episode_id=str(index // 3),
                step=index % 3,
            )
        )
    ids = [2, 4, 6, 6, 5]
    request = BatchRequest(len(ids), n_step=1, gamma=0.995)
    reference = [store._materialize_n_step_locked(item, request) for item in ids]
    actual, discounts = store.materialize_n_step(ids, request)
    assert discounts == [discount for _, discount in reference]
    for row, (expected, _) in zip(actual, reference, strict=True):
        torch.testing.assert_close(row.observation["x"], expected.observation["x"], rtol=0, atol=0)
        torch.testing.assert_close(row.action, expected.action, rtol=0, atol=0)
        if expected.next_observation is None:
            assert row.next_observation is None
        else:
            torch.testing.assert_close(
                row.next_observation["x"], expected.next_observation["x"], rtol=0, atol=0
            )
        assert (row.reward, row.terminated, row.truncated, row.info, row.episode_id, row.step) == (
            expected.reward,
            expected.terminated,
            expected.truncated,
            expected.info,
            expected.episode_id,
            expected.step,
        )
    actual[2].observation["x"].fill_(44)
    actual[2].action.fill_(44)
    actual[2].next_observation["x"].fill_(44)
    torch.testing.assert_close(actual[3].observation["x"], reference[3][0].observation["x"])
    torch.testing.assert_close(actual[3].action, reference[3][0].action)
    torch.testing.assert_close(
        actual[3].next_observation["x"], reference[3][0].next_observation["x"]
    )
    with pytest.raises(KeyError, match="no longer available"):
        store.materialize_n_step([0, 6], request)
    torch.testing.assert_close(store.get([6])[0].observation["x"], torch.tensor([6.0, -0.0]))


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA"))],
)
def test_checkpoint_snapshot_owns_ring_data_and_metadata_during_later_writes(
    device: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if device == "cuda":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    store = InMemoryReplayStore(capacity=2)

    def append(step: int) -> None:
        store.append(
            Transition(
                observation={"x": torch.tensor([float(step)], device=device)},
                action=step % 2,
                reward=float(step),
                next_observation={"x": torch.tensor([float(step + 1)], device=device)},
                terminated=False,
                truncated=False,
                info={"tensor": torch.tensor([float(step)], device=device, requires_grad=True)},
                episode_id="episode",
                step=step,
            )
        )

    for step in range(3):
        append(step)
    snapshot = snapshot_state_dict(store)
    saved_metadata = snapshot["info"][2]["tensor"]
    assert saved_metadata.device.type == "cpu"
    assert not saved_metadata.requires_grad
    path = tmp_path / "snapshot.pt"
    codec = TorchCheckpointCodec()
    entered, resume = Event(), Event()

    def write() -> None:
        entered.set()
        if not resume.wait(10):
            raise TimeoutError("checkpoint writer was not released")
        codec.save(snapshot, path)

    # A background writer retains the snapshot while live metadata and ring
    # rows change. It must subsequently serialize the original frozen values.
    with ThreadPoolExecutor(max_workers=1) as writer:
        pending = writer.submit(write)
        try:
            assert entered.wait(10)
            live_metadata = store.state_dict()["info"][2]["tensor"]
            live_metadata.fill_(99)
            append(3)
        finally:
            resume.set()
        pending.result(timeout=20)
    torch.testing.assert_close(saved_metadata, torch.tensor([2.0]))
    snapshot["observations"]["arrays"][0].fill(-5)
    saved_metadata.fill_(7)
    torch.testing.assert_close(store.get([2])[0].observation["x"], torch.tensor([2.0]))
    torch.testing.assert_close(live_metadata.cpu(), torch.tensor([99.0]))

    restored = InMemoryReplayStore(capacity=2)
    restored.load_state_dict(codec.load(path))
    assert restored.available_ids() == [1, 2]
    rows = restored.get([1, 2])
    for step, row in zip((1, 2), rows, strict=True):
        torch.testing.assert_close(row.observation["x"], torch.tensor([float(step)]))
        torch.testing.assert_close(row.next_observation["x"], torch.tensor([float(step + 1)]))
        torch.testing.assert_close(row.info["tensor"], torch.tensor([float(step)]))
        assert row.action == step % 2


def test_legacy_component_checkpoint_still_detaches_mutable_state() -> None:
    class LegacyState:
        def __init__(self) -> None:
            self.values = np.arange(4, dtype=np.float32)
            self.tensor = torch.tensor([2.0], requires_grad=True)

        def state_dict(self) -> dict[str, object]:
            return {"values": self.values, "tensor": self.tensor}

    component = LegacyState()
    snapshot = snapshot_state_dict(component)
    component.values.fill(99)
    with torch.no_grad():
        component.tensor.fill_(99)
    np.testing.assert_array_equal(snapshot["values"], np.arange(4, dtype=np.float32))
    torch.testing.assert_close(snapshot["tensor"], torch.tensor([2.0]))
    assert not snapshot["tensor"].requires_grad


def test_explicit_checkpoint_snapshot_must_return_a_mapping() -> None:
    class InvalidSnapshot:
        def snapshot_state_dict(self) -> list[int]:
            return [1]

    with pytest.raises(TypeError, match=r"snapshot_state_dict.*must return a mapping"):
        snapshot_state_dict(InvalidSnapshot())
