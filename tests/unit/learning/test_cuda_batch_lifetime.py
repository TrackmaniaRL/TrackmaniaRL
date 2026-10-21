"""Cross-stream replay batches must retain every CUDA allocation during use."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch

from trackmaniarl.algorithms._torch import TorchLearnerBase
from trackmaniarl.algorithms.torch_batches import transform_batch
from trackmaniarl.core.data import TrainingBatch
from trackmaniarl.core.pytree import tree_map


class _CudaTensor(torch.Tensor):
    @property
    def is_cuda(self) -> bool:
        return True


class _ConsumerStream:
    def __init__(self) -> None:
        self.events: list[object] = []

    def wait_event(self, event: object) -> None:
        self.events.append(event)


def _batch() -> TrainingBatch:
    return TrainingBatch(
        data={"nested": (torch.ones(2),)},
        observations={"first": torch.ones(2), "second": [torch.zeros(2)]},
        actions=torch.zeros(2),
        rewards=torch.ones(2),
        next_observations=torch.ones(2),
        terminated=torch.zeros(2, dtype=torch.bool),
        truncated=torch.zeros(2, dtype=torch.bool),
        bootstrap_discounts=torch.ones(2),
        transition_ids=[0, 1],
        importance_weights=torch.ones(2),
        masks=torch.ones(2, dtype=torch.bool),
        metadata={"unchanged": "metadata"},
    )


def test_prefetched_batch_records_all_tensors_on_consumer_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = TorchLearnerBase()
    learner.device = torch.device("cuda:0")
    consumer = _ConsumerStream()
    event = object()

    def as_cuda(value: Any) -> Any:
        return tree_map(lambda leaf: leaf.as_subclass(_CudaTensor), value)

    staged = transform_batch(_batch(), as_cuda)
    staged = replace(
        staged,
        metadata={**staged.metadata, "_trackmaniarl_transfer_event": event},
    )
    recorded: set[int] = set()

    def record(tensor: torch.Tensor, stream: _ConsumerStream) -> None:
        assert stream is consumer
        recorded.add(tensor.data_ptr())

    monkeypatch.setattr(torch.Tensor, "record_stream", record)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: consumer)
    consumed = learner._batch(staged)

    expected: set[int] = set()

    def collect(value: object) -> object:
        if isinstance(value, torch.Tensor) and value.is_cuda:
            expected.add(value.data_ptr())
        return value

    for field in (
        "data",
        "observations",
        "actions",
        "rewards",
        "next_observations",
        "terminated",
        "truncated",
        "bootstrap_discounts",
        "importance_weights",
        "masks",
    ):
        tree_map(collect, getattr(staged, field))
    assert recorded == expected
    assert len(recorded) == 11
    assert consumed.metadata == {"unchanged": "metadata"}
    assert consumer.events == [event]
