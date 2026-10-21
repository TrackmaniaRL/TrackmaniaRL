"""Uniform native n-step-one sampling over the existing replay store.

Only real nonterminal endpoints form the bootstrap subset. Terminal rows retain
an absent next observation and receive a zero discount, never a fabricated44.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any, cast

import torch

from trackmaniarl.core.contracts import ReplayStore
from trackmaniarl.core.data import BatchRequest, PriorityUpdate, TrainingBatch, Transition
from trackmaniarl.trackmania.native_headless.training_contract import (
    NATIVE_INTERVAL_KEY,
    validate_native_transition,
)

CONTRACT = "tmrl.native-headless.example-ragged/v1"
GAMMA = 0.995


def _validate_request(request: BatchRequest) -> None:
    if (
        type(request.n_step) is not int
        or request.n_step != 1
        or type(request.sequence_length) is not int
        or request.sequence_length != 1
        or request.gamma != GAMMA
    ):
        raise ValueError("native ragged sampling requires n_step=sequence_length=1 and gamma=0.995")
    if type(request.batch_size) is not int or request.batch_size < 1:
        raise ValueError("positive integer native batch size required")


def _batch(rows: list[Transition], identities: list[int]) -> TrainingBatch:
    for row in rows:
        validate_native_transition(row)
    durations = [int(row.info[NATIVE_INTERVAL_KEY]["deltaNative"]) for row in rows]
    indices = [index for index, row in enumerate(rows) if not row.terminated]
    observations = torch.stack([cast(torch.Tensor, row.observation) for row in rows])
    next_observations = (
        torch.stack([cast(torch.Tensor, rows[index].next_observation) for index in indices])
        if indices
        else torch.empty((0, 44), dtype=torch.float32)
    )
    rewards = torch.tensor([row.reward for row in rows], dtype=torch.float32)
    if not bool(torch.isfinite(rewards).all()):
        raise ValueError("native rewards must remain finite in float32")
    return TrainingBatch(
        data={"bootstrap_row_indices": torch.tensor(indices, dtype=torch.long)},
        observations=observations,
        actions=torch.tensor([row.action for row in rows], dtype=torch.long),
        rewards=rewards,
        next_observations=next_observations,
        terminated=torch.tensor([row.terminated for row in rows], dtype=torch.bool),
        truncated=torch.tensor([row.truncated for row in rows], dtype=torch.bool),
        bootstrap_discounts=torch.tensor(
            [
                0.0 if row.terminated else GAMMA ** (dt / 50)
                for row, dt in zip(rows, durations, strict=True)
            ],
            dtype=torch.float32,
        ),
        transition_ids=identities,
        metadata={
            "native_ragged_contract": CONTRACT,
            "n_step": 1,
            "sequence_length": 1,
            "gamma": GAMMA,
            "delta_native": durations,
        },
    )


class NativeRaggedSampler:
    """Select stable store IDs uniformly without replacement, with owned RNG.

    ``pipeline`` is accepted for the normal SDK component resolver. The native
    tensors have already been validated, so generic dense collation is bypassed.
    """

    supports_sequence_sampling = False

    def __init__(self, pipeline: Any = None, seed: int = 0) -> None:
        del pipeline
        if type(seed) is not int:
            raise ValueError("integer native sampler seed required")
        self._rng = random.Random(seed)

    def sample(self, store: ReplayStore, request: BatchRequest) -> TrainingBatch:
        _validate_request(request)
        transaction = getattr(store, "sampling_transaction", None)
        with transaction() if callable(transaction) else nullcontext():
            available = store.available_ids()
            if any(type(identity) is not int or identity < 0 for identity in available) or len(
                set(available)
            ) != len(available):
                raise ValueError("unique nonnegative integer native replay IDs required")
            if len(available) < request.batch_size:
                raise RuntimeError("native replay has fewer rows than the requested batch")
            identities = self._rng.sample(available, request.batch_size)
            rows = store.get(identities)
            if len(rows) != len(identities):
                raise ValueError("native replay returned a different row count")
            return _batch(rows, identities)

    def update_priorities(self, update: PriorityUpdate) -> None:
        del update  # This is the same uniform policy as the SDK UniformSampler.

    def state_dict(self) -> dict[str, Any]:
        return {"native_ragged_contract": CONTRACT, "rng": self._rng.getstate()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("native_ragged_contract") != CONTRACT:
            raise ValueError("native ragged sampler checkpoint contract mismatch")
        self._rng.setstate(state["rng"])
