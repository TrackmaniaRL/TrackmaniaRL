"""Qualified native n-step-one replay; terminal endpoints remain absent.

This example ports the previously exercised ragged endpoint contract, without
importing research folders. It is not a replacement for native qualification.
"""

from __future__ import annotations

import random
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any, cast

import torch

from trackmaniarl.core.data import TrainingBatch
from trackmaniarl.trackmania.native_headless import NativeFeaturePipeline

CONTRACT = "tmrl.native-headless.example-ragged/v1"
PROFILE = "tmre.native16-pending-reference78/44/v2"
SCHEMA = "tmre.native16-pending-reference78/v1"
REFERENCE = "e256a74ca7ee82b3ea25287f49459a9d6e7773719d5eaafc996a9446f95907ae"
GAMMA = 0.995


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: Any) -> int:
    if type(value) is not int:
        raise ValueError("integer native metadata required")
    return value


def _flag(value: Any) -> bool:
    if type(value) is not bool:
        raise ValueError("explicit bool flags required")
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("actual native metadata mapping required")
    return value


def _features(state: Mapping[str, Any]) -> tuple[float, ...]:
    values = NativeFeaturePipeline().transform_observation(state)
    try:
        rounded = struct.unpack("<44f", struct.pack("<44f", *values))
    except (OverflowError, struct.error) as error:
        raise ValueError("actual44 must be representable as finite float32") from error
    _require(all(isfinite(value) for value in rounded), "finite float32 actual44 required")
    return tuple(float(value) for value in rounded)


@dataclass(frozen=True, slots=True)
class Row:
    observation: tuple[float, ...]
    action: int
    reward: float
    next_observation: tuple[float, ...] | None
    terminated: bool
    truncated: bool
    delta_native: int

    @property
    def discount(self) -> float:
        return 0.0 if self.terminated else GAMMA ** (self.delta_native / 50)


def _transition(before: Mapping[str, Any], action: int, result: Mapping[str, Any]) -> Row:
    current = _features(before)
    _require(type(action) is int and 0 <= action < 78, "action78 required")
    _require(result.get("trainingEligible") is True, "unqualified result never enters replay")
    terminal, truncated = _flag(result.get("terminated")), _flag(result.get("truncated"))
    _require(not (terminal and truncated), "contradictory endpoint flags")
    status = "terminated" if terminal else "truncated" if truncated else "continue"
    _require(result.get("status") == status, "status/flags mismatch")
    n = _integer(result.get("actualIterations"))
    effective = _integer(result.get("effectiveRequestedIterations"))
    requested = _integer(result.get("requestedIterations"))
    _require(1 <= n <= effective <= requested <= 5, "bounded actual/requested N required")
    _require(terminal or n == effective, "partial nonterminal boundary is inconclusive")
    dt = _integer(result.get("deltaNative"))
    _require(dt == 10 * n, "duration must equal actual native intervals")
    credit = _mapping(result.get("credit"))
    _require(
        type(credit.get("intervals")) is int
        and type(credit.get("nativeDelta")) is int
        and credit["intervals"] == n
        and credit["nativeDelta"] == dt
        and credit.get("pendingNotCredited") is True,
        "pending/interval credit mismatch",
    )
    after = _mapping(result.get("observation"))
    _require(
        after.get("schema") == SCHEMA and after.get("featureProfile") == PROFILE,
        "qualified endpoint profile required",
    )
    _require(
        _integer(after.get("nativeT")) == _integer(before.get("nativeT")) + dt,
        "native clock continuity required",
    )
    raw = after.get("native76Hex")
    try:
        valid_raw = isinstance(raw, str) and len(bytes.fromhex(raw)) == 76
    except ValueError:
        valid_raw = False
    _require(valid_raw, "actual native76 endpoint required")
    history = before.get("historyActionIds")
    if not isinstance(history, list):
        raise ValueError("actual input history required")
    _require(after.get("historyActionIds") == [*history, action][-2:], "history mismatch")
    projection = _mapping(after.get("projection"))
    _require(
        isinstance(projection, Mapping) and projection.get("referenceFingerprint") == REFERENCE,
        "reference profile mismatch",
    )
    reward = result.get("reward")
    if type(reward) not in (float, int):
        raise ValueError("finite actual reward required")
    reward_value = float(cast(float, reward))
    _require(isfinite(reward_value), "finite actual reward required")
    if terminal:
        _require(
            all(
                key in after and after[key] is None
                for key in ("features", "pendingSGB", "readerHex", "pendingOwnP")
            ),
            "terminal next features and pending input must remain absent",
        )
        _require(result.get("bootstrapFeatures") is None, "terminal bootstrap must be absent")
        outcome = _mapping(result.get("outcome"))
        _require(
            isinstance(outcome, Mapping)
            and outcome.get("status") == "terminated"
            and outcome.get("terminated") is True
            and outcome.get("truncated") is False
            and outcome.get("qualifiedBy") == "env002.full-original-validator+TMRESULT",
            "bound validator/TMRESULT proof required",
        )
        computed = _mapping(outcome.get("computedResult"))
        _require(
            isinstance(computed, Mapping)
            and type(computed.get("Time")) is int
            and 0 <= computed["Time"] < 0xFFFFFFFF
            and computed.get("NbCheckpoints") == 1,
            "computed finish required; IsValid is not finish",
        )
        next_observation = None
    else:
        next_observation = _features(after)
        bootstrap = {"features": result.get("bootstrapFeatures"), "featureProfile": PROFILE}
        _require(_features(bootstrap) == next_observation, "actual final44 bootstrap required")
    return Row(current, action, reward_value, next_observation, terminal, truncated, dt)


class Replay:
    """Bounded owned rows, uniform sampling without replacement, separate RNG."""

    def __init__(self, capacity: int, seed: int) -> None:
        _require(type(capacity) is int and capacity > 0, "positive capacity required")
        _require(type(seed) is int, "integer replay seed required")
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.rows: dict[int, Row] = {}
        self.next_id = 0

    def __len__(self) -> int:
        return len(self.rows)

    def append(self, before: Mapping[str, Any], action: int, result: Mapping[str, Any]) -> int:
        row = _transition(before, action, result)
        identity = self.next_id
        self.rows[identity] = row
        self.next_id += 1
        if len(self.rows) > self.capacity:
            del self.rows[next(iter(self.rows))]
        return identity

    def sample(self, batch_size: int) -> TrainingBatch:
        _require(
            type(batch_size) is int and 0 < batch_size <= len(self), "available batch required"
        )
        identities = self.rng.sample(list(self.rows), batch_size)
        rows = [self.rows[identity] for identity in identities]
        indices = [i for i, row in enumerate(rows) if not row.terminated]
        actual_next = [rows[i].next_observation for i in indices]
        return TrainingBatch(
            data={"bootstrap_row_indices": torch.tensor(indices, dtype=torch.long)},
            observations=torch.tensor([row.observation for row in rows], dtype=torch.float32),
            actions=torch.tensor([row.action for row in rows], dtype=torch.long),
            rewards=torch.tensor([row.reward for row in rows], dtype=torch.float32),
            next_observations=torch.tensor(actual_next, dtype=torch.float32).reshape(-1, 44),
            terminated=torch.tensor([row.terminated for row in rows], dtype=torch.bool),
            truncated=torch.tensor([row.truncated for row in rows], dtype=torch.bool),
            bootstrap_discounts=torch.tensor([row.discount for row in rows], dtype=torch.float32),
            transition_ids=identities,
            metadata={
                "native_ragged_contract": CONTRACT,
                "n_step": 1,
                "sequence_length": 1,
                "delta_native": [row.delta_native for row in rows],
            },
        )
