"""The SDK replay store with explicit common8 duration/ragged sampling."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping
from fractions import Fraction
from typing import Any, cast

import torch

from trackmaniarl.core.data import BatchRequest, PriorityUpdate, TrainingBatch, Transition
from trackmaniarl.core.replay.store import InMemoryReplayStore
from trackmaniarl.trackmania.common_training.contract import (
    SCHEMA,
    CommonTrainingContract,
    finite_number,
)
from trackmaniarl.trackmania.common_training.features import (
    _validate_feature_digests,
    _validate_tensor_structure,
    _validate_transition_record,
    _validate_transition_reward,
    _validation_context,
    validate_transition,
)


class CommonReplayStore(InMemoryReplayStore):
    def __init__(self, contract: CommonTrainingContract, capacity: int = 100_000) -> None:
        self.contract = contract
        super().__init__(capacity)

    def append(self, transition: Transition) -> int:
        validate_transition(transition, self.contract)
        return super().append(transition)

    def available_id_range(self) -> range:
        """Snapshot the same dense ascending IDs without allocating their list.

        Samplers hold ``sampling_transaction`` until collation completes, so
        these IDs remain available while the sampled rows are materialized.
        """
        with self._lock:
            return range(self._next_index - self._size, self._next_index)

    def state_dict(self) -> dict[str, Any]:
        return {"common_contract": self.contract.envelope(), "store": super().state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.contract.require_checkpoint(state.get("common_contract", {}))
        # Validate a temporary store before replacing any live replay state.
        candidate = CommonReplayStore(self.contract, self.capacity)
        InMemoryReplayStore.load_state_dict(candidate, state["store"])
        for row in candidate.get(candidate.available_ids()):
            validate_transition(row, self.contract)
        super().load_state_dict(state["store"])


def collate_rows(
    rows: list[Transition], identities: list[int], contract: CommonTrainingContract
) -> TrainingBatch:
    if not rows or len(rows) != len(identities) or len(set(identities)) != len(rows):
        raise ValueError("nonempty rows with unique replay IDs required")
    expected = _validation_context(contract)
    observations, next_observations, indices = _stack_owned_observations(rows, expected.feature_dim)
    # Hash the actual copied learner inputs, using one NumPy view per stack.
    # No row hash or validation context survives this call or a later mutation.
    current_array = observations.detach().numpy()
    next_array = next_observations.detach().numpy()
    next_hashes = dict(
        zip(
            indices,
            (hashlib.sha256(value.tobytes()).hexdigest() for value in next_array),
            strict=True,
        )
    )
    durations, rewards, measurements = [], [], []
    for index, row in enumerate(rows):
        record, elapsed = _validate_transition_record(row, expected)
        _validate_feature_digests(
            record,
            hashlib.sha256(current_array[index].tobytes()).hexdigest(),
            next_hashes.get(index),
        )
        rewards.append(_validate_transition_reward(row, expected, record, elapsed))
        durations.append(elapsed)
        measurements.append(dict(record["reward_measurement"]))
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    if not bool(torch.isfinite(reward_tensor).all()):
        raise ValueError("transition reward differs from the shared reward specification")
    batch = TrainingBatch(
        data={"bootstrap_row_indices": torch.tensor(indices, dtype=torch.long)},
        observations=observations,
        actions=torch.tensor([row.action for row in rows], dtype=torch.long),
        rewards=reward_tensor,
        next_observations=next_observations,
        terminated=torch.tensor([row.terminated for row in rows], dtype=torch.bool),
        truncated=torch.tensor([row.truncated for row in rows], dtype=torch.bool),
        bootstrap_discounts=torch.tensor(
            [
                0.0
                if row.terminated
                else contract.gamma ** (elapsed / contract.discount_reference_seconds)
                for row, elapsed in zip(rows, durations, strict=True)
            ],
            dtype=torch.float32,
        ),
        transition_ids=identities,
        metadata={
            "common_ragged_contract": SCHEMA,
            "contract_sha256": expected.digest,
            "n_step": 1,
            "sequence_length": 1,
            "gamma": contract.gamma,
            "elapsed_seconds": durations,
            "reward_measurements": measurements,
        },
    )
    # No expected policy survives this call. Recheck its complete fresh digest
    # after collation so a changed contract/global specification cannot reach
    # the learner under the earlier batch identity.
    if contract.digest != expected.digest:
        raise ValueError("common training contract changed during batch collation")
    return batch


def _stack_owned_observations(
    rows: list[Transition], width: int
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Validate structures before copying, then finiteness on the owned stacks.

    The row-wise interval/clock/action/flags/reward/hash checks still run for
    every row after this copy. A mutable input alias cannot change the copied
    tensor between validation and its later construction for the learner.
    """
    for row in rows:
        if type(row.terminated) is not bool or type(row.truncated) is not bool:
            raise ValueError("explicit common interval flags required")
        _validate_tensor_structure(row.observation, width)
        if row.terminated:
            if row.next_observation is not None:
                raise ValueError("terminal NEXT must be absent")
        else:
            _validate_tensor_structure(row.next_observation, width)
    indices = [index for index, row in enumerate(rows) if not row.terminated]
    observations = torch.stack([cast(torch.Tensor, row.observation) for row in rows])
    next_observations = (
        torch.stack([cast(torch.Tensor, rows[index].next_observation) for index in indices])
        if indices
        else torch.empty((0, width), dtype=torch.float32)
    )
    if not bool(torch.isfinite(observations).all()) or not bool(
        torch.isfinite(next_observations).all()
    ):
        raise ValueError(f"actual finite CPU float32 common{width} tensor required")
    return observations, next_observations, indices


class CommonSampler:
    supports_sequence_sampling = False

    def __init__(self, contract: CommonTrainingContract, *, seed: int = 0) -> None:
        if type(seed) is not int:
            raise ValueError("integer common sampler seed required")
        self.contract, self._rng = contract, random.Random(seed)

    def sample(self, store: CommonReplayStore, request: BatchRequest) -> TrainingBatch:
        if (
            type(request.n_step) is not int
            or request.n_step != 1
            or type(request.sequence_length) is not int
            or request.sequence_length != 1
            or request.gamma != self.contract.gamma
            or type(request.batch_size) is not int
        ):
            raise ValueError("common sampler requires n-step-one frames and the declared gamma")
        self.contract.require_checkpoint(store.contract.envelope())
        with store.sampling_transaction():
            available = store.available_id_range()
            if len(available) < request.batch_size:
                raise RuntimeError("common replay has fewer rows than the requested batch")
            identities = self._rng.sample(available, request.batch_size)
            rows = store.get(identities)
            return collate_rows(rows, identities, self.contract)

    def update_priorities(self, update: PriorityUpdate) -> None:
        del update  # SDK uniform sampling policy, unchanged.

    def state_dict(self) -> dict[str, Any]:
        return {"common_contract": self.contract.envelope(), "rng": self._rng.getstate()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.contract.require_checkpoint(state.get("common_contract", {}))
        candidate = random.Random()
        candidate.setstate(state["rng"])
        self._rng = candidate


class DurationUpdateBudget:
    """Count applied simulated duration, independent of replay retention/reset.

    Call ``credit`` once after successfully storing a new real transition, and
    ``consume`` once after each successful learner update. Callers own this
    transaction boundary; reset, failed steps and future pending inputs add zero.

    Exact accounting sums the decimal representation of each canonically
    recorded elapsed value. It does not reconstruct timestamps or change units.
    Legacy checkpoints retain their saved approximation when first imported.
    """

    _tolerance = Fraction("1e-10")
    _exact_fields = ("elapsed_seconds_exact_numerator", "elapsed_seconds_exact_denominator")
    _max_exact_bits = 4096

    def __init__(self, contract: CommonTrainingContract) -> None:
        self.contract = contract
        self._elapsed_exact = Fraction(0)
        self._reference_exact = Fraction(str(contract.discount_reference_seconds))
        self._rate_exact = Fraction(str(contract.updates_per_reference))
        self.elapsed_seconds = 0.0
        self.transitions = 0
        self.updates = 0

    @property
    def credit_available(self) -> float:
        return float(max(Fraction(0), self._earned_exact(self._elapsed_exact) - self.updates))

    @property
    def integral_available(self) -> int:
        available = max(Fraction(0), self._earned_exact(self._elapsed_exact) - self.updates)
        return math.floor(available + self._tolerance)

    def _earned_exact(self, elapsed: Fraction) -> Fraction:
        return elapsed / self._reference_exact * self._rate_exact

    def credit(self, transition: Transition) -> None:
        elapsed = validate_transition(transition, self.contract)
        exact = self._elapsed_exact + Fraction(str(elapsed))
        reported = float(exact)
        self._elapsed_exact, self.elapsed_seconds = exact, reported
        self.transitions += 1

    def consume(self) -> None:
        if self.integral_available < 1:
            raise ValueError("no duration-earned update credit")
        self.updates += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "common_contract": self.contract.envelope(),
            "elapsed_seconds": self.elapsed_seconds,
            self._exact_fields[0]: self._elapsed_exact.numerator,
            self._exact_fields[1]: self._elapsed_exact.denominator,
            "transitions": self.transitions,
            "updates": self.updates,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.contract.require_checkpoint(state.get("common_contract", {}))
        elapsed = finite_number(state.get("elapsed_seconds"), "budget elapsed seconds")
        exact = self._load_exact_elapsed(state, elapsed)
        transitions, updates = state.get("transitions"), state.get("updates")
        if (
            elapsed < 0
            or type(transitions) is not int
            or transitions < 0
            or type(updates) is not int
            or updates < 0
            or (transitions == 0) != (elapsed == 0)
            or (transitions == 0) != (exact == 0)
            or updates > math.floor(self._earned_exact(exact) + self._tolerance)
        ):
            raise ValueError("invalid duration budget checkpoint")
        self.elapsed_seconds, self.transitions, self.updates = elapsed, transitions, updates
        self._elapsed_exact = exact

    @classmethod
    def _load_exact_elapsed(cls, state: Mapping[str, Any], elapsed: float) -> Fraction:
        present = [name in state for name in cls._exact_fields]
        if not any(present):
            # Preserve the historical saved approximation; do not infer a past
            # sequence of intervals from a transition count or elapsed value.
            return Fraction(str(elapsed))
        if not all(present):
            raise ValueError("both exact elapsed numerator and denominator are required")
        numerator, denominator = (state[name] for name in cls._exact_fields)
        if (
            type(numerator) is not int
            or type(denominator) is not int
            or numerator < 0
            or denominator <= 0
            or numerator.bit_length() > cls._max_exact_bits
            or denominator.bit_length() > cls._max_exact_bits
        ):
            raise ValueError("bounded nonnegative/positive exact elapsed integers required")
        exact = Fraction(numerator, denominator)
        try:
            matches_report = float(exact) == elapsed
        except OverflowError as error:
            raise ValueError("exact elapsed value exceeds its finite float report") from error
        if (exact.numerator, exact.denominator) != (numerator, denominator) or not matches_report:
            raise ValueError("canonical exact elapsed value must match the saved float report")
        return exact
