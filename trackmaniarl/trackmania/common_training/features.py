"""Common raw endpoint adapters and backend-neutral, duration-bearing rows."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from trackmaniarl.core.data import Transition
from trackmaniarl.trackmania.common_training.contract import (
    ClockSpec,
    CommonTrainingContract,
    RewardMeasurement,
    RewardSpec,
    canonical,
    finite_number,
    require_digest,
)
from trackmaniarl.trackmania.parity import COMMON_SCALES, from_connect33, from_native

COMMON_INTERVAL_KEY = "_trackmaniarl_common_interval"
INTERVAL_SCHEMA = "tmrl.common-kinematics.actual-interval/v1"
BACKENDS = ("windows-connect33", "native-headless")


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("explicit common metadata mapping required")
    return value


def tensor_digest(value: torch.Tensor | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _validate_tensor_structure(value: Any, width: int) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != (width,)
        or value.dtype != torch.float32
        or value.device.type != "cpu"
        or value.layout != torch.strided
    ):
        raise ValueError(f"actual finite CPU float32 common{width} tensor required")


def _validate_tensor(value: Any, width: int) -> None:
    _validate_tensor_structure(value, width)
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"actual finite CPU float32 common{width} tensor required")


@dataclass(frozen=True, kw_only=True)
class CommonTransitionInput:
    """The producer supplies actual stamps/geometry and a journal artifact hash.

    Raw endpoints are tagged ``{backend, raw}``; neither synthetic terminal
    features nor an inferred native milliseconds conversion is accepted here.
    """

    observation: Mapping[str, Any]
    next_observation: Mapping[str, Any] | None
    action: int
    before_timestamp: int | float
    after_timestamp: int | float
    reward_measurement: RewardMeasurement
    terminated: bool
    truncated: bool
    raw_transition_sha256: str
    episode_id: str | None = None
    step: int | None = None


class CommonFeaturePipeline:
    terminal_next_observation_optional = True
    requires_environment_endpoint = True

    def __init__(self, contract: CommonTrainingContract) -> None:
        self.contract = contract

    def transform_observation(self, observation: Any) -> torch.Tensor:
        endpoint = _mapping(observation)
        backend = endpoint.get("backend")
        if self.contract.feature_dim == 6:
            values = self._kinematics6(endpoint)
        elif backend == "windows-connect33":
            if endpoint.get("raw") is None:
                raise ValueError("actual Connect33 packet required")
            values = from_connect33(endpoint["raw"])
        elif backend == "native-headless":
            values = from_native(_mapping(endpoint.get("raw")))
        else:
            raise ValueError("explicit supported raw common8 backend endpoint required")
        result = torch.tensor(values, dtype=torch.float32, device="cpu")
        _validate_tensor(result, self.contract.feature_dim)
        return result

    @staticmethod
    def _kinematics6(endpoint: Mapping[str, Any]) -> np.ndarray:
        if endpoint.get("backend") == "windows-connect33":
            raw = endpoint.get("raw")
            if raw is None or len(raw) != 33:
                raise ValueError("actual Connect33 packet required")
            selected = raw[4:10]
        elif endpoint.get("backend") == "native-headless":
            raw = _mapping(endpoint.get("raw"))
            if raw.get("native76Hex") is None:
                raise ValueError("actual native endpoint is unavailable")
            packet = bytes.fromhex(raw["native76Hex"])
            if len(packet) != 76:
                raise ValueError("exact qualified native76 required")
            selected = struct.unpack_from("<6f", packet, 16)
        else:
            raise ValueError("explicit supported common6 backend endpoint required")
        values = np.asarray(selected, dtype=np.float64)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError("six actual finite kinematics fields required")
        with np.errstate(over="ignore"):
            result = (values / COMMON_SCALES[:6]).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("common6 observation overflows float32")
        return result

    def validate_replay_transition(self, transition: Transition) -> None:
        validate_transition(transition, self.contract)

    def replay_info(self, info: Mapping[str, Any]) -> Mapping[str, Any]:
        record = deepcopy(dict(_mapping(info.get(COMMON_INTERVAL_KEY))))
        validate_interval(record, self.contract)
        return {COMMON_INTERVAL_KEY: record}

    def collate(self, transitions: list[Transition]) -> Any:
        del transitions
        raise ValueError("common8 terminal endpoints require CommonSampler ragged collation")


def make_transition(source: CommonTransitionInput, contract: CommonTrainingContract) -> Transition:
    pipeline = CommonFeaturePipeline(contract)
    if type(source.terminated) is not bool or type(source.truncated) is not bool:
        raise ValueError("explicit boolean transition flags required")
    if source.terminated and source.truncated:
        raise ValueError("transition cannot be both terminated and truncated")
    if source.reward_measurement.finished and not source.terminated:
        raise ValueError("verified finish must terminate the transition")
    if source.terminated and source.next_observation is not None:
        raise ValueError("terminal common NEXT must remain absent")
    backend = source.observation.get("backend")
    if not source.terminated and (
        source.next_observation is None or source.next_observation.get("backend") != backend
    ):
        raise ValueError("continuing/truncated NEXT must retain an actual same-backend endpoint")
    observation = pipeline.transform_observation(source.observation)
    next_observation = (
        None if source.terminated else pipeline.transform_observation(source.next_observation)
    )
    elapsed = contract.clock.elapsed(source.before_timestamp, source.after_timestamp)
    reward = contract.reward.score(source.reward_measurement, elapsed)
    record = {
        "schema": INTERVAL_SCHEMA,
        "contract_sha256": contract.digest,
        "backend": backend,
        "before_timestamp": source.before_timestamp,
        "after_timestamp": source.after_timestamp,
        "elapsed_seconds": elapsed,
        "clock": asdict(contract.clock),
        "action": source.action,
        "terminated": source.terminated,
        "truncated": source.truncated,
        "reward_measurement": asdict(source.reward_measurement),
        "raw_transition_sha256": source.raw_transition_sha256,
        "observation_sha256": tensor_digest(observation),
        "next_observation_sha256": tensor_digest(next_observation),
    }
    row = Transition(
        observation,
        source.action,
        reward,
        next_observation,
        source.terminated,
        source.truncated,
        {COMMON_INTERVAL_KEY: record},
        source.episode_id,
        source.step,
    )
    validate_transition(row, contract)
    return row


@dataclass(frozen=True, slots=True)
class _TransitionValidation:
    """Expected policy for one validation call; never retained between batches."""

    digest: str
    clock_json: str
    clock: ClockSpec
    reward: RewardSpec
    feature_dim: int


def _validation_context(contract: CommonTrainingContract) -> _TransitionValidation:
    return _TransitionValidation(
        contract.digest,
        canonical(asdict(contract.clock)),
        contract.clock,
        contract.reward,
        contract.feature_dim,
    )


def validate_interval(record: Mapping[str, Any], contract: CommonTrainingContract) -> float:
    return _validate_interval(record, _validation_context(contract))


def _validate_interval(record: Mapping[str, Any], expected: _TransitionValidation) -> float:
    record = _mapping(record)
    if record.get("schema") != INTERVAL_SCHEMA or record.get("contract_sha256") != expected.digest:
        raise ValueError("common interval contract mismatch")
    if record.get("backend") not in BACKENDS:
        raise ValueError("explicit common interval backend required")
    if canonical(record.get("clock")) != expected.clock_json:
        raise ValueError("interval clock unit/conversion/evidence mismatch")
    elapsed = expected.clock.elapsed(record.get("before_timestamp"), record.get("after_timestamp"))
    if record.get("elapsed_seconds") != elapsed:
        raise ValueError("elapsed time must derive from actual timestamps")
    if type(record.get("action")) is not int or not 0 <= record["action"] < 78:
        raise ValueError("explicit discrete78 action required")
    if any(type(record.get(flag)) is not bool for flag in ("terminated", "truncated")):
        raise ValueError("explicit common interval flags required")
    if record["terminated"] and record["truncated"]:
        raise ValueError("contradictory terminal/truncated flags")
    require_digest(record.get("raw_transition_sha256"), "raw transition journal")
    require_digest(record.get("observation_sha256"), "current observation")
    if record["terminated"]:
        if record.get("next_observation_sha256") is not None:
            raise ValueError("terminal NEXT must be absent")
    else:
        require_digest(record.get("next_observation_sha256"), "actual bootstrap observation")
    measurement = RewardMeasurement(**_mapping(record.get("reward_measurement")))
    if measurement.finished and not record["terminated"]:
        raise ValueError("finish measurement without terminal outcome")
    return elapsed


def validate_transition(row: Transition, contract: CommonTrainingContract) -> float:
    return _validate_transition(row, _validation_context(contract))


def _validate_transition(row: Transition, expected: _TransitionValidation) -> float:
    record, elapsed = _validate_transition_record(row, expected)
    _validate_tensor(row.observation, expected.feature_dim)
    if not row.terminated:
        _validate_tensor(row.next_observation, expected.feature_dim)
    _validate_feature_digests(
        record, tensor_digest(row.observation), tensor_digest(row.next_observation)
    )
    reward = _validate_transition_reward(row, expected, record, elapsed)
    if not math.isfinite(torch.tensor(reward, dtype=torch.float32).item()):
        raise ValueError("transition reward differs from the shared reward specification")
    return elapsed


def _validate_transition_record(
    row: Transition, expected: _TransitionValidation
) -> tuple[Mapping[str, Any], float]:
    """Validate every scalar/provenance field, independently of tensor storage.

    The public transition validator checks its original tensors. Collation
    instead checks the owned stacks that the returned learner batch consumes.
    """
    record = _mapping(row.info.get(COMMON_INTERVAL_KEY))
    elapsed = _validate_interval(record, expected)
    if any(
        type(getattr(row, name)) is not type(record[name]) or getattr(row, name) != record[name]
        for name in ("action", "terminated", "truncated")
    ):
        raise ValueError("transition action/flags differ from actual interval")
    if row.terminated and row.next_observation is not None:
        raise ValueError("terminal NEXT must be absent")
    return record, elapsed


def _validate_feature_digests(
    record: Mapping[str, Any], observation_sha256: str | None, next_sha256: str | None
) -> None:
    if observation_sha256 != record["observation_sha256"] or (
        next_sha256 != record["next_observation_sha256"]
    ):
        raise ValueError("replay features differ from qualified actual endpoints")


def _validate_transition_reward(  # noqa: PLR0913 -- scalar fields have already been validated
    row: Transition,
    expected: _TransitionValidation,
    record: Mapping[str, Any],
    elapsed: float,
) -> float:
    measured = RewardMeasurement(**record["reward_measurement"])
    expected_reward = expected.reward.score(measured, elapsed)
    actual_reward = finite_number(row.reward, "transition reward")
    if actual_reward != expected_reward:
        raise ValueError("transition reward differs from the shared reward specification")
    return actual_reward
