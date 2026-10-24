"""A narrow common8 ragged endpoint extension of the existing QR-DQN updater.

Losses, optimization, priorities and target synchronization remain owned by the
SDK's ValueUpdater. Only encoding of absent terminal NEXT rows is bypassed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from trackmaniarl.algorithms.execution import TorchExecutionConfig
from trackmaniarl.algorithms.value_based import DiscreteValueLearner
from trackmaniarl.algorithms.value_based.updater import ValueUpdater, _BatchStep, _ForwardStep
from trackmaniarl.core.data import PriorityUpdate, TrainingBatch
from trackmaniarl.models.composite import BatchLayout, CompositeModules, CompositeValueModel
from trackmaniarl.models.contracts import ValuePhase
from trackmaniarl.models.encoders import MlpSensorEncoder
from trackmaniarl.models.heads import FixedQuantileHead
from trackmaniarl.models.strategies import FixedQuantileStrategy
from trackmaniarl.models.temporal import IdentityTemporalCore
from trackmaniarl.trackmania.common_training.contract import (
    SCHEMA,
    CommonTrainingContract,
    RewardMeasurement,
    finite_number,
)


def _tensor(value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError("explicit common8 training tensors required")
    return value


def validate_batch(batch: TrainingBatch, contract: CommonTrainingContract) -> torch.Tensor:
    if (
        batch.metadata.get("common_ragged_contract") != SCHEMA
        or batch.metadata.get("contract_sha256") != contract.digest
        or batch.metadata.get("n_step") != 1
        or batch.metadata.get("sequence_length") != 1
        or batch.metadata.get("gamma") != contract.gamma
    ):
        raise ValueError("explicit common8 duration/ragged contract required")
    observations, next_observations = _tensor(batch.observations), _tensor(batch.next_observations)
    rewards, discounts = _tensor(batch.rewards), _tensor(batch.bootstrap_discounts)
    actions, terminated, truncated = map(
        _tensor, (batch.actions, batch.terminated, batch.truncated)
    )
    if (
        observations.ndim != 2
        or observations.shape[1] != contract.feature_dim
        or not observations.shape[0]
    ):
        raise ValueError("nonempty actual [B,common_dim] observations required")
    size = observations.shape[0]
    floats = (observations, next_observations, rewards, discounts)
    if any(v.dtype != torch.float32 or not bool(torch.isfinite(v).all()) for v in floats):
        raise ValueError("finite float32 common observations/rewards/discounts required")
    if (
        actions.dtype != torch.long
        or terminated.dtype != torch.bool
        or truncated.dtype != torch.bool
    ):
        raise ValueError("long actions and explicit bool flags required")
    if any(tuple(v.shape) != (size,) for v in (actions, rewards, discounts, terminated, truncated)):
        raise ValueError("common training row shape mismatch")
    if bool((terminated & truncated).any()) or bool(((actions < 0) | (actions >= 78)).any()):
        raise ValueError("contradictory flags or action outside discrete78")
    if not isinstance(batch.data, Mapping):
        raise ValueError("explicit ragged common8 bootstrap indices required")
    indices = _tensor(batch.data.get("bootstrap_row_indices"))
    if indices.dtype != torch.long or not torch.equal(
        indices, (~terminated).nonzero(as_tuple=True)[0]
    ):
        raise ValueError("ragged bootstrap indices must select all nonterminal rows")
    if tuple(next_observations.shape) != (len(indices), contract.feature_dim):
        raise ValueError("actual ragged common8 NEXT subset required")
    durations, measurements = (
        batch.metadata.get("elapsed_seconds"),
        batch.metadata.get("reward_measurements"),
    )
    if not isinstance(durations, list) or len(durations) != size:
        raise ValueError("actual timestamp-derived durations required")
    if not isinstance(measurements, list) or len(measurements) != size:
        raise ValueError("actual shared reward measurements required")
    for elapsed in durations:
        if finite_number(elapsed, "batch duration") <= 0:
            raise ValueError("positive actual batch duration required")
    expected_discounts = torch.tensor(
        [
            0.0
            if bool(terminated[i])
            else contract.gamma ** (elapsed / contract.discount_reference_seconds)
            for i, elapsed in enumerate(durations)
        ],
        dtype=torch.float32,
        device=discounts.device,
    )
    if not torch.equal(discounts, expected_discounts):
        raise ValueError("duration/terminal bootstrap discount mismatch")
    expected_rewards = torch.tensor(
        [
            contract.reward.score(RewardMeasurement(**measurement), elapsed)
            for measurement, elapsed in zip(measurements, durations, strict=True)
        ],
        dtype=torch.float32,
        device=rewards.device,
    )
    if not torch.equal(rewards, expected_rewards):
        raise ValueError("batch reward differs from the common specification")
    if (
        len(batch.transition_ids) != size
        or len(set(batch.transition_ids)) != size
        or any(type(identity) is not int or identity < 0 for identity in batch.transition_ids)
    ):
        raise ValueError("unique nonnegative common replay IDs required")
    return indices


def lazy_targets(
    batch: TrainingBatch,
    contract: CommonTrainingContract,
    evaluate_actual_next: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    indices = validate_batch(batch, contract)
    return _validated_targets(batch, indices, evaluate_actual_next)


def _validated_targets(
    batch: TrainingBatch,
    indices: torch.Tensor,
    evaluate_actual_next: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Build unchanged targets after the caller's full batch validation.

    The learner validates before any forward pass. Its batch transfer preserves
    values and moves the validated index tensor with the other batch tensors.
    Public ``lazy_targets`` retains its independent validation boundary.
    """
    rewards, discounts = _tensor(batch.rewards), _tensor(batch.bootstrap_discounts)
    targets = rewards[:, None].expand(-1, 32).clone()
    if indices.numel():
        values = evaluate_actual_next(_tensor(batch.next_observations))
        if tuple(values.shape) != (len(indices), 32) or not bool(torch.isfinite(values).all()):
            raise ValueError("actual finite NEXT fixed32 quantiles required")
        targets[indices] = rewards[indices, None] + discounts[indices, None] * values
    return targets


class _CommonUpdater(ValueUpdater):
    def _forward_step(self, step: _BatchStep) -> _ForwardStep:
        learner = self.learner
        assert isinstance(learner, CommonLearner)
        model = learner.model
        if (
            not isinstance(model, CompositeValueModel)
            or not isinstance(model.head, FixedQuantileHead)
            or not isinstance(model.strategy, FixedQuantileStrategy)
            or not isinstance(model.temporal, IdentityTemporalCore)
            or getattr(model.encoder, "input_dim", None) != learner.common_contract.feature_dim
            or model.action_count != 78
            or model.head.quantile_count != 32
            or model.strategy.quantile_count != 32
            or learner.burn_in != 0
            or learner.value_rescaling
        ):
            raise ValueError("shared learner requires common8/78 fixed32 identity temporal model")
        with learner.autocast():
            current = model.encode_sequence(step.batch.observations, BatchLayout.FRAMES, 0).float()
        support = model.support(current, ValuePhase.TRAIN)
        predictions = model.distribution_for_actions(
            current, support.detached_points(), step.actions
        )

        def evaluate(actual_next: torch.Tensor) -> torch.Tensor:
            with torch.no_grad(), learner.autocast():
                online = model.encode_sequence(actual_next, BatchLayout.FRAMES, 0).float()
                target = learner.target_model.encode_sequence(
                    actual_next, BatchLayout.FRAMES, 0
                ).float()
            with torch.no_grad():
                actions = self._next_actions(learner, online)
                target_support = learner.target_model.support(target, ValuePhase.TARGET)
                return learner.target_model.distribution_for_actions(
                    target, target_support, actions
                )[:, 0].float()

        with torch.no_grad():
            indices = _tensor(step.batch.data["bootstrap_row_indices"])
            targets = _validated_targets(step.batch, indices, evaluate)[:, None, :]
            target_support = learner.target_model.support(current.detach(), ValuePhase.TARGET)
        rewards, discounts = step.view.returns_and_discounts(step.positions)
        return _ForwardStep(
            step, current, support, predictions, target_support, targets, rewards, discounts
        )


class CommonLearner(DiscreteValueLearner):
    """The same model and learner settings for both explicit endpoint adapters."""

    def __init__(
        self,
        contract: CommonTrainingContract,
        *,
        seed: int = 1729,
        execution: TorchExecutionConfig | None = None,
    ) -> None:
        if type(seed) is not int:
            raise ValueError("integer shared model initialization seed required")
        self.common_contract = contract
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = CompositeValueModel(
                CompositeModules(
                    MlpSensorEncoder(contract.feature_dim, 256),
                    IdentityTemporalCore(256),
                    FixedQuantileHead(
                        {"feature_dim": 256, "action_count": 78, "quantile_count": 32}
                    ),
                    FixedQuantileStrategy(32),
                )
            )
        super().__init__(
            model,
            learning_rate=0.0001,
            target_update_interval=1000,
            target_tau=0.0,
            gradient_clip_norm=10.0,
            exploration_epsilon=0.1,
            online_quantile_distortion="neutral",
            evaluation_quantile_distortion="neutral",
            value_rescaling=False,
            burn_in=0,
            seed=seed,
            execution=execution
            or TorchExecutionConfig(device="cpu", precision="float32", torch_threads=2),
        )

    def update(self, batch: TrainingBatch) -> tuple[Mapping[str, float], PriorityUpdate]:
        validate_batch(batch, self.common_contract)
        return _CommonUpdater(self).update(batch)

    def state_dict(self) -> Mapping[str, Any]:
        return {**super().state_dict(), "common_contract": self.common_contract.envelope()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.common_contract.require_checkpoint(state.get("common_contract", {}))
        primary = dict(state)
        primary.pop("common_contract")
        super().load_state_dict(primary)

    def load_policy_state_dict(self, state: Mapping[str, Any]) -> None:
        self.common_contract.require_checkpoint(state.get("common_contract", {}))
        primary = dict(state)
        primary.pop("common_contract")
        super().load_policy_state_dict(primary)


def make_learner(
    contract: CommonTrainingContract,
    *,
    seed: int = 1729,
    execution: TorchExecutionConfig | None = None,
) -> CommonLearner:
    learner = CommonLearner(contract, seed=seed, execution=execution)
    learner.setup({"seed": seed})
    return learner
