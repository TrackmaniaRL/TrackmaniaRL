"""Native fixed QR-DQN bridge with lazy ragged next-state targets.

Explicit learner construction/setup builds the model. The main learner owns losses,
optimization, priorities and target synchronization; this replaces its eager
next-state encoder path for qualified terminal observations with no features.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from trackmaniarl.algorithms.value_based import DiscreteValueLearner
from trackmaniarl.algorithms.value_based.updater import ValueUpdater, _BatchStep, _ForwardStep
from trackmaniarl.core.data import PriorityUpdate, TrainingBatch
from trackmaniarl.models.composite import BatchLayout, CompositeValueModel
from trackmaniarl.models.contracts import ValuePhase
from trackmaniarl.models.heads import FixedQuantileHead
from trackmaniarl.models.strategies import FixedQuantileStrategy
from trackmaniarl.models.temporal import IdentityTemporalCore
from trackmaniarl.trackmania.native_headless.training_sampler import CONTRACT


def _tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    return value


def validate_batch(batch: TrainingBatch) -> torch.Tensor:
    if (
        batch.metadata.get("native_ragged_contract") != CONTRACT
        or batch.metadata.get("n_step") != 1
        or batch.metadata.get("sequence_length") != 1
        or batch.metadata.get("gamma", 0.995) != 0.995
    ):
        raise ValueError("explicit native44 ragged n-step-one contract required")
    observations = _tensor(batch.observations, "observations")
    next_observations = _tensor(batch.next_observations, "next_observations")
    rewards = _tensor(batch.rewards, "rewards")
    discounts = _tensor(batch.bootstrap_discounts, "bootstrap_discounts")
    actions = _tensor(batch.actions, "actions")
    terminated = _tensor(batch.terminated, "terminated")
    truncated = _tensor(batch.truncated, "truncated")
    if observations.ndim != 2 or observations.shape[1] != 44 or not observations.shape[0]:
        raise ValueError("nonempty actual current [B,44] required")
    size = observations.shape[0]
    floats = (observations, next_observations, rewards, discounts)
    if any(value.dtype != torch.float32 for value in floats):
        raise ValueError("float32 observations/returns required")
    if (
        actions.dtype != torch.long
        or terminated.dtype != torch.bool
        or truncated.dtype != torch.bool
    ):
        raise ValueError("long actions and bool flags required")
    vectors = (actions, rewards, discounts, terminated, truncated)
    if any(tuple(value.shape) != (size,) for value in vectors):
        raise ValueError("dense current/reward/flag/discount row mismatch")
    if bool((terminated & truncated).any()) or bool(((actions < 0) | (actions >= 78)).any()):
        raise ValueError("contradictory flags or out-of-range action78")
    if not isinstance(batch.data, Mapping):
        raise ValueError("explicit ragged indices required")
    indices = _tensor(batch.data.get("bootstrap_row_indices"), "bootstrap_row_indices")
    expected = (~terminated).nonzero(as_tuple=True)[0]
    if indices.dtype != torch.long or not torch.equal(indices, expected):
        raise ValueError("bootstrap indices must exactly select nonterminal rows")
    if tuple(next_observations.shape) != (len(indices), 44):
        raise ValueError("actual ragged next subset mismatch")
    durations = batch.metadata.get("delta_native")
    if (
        not isinstance(durations, list)
        or len(durations) != size
        or any(type(dt) is not int or dt not in (10, 20, 30, 40, 50) for dt in durations)
    ):
        raise ValueError("qualified native durations required")
    wanted = torch.tensor(
        [0.0 if bool(terminated[i]) else 0.995 ** (dt / 50) for i, dt in enumerate(durations)],
        dtype=discounts.dtype,
        device=discounts.device,
    )
    if not torch.equal(discounts, wanted):
        raise ValueError("duration/terminal discount mismatch")
    if any(not bool(torch.isfinite(value).all()) for value in floats):
        raise ValueError("finite actual observations/returns required")
    if (
        len(batch.transition_ids) != size
        or len(set(batch.transition_ids)) != size
        or any(type(identity) is not int or identity < 0 for identity in batch.transition_ids)
    ):
        raise ValueError("unique row identities required")
    return indices


def lazy_targets(
    batch: TrainingBatch, evaluate_actual_next: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """Return [B,32]; reward-only terminal rows are targets, never observations."""
    indices = validate_batch(batch)
    rewards = _tensor(batch.rewards, "rewards")
    discounts = _tensor(batch.bootstrap_discounts, "discounts")
    targets = rewards[:, None].expand(-1, 32).clone()
    if indices.numel():
        values = evaluate_actual_next(_tensor(batch.next_observations, "actual next"))
        if tuple(values.shape) != (len(indices), 32) or not bool(torch.isfinite(values).all()):
            raise ValueError("finite bootstrap quantile subset required")
        targets[indices] = rewards[indices, None] + discounts[indices, None] * values
    return targets


class RaggedUpdater(ValueUpdater):
    """Narrow fixed44/78/32 identity-temporal extension to the main updater."""

    def _forward_step(self, step: _BatchStep) -> _ForwardStep:
        learner, batch = self.learner, step.batch
        validate_batch(batch)
        model = learner.model
        if (
            not isinstance(model, CompositeValueModel)
            or not isinstance(model.head, FixedQuantileHead)
            or not isinstance(model.strategy, FixedQuantileStrategy)
            or not isinstance(model.temporal, IdentityTemporalCore)
            or model.action_count != 78
            or getattr(model.encoder, "input_dim", None) != 44
            or learner.burn_in != 0
            or learner.value_rescaling
        ):
            raise ValueError("extension requires native44/78 fixed QR-DQN identity temporal")
        if model.head.quantile_count != 32 or model.strategy.quantile_count != 32:
            raise ValueError("fixed head/strategy32 required")
        with learner.autocast():
            current = model.encode_sequence(batch.observations, BatchLayout.FRAMES, 0).float()
        support = model.support(current, ValuePhase.TRAIN)
        predictions = model.distribution_for_actions(
            current, support.detached_points(), step.actions
        )

        def evaluate(actual_next: torch.Tensor) -> torch.Tensor:
            with torch.no_grad(), learner.autocast():
                online = model.encode_sequence(actual_next, BatchLayout.FRAMES, 0).float()
                target = learner.target_model.encode_sequence(actual_next, BatchLayout.FRAMES, 0)
            with torch.no_grad():
                actions = self._next_actions(learner, online)
                target_support = learner.target_model.support(target.float(), ValuePhase.TARGET)
                return learner.target_model.distribution_for_actions(
                    target.float(), target_support, actions
                )[:, 0].float()

        with torch.no_grad():
            targets = lazy_targets(batch, evaluate)[:, None, :]
            # Fixed midpoint support is state independent; CURRENT rows supply B.
            target_support = learner.target_model.support(current.detach(), ValuePhase.TARGET)
        rewards, discounts = step.view.returns_and_discounts(step.positions)
        return _ForwardStep(
            step, current, support, predictions, target_support, targets, rewards, discounts
        )


class NativeRaggedLearner(DiscreteValueLearner):
    """Existing value learner with the explicit native44 ragged target contract."""

    def update(self, batch: TrainingBatch) -> tuple[Mapping[str, float], PriorityUpdate]:
        validate_batch(batch)
        return RaggedUpdater(self).update(batch)

    def state_dict(self) -> Mapping[str, Any]:
        return {**super().state_dict(), "native_example_contract": CONTRACT}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("native_example_contract") != CONTRACT:
            raise ValueError("no implicit old checkpoint migration to native ragged example")
        primary = dict(state)
        primary.pop("native_example_contract")
        super().load_state_dict(primary)

    def load_policy_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("native_example_contract") != CONTRACT:
            raise ValueError("native ragged policy checkpoint contract mismatch")
        primary = dict(state)
        primary.pop("native_example_contract")
        super().load_policy_state_dict(primary)
