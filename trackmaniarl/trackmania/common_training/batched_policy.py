"""Opt-in common6 inference batching for the pinned stateless CPU policy.

Matrix kernels can round differently for different batch sizes. Close greedy
choices are re-evaluated with the serial batch shape, but that guard is a
heuristic, not a universal error bound or a claim of bit-exact Q values. The
exploration random draws retain the existing policy's sorted actor-ID order.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from trackmaniarl.algorithms.value_based.policy import DiscreteValuePolicy
from trackmaniarl.core.contracts import PolicyMode
from trackmaniarl.models.composite import CompositeValueModel
from trackmaniarl.models.contracts import RiskDistortion, ValuePhase
from trackmaniarl.models.encoders import MlpSensorEncoder
from trackmaniarl.models.heads import FixedQuantileHead
from trackmaniarl.models.strategies import FixedQuantileStrategy
from trackmaniarl.models.temporal import IdentityTemporalCore

BATCH_NUMERICS = "cpu-float32/batched-gemm/near-tie-serial-recheck/v1"
_TIE_ABSOLUTE = 1e-5
_TIE_RELATIVE = 1e-4


class BatchedCommonPolicy:
    """Share one policy copy and batch observations from one barrier round.

    The caller owns policy weight refresh and episode reset. Identity temporal
    state is empty for every actor; a recurrent core or action selector cannot
    use this adapter. ``act_many`` accepts 1..8 CPU float32 common6 tensors.
    """

    def __init__(self, policy: DiscreteValuePolicy) -> None:
        self.policy = policy
        self.last_serial_rechecks: tuple[int, ...] = ()
        self.serial_recheck_count = 0
        self.batch_count = 0
        self._require_profile()

    def _require_profile(self) -> None:
        policy, model = self.policy, self.policy.model
        if (
            type(policy) is not DiscreteValuePolicy
            or type(model) is not CompositeValueModel
            or type(model.encoder) is not MlpSensorEncoder
            or model.encoder.input_dim != 6
            or model.encoder.output_dim != 256
            or type(model.temporal) is not IdentityTemporalCore
            or model.temporal.input_dim != 256
            or type(model.head) is not FixedQuantileHead
            or model.head.feature_dim != 256
            or model.head.action_count != 78
            or model.head.quantile_count != 32
            or model.head.value is not None
            or type(model.strategy) is not FixedQuantileStrategy
            or model.strategy.quantile_count != 32
            or model.action_count != 78
            or policy.device.type != "cpu"
            or any(module.training for module in model.modules())
            or any(p.dtype != torch.float32 or p.device.type != "cpu" for p in model.parameters())
        ):
            raise ValueError("batched inference requires the pinned common6 fixed32 CPU policy")
        self._require_controls()

    def _require_controls(self) -> None:
        policy = self.policy
        epsilon = policy.exploration_epsilon
        choices = policy.policy_action_ids
        if (
            policy.action_selector is not None
            or policy._state != ()
            or policy.online_risk.distortion is not RiskDistortion.NEUTRAL
            or policy.evaluation_risk.distortion is not RiskDistortion.NEUTRAL
        ):
            raise ValueError("stateless identity policy without selector and neutral risk required")
        if isinstance(epsilon, bool) or not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError("finite exploration epsilon between zero and one required")
        if choices is not None and (
            not isinstance(choices, tuple)
            or not choices
            or any(type(action) is not int or not 0 <= action < 78 for action in choices)
            or len(set(choices)) != len(choices)
        ):
            raise ValueError("nonempty unique discrete78 action IDs required")

    def _q_values(self, observations: torch.Tensor, mode: PolicyMode) -> torch.Tensor:
        model = self.policy.model
        features, state = model.policy_step(observations, ())
        if state != ():
            raise ValueError("batched identity policy produced temporal state")
        support = model.support(features, ValuePhase.EVALUATE)
        risk = (
            self.policy.evaluation_risk
            if mode is PolicyMode.EVALUATION
            else self.policy.online_risk
        )
        values = model.expected_all_actions(features, support, risk)
        if tuple(values.shape) != (len(observations), 78) or not bool(torch.isfinite(values).all()):
            raise ValueError("finite batched discrete78 Q values required")
        return self.policy._mask(values)

    def act_many(
        self,
        observations: Mapping[int, torch.Tensor],
        mode: PolicyMode = PolicyMode.ONLINE,
    ) -> dict[int, int]:
        """Validate the whole round before forward passes or random draws."""
        self._require_controls()
        if not isinstance(mode, PolicyMode):
            raise ValueError("explicit policy mode required")
        if not 1 <= len(observations) <= 8 or any(
            type(actor) is not int or not 0 <= actor < 8 for actor in observations
        ):
            raise ValueError("one to eight distinct actor IDs in 0..7 required")
        actors = sorted(observations)
        rows = [observations[actor] for actor in actors]
        if any(
            not isinstance(row, torch.Tensor)
            or row.dtype != torch.float32
            or row.device.type != "cpu"
            or tuple(row.shape) != (6,)
            for row in rows
        ):
            raise ValueError("CPU float32 common6 observation tensors required")
        stacked = torch.stack(rows)
        if not bool(torch.isfinite(stacked).all()):
            raise ValueError("finite common6 observations required")
        rechecked: list[int] = []
        with torch.no_grad():
            values = self._q_values(stacked, mode)
            # A single allowed action needs no numerical tie guard. Masked -inf
            # entries are intentional and never enter scale or gap arithmetic.
            choices = self.policy.policy_action_ids
            allowed = values if choices is None else values[:, list(choices)]
            if len(actors) > 1 and allowed.shape[1] > 1:
                largest = allowed.topk(2, dim=-1).values
                scale = allowed.abs().amax(dim=-1)
                ambiguous = largest[:, 0] - largest[:, 1] <= (_TIE_ABSOLUTE + _TIE_RELATIVE * scale)
                for index in ambiguous.nonzero(as_tuple=True)[0].tolist():
                    values[index] = self._q_values(stacked[index : index + 1], mode)[0]
                    rechecked.append(actors[index])
            # Keep the old policy's exact rand/randint interleaving per actor.
            # A vectorized epsilon mask would change the saved Torch RNG stream.
            actions = {
                actor: int(self.policy._select_action(values[index : index + 1], mode).item())
                for index, actor in enumerate(actors)
            }
        if any(type(action) is not int or not 0 <= action < 78 for action in actions.values()):
            raise ValueError("policy action outside discrete78")
        self.last_serial_rechecks = tuple(rechecked)
        self.serial_recheck_count += len(rechecked)
        self.batch_count += 1
        return actions
