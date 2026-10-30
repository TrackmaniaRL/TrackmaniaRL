"""Import real native demonstration transitions without inventing ghost channels."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from trackmaniarl.algorithms.value_based.objectives import (
    DemonstrationCrossEntropyObjective,
    ValueObjectiveContext,
)
from trackmaniarl.core.data import Transition
from trackmaniarl.core.replay.demo_sampler import DemoMixSampler
from trackmaniarl.core.replay.prioritized import PrioritizedSampler
from trackmaniarl.experiments.graph_iqn_v5 import BoundaryGraphFeaturePipelineV5
from trackmaniarl.experiments.shortcut_macro import MacroCatalog, pinned_arrays

DEMO_SCHEMA = "tmrl.native-macro-demonstration/v1"


def load_native_demonstration(
    binding: Mapping[str, Any], catalog: MacroCatalog, identity: Mapping[str, str]
) -> list[Transition]:
    data = pinned_arrays(binding)
    if (
        str(data["schema"].item()) != DEMO_SCHEMA
        or any(str(data[key].item()) != expected for key, expected in identity.items())
        or str(data["catalog_sha256"].item()) != catalog.sha256
    ):
        raise ValueError("native demonstration schema/map/action-vocabulary differs")
    actions, rewards, terminated = data["actions"], data["rewards"], data["terminated"]
    count = len(actions)
    if (
        actions.dtype.kind not in "iu"
        or actions.shape != (count,)
        or count < 256
        or rewards.shape != (count,)
        or not np.isfinite(rewards).all()
        or terminated.dtype.kind != "b"
        or terminated.shape != (count,)
        or not terminated[-1]
        or terminated[:-1].any()
        or np.any((actions < 0) | (actions >= len(catalog.programs)))
    ):
        raise ValueError("complete finite native action/reward/terminal sequence required")
    times = data["observed_physics_ms"]
    if (
        times.dtype.kind not in "iu"
        or times.shape != (count + 1,)
        or times[0] != 0
        or np.any(np.diff(times) != 50)
    ):
        raise ValueError("demonstration requires actual consecutive fifty-ms observations")
    names = json.loads(str(data["observation_keys"].item()))
    if set(names) != {"physics", "track", "context", "recovery"} or len(names) != 4:
        raise ValueError("complete full V5 observation leaves required")
    observations = []
    for name in names:
        values = data["observation/" + name]
        if values.shape[0] != count + 1 or not np.isfinite(values).all():
            raise ValueError("all native demonstration observations must be finite and complete")
    infos = json.loads(str(data["infos_json"].item()))
    if len(infos) != count or infos[-1].get("termination_reason") != "finished":
        raise ValueError("native demonstration requires the sourced Finish result")
    goal_ms = infos[-1].get("actual_goal_race_time_ms")
    if (
        type(goal_ms) is not int
        or not int(times[-2]) < goal_ms <= int(times[-1])
        or any(info.get("reference_guided") is not False for info in infos)
    ):
        raise ValueError("actual final-interval Goal required; residual-controller data forbidden")
    for index in range(count + 1):
        observations.append(
            {
                name: torch.from_numpy(np.asarray(data["observation/" + name][index]).copy())
                for name in names
            }
        )
        BoundaryGraphFeaturePipelineV5._validate_v5_tensors(observations[-1])
    return [
        Transition(
            observation=observations[i],
            action=int(actions[i]),
            reward=float(rewards[i]),
            next_observation=observations[i + 1],
            terminated=bool(terminated[i]),
            truncated=False,
            info={**infos[i], "is_demo": True, "sampling/projected_lap_time_s": goal_ms / 1000},
            episode_id="native-expert-" + str(binding["sha256"]),
            step=i,
        )
        for i in range(count)
    ]


class OfflineThenMixedSampler:
    """Expert-only offline fitting, then uniform replay plus an expert mixture.

    The coordinator's offline requests have transition_count0. Online requests
    carry actual collected-transition counts and begin after replay warmup.
    The SDK replay store already preserves demos on FIFO eviction.
    """

    supports_sequence_sampling = False

    def __init__(
        self,
        pipeline: Any,
        seed: int = 0,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        config = {} if config is None else config
        online_demo_fraction = float(config.get("online_demo_fraction", 0.25))
        expert_lap_time_s = float(config.get("expert_lap_time_s", 17.715))
        # Replay stores sampling pace as float32. Compare at that same precision;
        # this does not round or relax the independent integer native Goal target.
        expert_lap_time_s = float(np.float32(expert_lap_time_s))
        self.offline = DemoMixSampler(pipeline, seed=seed, min_demo_fraction=1, max_demo_fraction=1)
        self.online = PrioritizedSampler(
            pipeline,
            seed=seed,
            alpha=0,
            beta=0,
            expert_demo_time_s=expert_lap_time_s,
            expert_fraction=online_demo_fraction,
            expert_fraction_final=float(config.get("online_demo_fraction_final", 0.05)),
            expert_fraction_anneal_transitions=int(
                config.get("expert_fraction_anneal_transitions", 500_000)
            ),
        )

    def sample(self, store: Any, request: Any) -> Any:
        batch = (self.offline if request.transition_count == 0 else self.online).sample(
            store, request
        )
        return replace(
            batch,
            metadata={**batch.metadata, "collected_transition_count": request.transition_count},
        )

    def update_priorities(self, priorities: Any) -> None:
        # alpha0 is uniform within each pool; TD-error feedback cannot alter it.
        return None

    def state_dict(self) -> dict[str, Any]:
        return {"offline": self.offline.state_dict(), "online": self.online.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.offline.load_state_dict(state["offline"])
        self.online.load_state_dict(state["online"])


class AnnealedExpertObjective:
    """Use imitation to bootstrap, then release its pull toward the old record."""

    requires_all_actions = True

    def __init__(self, release_transitions: int = 250_000) -> None:
        if type(release_transitions) is not int or release_transitions < 1:
            raise ValueError("positive expert-guidance release horizon required")
        self.release_transitions = release_transitions
        self.objective = DemonstrationCrossEntropyObjective(weight=1)

    def loss(self, context: ValueObjectiveContext) -> torch.Tensor | None:
        count = context.metadata.get("collected_transition_count")
        if type(count) is not int or count < 0:
            raise ValueError("actual collected-transition counter required for expert schedule")
        weight = 1.0 if count == 0 else 0.1 * max(0.0, 1 - count / self.release_transitions)
        if not weight:
            return None
        value = self.objective.loss(context)
        return None if value is None else weight * value


class RetainedExpertObjective:
    """Keep class-balanced imitation during TD learning, without a time-based release.

    Weights describe training labels only. They are never consulted by actors.
    A finite coefficient balances the summed QR loss; it does not mask actions
    or force the deployed policy to reproduce the demonstrator.
    """

    requires_all_actions = True

    def __init__(self, class_weights: Sequence[float], weight: float = 64.0) -> None:
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("positive finite imitation weight required")
        if len(class_weights) < 2 or any(not math.isfinite(v) or v < 0 for v in class_weights):
            raise ValueError("finite nonnegative action class weights required")
        if not any(class_weights):
            raise ValueError("at least one demonstrated action required")
        self.class_weights = tuple(class_weights)
        self.weight = weight

    def loss(self, context: ValueObjectiveContext) -> torch.Tensor:
        values = context.expected_values
        if values.shape[-1] != len(self.class_weights):
            raise ValueError("class weights must match the action vocabulary")
        flags = context.metadata.get("demo_flags")
        if flags is None:
            raise ValueError("retained imitation requires demo_flags")
        demo = torch.as_tensor(flags, device=values.device, dtype=torch.bool)
        if demo.numel() != context.actions.numel():
            raise ValueError("demonstration flags must match training positions")
        valid = context.valid & demo.reshape(context.actions.shape)
        weights = values.new_tensor(self.class_weights)
        if context.action_mask is not None:
            values = values.masked_fill(~context.action_mask, torch.finfo(values.dtype).min)
        losses = torch.nn.functional.cross_entropy(
            values.reshape(-1, values.shape[-1]),
            context.actions.reshape(-1),
            weight=weights,
            reduction="none",
        ).reshape(context.actions.shape)
        quality = torch.as_tensor(
            context.metadata.get("demonstration_weights", torch.ones_like(losses)),
            device=values.device,
            dtype=losses.dtype,
        )
        if quality.numel() != losses.numel() or not bool(
            torch.isfinite(quality).all() & ((quality >= 0) & (quality <= 1)).all()
        ):
            raise ValueError("one finite demonstration weight in [0,1] per position required")
        denominator = (weights[context.actions] * valid).sum().clamp_min(1e-12)
        return self.weight * (losses * valid * quality.reshape(losses.shape)).sum() / denominator
