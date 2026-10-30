"""Pinned multi-lap demonstrations and a bounded imitation release for time attack."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from trackmaniarl.algorithms.value_based.objectives import ValueObjectiveContext
from trackmaniarl.experiments.shortcut_demonstrations import (
    OfflineThenMixedSampler,
    RetainedExpertObjective,
    load_native_demonstration,
)
from trackmaniarl.experiments.shortcut_macro import AutonomousMacroShortcutFactory
from trackmaniarl.trackmania.geometry import BoundaryGeometry


class MultiDemonstrationShortcutFactory(AutonomousMacroShortcutFactory):
    """Import complete native laps from a hash-pinned manifest, only in the learner."""

    def load_demonstration(self, path: Path, pipeline: Any) -> list[Any]:
        binding = self.demonstration_binding
        if binding is None or Path(binding["path"]) != path:
            raise ValueError("only the pinned multi-demonstration manifest may be imported")
        if not path.is_absolute() or path.resolve(strict=True) != path:
            raise ValueError("canonical manifest path required")
        with path.open("rb") as source:
            raw = source.read(1024**2 + 1)
        if len(raw) > 1024**2 or hashlib.sha256(raw).hexdigest() != binding["sha256"]:
            raise ValueError("demonstration manifest size or SHA-256 mismatch")
        manifest = json.loads(raw)
        if manifest.get("schema") != "tmrl.native-multi-demonstration/v1":
            raise ValueError("native multi-demonstration manifest required")
        episodes = manifest.get("episodes", [])
        if not 2 <= len(episodes) <= 64:
            raise ValueError("two to sixty-four independently qualified native laps required")
        hashes = [episode["sha256"] for episode in episodes]
        if len(set(hashes)) != len(hashes):
            raise ValueError("duplicate demonstration episodes are forbidden")
        identity = {
            "map_uid": self._map_uid,
            "map_sha256": BoundaryGeometry(self._geometry_path).map_sha256,
            "reward_sha256": hashlib.sha256(
                json.dumps(asdict(self._reward), sort_keys=True, allow_nan=False).encode()
            ).hexdigest(),
            "reward_clock": "native_goal_event/v1",
        }
        transitions = []
        for episode in episodes:
            lap = load_native_demonstration(episode, self.catalog, identity)
            finish_ms = lap[-1].info["actual_goal_race_time_ms"]
            if type(episode.get("native_goal_ms")) is not int or not (
                0 < finish_ms == episode["native_goal_ms"] < 20_000
            ):
                raise ValueError("manifest must match the actual native Goal below twenty seconds")
            transitions.extend(lap)
        return transitions


class PaceAwareMixedSampler(OfflineThenMixedSampler):
    """Keep all qualified demos for TD; give faster laps stronger imitation.

    The public pace accessor follows protected demos when replay relocates them
    on FIFO eviction. No observation copy or inference-time lookup is needed.
    """

    def __init__(self, pipeline: Any, seed: int = 0, config: Any = None) -> None:
        super().__init__(pipeline, seed=seed, config=config)
        config = {} if config is None else config
        self.best_demo_time_s = float(config.get("best_demo_time_s", 17.715))
        if not math.isfinite(self.best_demo_time_s) or not 0 < self.best_demo_time_s < 20:
            raise ValueError(
                "positive finite native reference finish below twenty seconds required"
            )

    def sample(self, store: Any, request: Any) -> Any:
        if request.sequence_length != 1:
            raise ValueError("pace-weighted demonstration sampling requires single-step positions")
        with store.sampling_transaction():
            batch = super().sample(store, request)
            return replace(
                batch,
                metadata={
                    **batch.metadata,
                    "demonstration_weights": self._pace_weights(store, batch),
                },
            )

    def _pace_weights(self, store: Any, batch: Any) -> tuple[float, ...]:
        weights = []
        for transition_id, demo in zip(
            batch.transition_ids, batch.metadata["demo_flags"], strict=True
        ):
            if not demo:
                weights.append(0.0)
                continue
            pace = store.sampling_pace_s(transition_id)
            if not math.isfinite(pace) or not 0 < pace < 20:
                raise ValueError("complete native demonstrations below twenty seconds required")
            weights.append(math.exp(min(0, self.best_demo_time_s - pace) / 0.75))
        return tuple(weights)


class ReleasingExpertObjective:
    """Release a finite retention loss gradually; retain a nonzero recovery floor.

    The schedule uses actual collected transitions and is fixed before launch.
    Native evaluations and the campaign quality fuse decide whether the resulting
    policy is useful. This objective never selects or masks deployed actions.
    """

    requires_all_actions = True

    def __init__(
        self,
        class_weights: Sequence[float],
        config: Mapping[str, Any] | None = None,
    ) -> None:
        config = {} if config is None else config
        initial_weight = float(config.get("initial_weight", 128))
        final_weight = float(config.get("final_weight", 16))
        hold_transitions = config.get("hold_transitions", 100_000)
        release_transitions = config.get("release_transitions", 400_000)
        if not all(math.isfinite(v) and v > 0 for v in (initial_weight, final_weight)):
            raise ValueError("positive finite imitation weights required")
        if final_weight > initial_weight:
            raise ValueError("final imitation weight must not exceed its initial weight")
        if type(hold_transitions) is not int or hold_transitions < 0:
            raise ValueError("nonnegative integer retention hold required")
        if type(release_transitions) is not int or release_transitions <= 0:
            raise ValueError("positive integer release horizon required")
        self.objective = RetainedExpertObjective(class_weights, weight=initial_weight)
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.hold_transitions = hold_transitions
        self.release_transitions = release_transitions

    def weight_at(self, transitions: int) -> float:
        if type(transitions) is not int or transitions < 0:
            raise ValueError("actual nonnegative collected-transition count required")
        elapsed = max(0.0, (transitions - self.hold_transitions) / self.release_transitions)
        fraction = min(1.0, elapsed)
        return self.initial_weight + fraction * (self.final_weight - self.initial_weight)

    def loss(self, context: ValueObjectiveContext) -> torch.Tensor:
        count = context.metadata.get("collected_transition_count")
        if type(count) is not int:
            raise ValueError("actual nonnegative collected-transition count required")
        return self.objective.loss(context) * (self.weight_at(count) / self.initial_weight)
