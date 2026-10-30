"""Autonomous five-tick actions: the policy selects every command program.

There is no reference clock, expert-action sequence or residual controller here.
A static unordered motor vocabulary is part of the action space. Demonstration
order is accessible only to the offline data collector and replay importer.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from trackmaniarl.core.spec import EvaluationMapSpec
from trackmaniarl.experiments.evaluation import EvaluationResult
from trackmaniarl.trackmania.evaluation import TrackmaniaEvaluator, _EpisodeOutcome
from trackmaniarl.trackmania.geometry import BoundaryGeometry
from trackmaniarl.trackmania.native_headless.full_qr.environment import (
    FullQrEnvironment,
    FullQrEnvironmentSettings,
    FullQrStep,
    FullQrTerminalStep,
    NativeObservationPipeline,
)
from trackmaniarl.trackmania.native_headless.full_qr.factory import FullQrEnvironmentFactory
from trackmaniarl.trackmania.native_headless.full_qr.native_provider import (
    FullQrEventSource,
    NativeFullQrProvider,
    NativeFullQrProviderSettings,
    _issued_steering,
)
from trackmaniarl.trackmania.native_headless.protocols import NativeLifecycleError
from trackmaniarl.trackmania.reward_config import RewardConfig
from trackmaniarl.trackmania.reward_types import RewardResult

CONTROL_SCHEMA = "tmrl.autonomous-macro-action50ms-native10ms/v1"


def actual_goal_reward(
    result: RewardResult, config: RewardConfig, clocks_ms: tuple[float, int]
) -> RewardResult:
    score_ms, goal_ms = clocks_ms
    correction_s = (score_ms - goal_ms) / 1000
    if result.reason != "finished" or not 0 <= correction_s <= 0.05:
        raise NativeLifecycleError("actual Goal must lie in the sourced final reward interval")
    time_adjustment = correction_s * config.time_penalty_per_second
    attack_adjustment = correction_s * config.time_attack_linear_scale
    return replace(
        result,
        reward=result.reward + time_adjustment + attack_adjustment,
        time_reward=result.time_reward + time_adjustment,
        time_attack_terminal_reward=result.time_attack_terminal_reward + attack_adjustment,
    )


def pinned_arrays(binding: Mapping[str, Any]) -> dict[str, np.ndarray]:
    path = Path(binding["path"])
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError("a canonical absolute data path is required")
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError("task data exceeds64MiB")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise ValueError("task data SHA-256 mismatch")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


class MacroCatalog:
    """Five literal native commands per action; no timeline is stored."""

    def __init__(self, binding: Mapping[str, Any]) -> None:
        data = pinned_arrays(binding)
        if set(data) != {"schema", "programs"} or str(data["schema"].item()) != CONTROL_SCHEMA:
            raise ValueError("only a static macro vocabulary is allowed in the action asset")
        programs = data["programs"]
        if (
            programs.dtype.kind not in "iu"
            or programs.ndim != 2
            or programs.shape[1] != 5
            or not 78 <= len(programs) <= 1024
            or np.any((programs < 0) | (programs >= 78))
        ):
            raise ValueError("78..1024 five-tick native action78 programs required")
        rows = [tuple(int(x) for x in row) for row in programs]
        if rows != sorted(set(rows)):
            raise ValueError("macro vocabulary must be deduplicated and lexicographically ordered")
        self.programs = tuple(rows)
        self.sha256 = str(binding["sha256"])

    def program(self, action: int) -> tuple[int, ...]:
        if type(action) is not int or not 0 <= action < len(self.programs):
            raise ValueError("policy action is outside the pinned macro vocabulary")
        return self.programs[action]


class AutonomousMacroProvider(NativeFullQrProvider):
    def __init__(
        self,
        settings: NativeFullQrProviderSettings,
        events: FullQrEventSource,
        catalog: MacroCatalog,
    ) -> None:
        super().__init__(settings, events)
        self.catalog = catalog
        self.program: tuple[int, ...] = ()
        self.policy_action: int | None = None

    def advance(self, action: int) -> FullQrStep | FullQrTerminalStep:
        program = self.catalog.program(action)
        if (
            self._closed
            or self._session is None
            or self._before is None
            or self._collector is None
            or self._identity is None
        ):
            raise NativeLifecycleError("an active autonomous macro reset is required")
        try:
            if self._physics_delivery()[0] != 1:
                raise NativeLifecycleError("macro controls require complete per-tick sources")
            self.program, self.policy_action = program, action
            return self._advance_decision(action)
        except BaseException as error:
            self._fail(error)
            raise

    def _decision_steering(self, action: int) -> float:
        return sum(_issued_steering(command) for command in self.program) / 5

    def _control_for_tick(self, action: int, offset: int) -> int:
        return self.program[offset]


class AutonomousMacroEnvironment(FullQrEnvironment):
    def _observe_and_reward(
        self, step: FullQrStep | FullQrTerminalStep
    ) -> tuple[dict[str, Any], RewardResult]:
        observation, result = super()._observe_and_reward(step)
        if result.reason != "finished":
            return observation, result
        score_ms = getattr(step.frame, "score_race_time_ms", step.frame.channels[0])
        goal_ms = actual_goal_ms(self)
        return observation, actual_goal_reward(result, self.reward_config, (score_ms, goal_ms))

    def _validate_action(self, action: int) -> None:
        provider = self.provider
        if not isinstance(provider, AutonomousMacroProvider):
            raise NativeLifecycleError("autonomous macro provider required")
        provider.catalog.program(action)

    def _record_step_info(
        self, step: FullQrStep | FullQrTerminalStep, result: RewardResult
    ) -> dict[str, Any]:
        info = super()._record_step_info(step, result)
        provider = self.provider
        if not isinstance(provider, AutonomousMacroProvider):
            raise NativeLifecycleError("autonomous macro provider required")
        info.update(
            control_schema=CONTROL_SCHEMA,
            reference_guided=False,
            policy_macro_action=provider.policy_action,
            planned_native_actions=list(provider.program),
        )
        if result.reason == "finished":
            info["actual_goal_race_time_ms"] = actual_goal_ms(self)
        return info


def actual_goal_ms(environment: AutonomousMacroEnvironment) -> int:
    provider = environment.provider
    finish = (
        getattr(provider.events, "_finish", None)
        if isinstance(provider, AutonomousMacroProvider)
        else None
    )
    if finish is None:
        raise NativeLifecycleError("actual native Goal reader required")
    goals = [fact for fact in finish.reader.waypoints if fact.endpoint_requested]
    if not goals:
        raise NativeLifecycleError("actual Goal fact required")
    return int(goals[-1].event_native_t - 2200)


class AutonomousMacroShortcutFactory(FullQrEnvironmentFactory):
    def __init__(self, config: Mapping[str, Any]) -> None:
        if config.get("control_schema") != CONTROL_SCHEMA or "reference_controls" in config:
            raise ValueError(
                "autonomous macro schema required; reference-control playback forbidden"
            )
        super().__init__(config)
        if (
            config.get("reward_finish_clock") != "native_goal_event/v1"
            or self._reward.time_attack_target_s is None
            or any(
                (
                    self._reward.time_attack_bonus_scale,
                    self._reward.pace_reward_scale,
                    self._reward.projected_velocity_scale,
                    self._reward.projected_speed_bonus_scale,
                    self._reward.potential_progress_weight,
                    self._reward.steering_delta_penalty,
                )
            )
        ):
            raise ValueError("explicit actual-Goal linear time reward profile required")
        self.catalog = MacroCatalog(config["macro_catalog"])
        self.demonstration_binding = config.get("training_demonstration")

    def create(
        self, *, seed: int, evaluation_map: EvaluationMapSpec | None = None
    ) -> FullQrEnvironment:
        if evaluation_map is not None:
            if (
                evaluation_map.expected_map_uid != self._map_uid
                or evaluation_map.geometry_path is None
                or evaluation_map.geometry_path.resolve() != self._geometry_path.resolve()
            ):
                raise ValueError("autonomous evaluation requires the pinned map and geometry")
            BoundaryGeometry(self._geometry_path).validate_map(evaluation_map.map_path)
        return super().create(seed=seed)

    def _create_provider(
        self, settings: NativeFullQrProviderSettings, events: FullQrEventSource
    ) -> NativeFullQrProvider:
        return AutonomousMacroProvider(settings, events, self.catalog)

    def _create_environment(
        self,
        provider: NativeFullQrProvider,
        pipeline: NativeObservationPipeline,
        settings: FullQrEnvironmentSettings,
    ) -> FullQrEnvironment:
        return AutonomousMacroEnvironment(provider, pipeline, settings)

    def load_demonstration(self, path: Path, pipeline: Any) -> list[Any]:
        from trackmaniarl.experiments.shortcut_demonstrations import load_native_demonstration

        if self.demonstration_binding is None or Path(self.demonstration_binding["path"]) != Path(
            path
        ):
            raise ValueError("only the explicitly pinned training demonstration may be imported")
        reward_sha = hashlib.sha256(
            json.dumps(asdict(self._reward), sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        identity = {
            "map_uid": self._map_uid,
            "map_sha256": BoundaryGeometry(self._geometry_path).map_sha256,
            "reward_sha256": reward_sha,
            "reward_clock": "native_goal_event/v1",
        }
        return load_native_demonstration(self.demonstration_binding, self.catalog, identity)


class AutonomousMacroEvaluator(TrackmaniaEvaluator):
    def _episode_result(self, outcome: _EpisodeOutcome) -> EvaluationResult:
        result = super()._episode_result(outcome)
        if not result.finished:
            return result
        if not isinstance(outcome.request.environment, AutonomousMacroEnvironment):
            raise NativeLifecycleError("autonomous macro evaluation environment required")
        return replace(
            result,
            finish_time_s=actual_goal_ms(outcome.request.environment) / 1000,
            finish_time_source="native_goal_event",
        )
