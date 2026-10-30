"""Experimental source-guided residual controls, distinct from absolute action78.

Decisions and complete event/reward observations remain at 50 ms. A pinned
reference supplies ten-ms feed-forward commands; the policy chooses a residual.
Action39 preserves that reference. No trained weights or simulator snapshots are
loaded. This is a hybrid task, not evidence of an autonomous learned shortcut.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from trackmaniarl.core.spec import EvaluationMapSpec
from trackmaniarl.experiments.evaluation import EvaluationResult
from trackmaniarl.trackmania.evaluation import TrackmaniaEvaluator, _EpisodeOutcome
from trackmaniarl.trackmania.geometry import BoundaryGeometry
from trackmaniarl.trackmania.native_headless._runtime.npz_limits import validate_npz_size
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
from trackmaniarl.trackmania.reward_types import RewardResult

CONTROL_SCHEMA = "tmrl.shortcut-residual78-reference10ms-decision50ms/v1"
REFERENCE_ACTION = 39


def residual_tick_action(reference: int, correction: int, offset: int) -> int:
    """13 additive steering bins x six explicit longitudinal modes.

    Modes: coast, brake, first-tick brake, inherit, throttle, throttle+tap.
    Steering additions clip on the unchanged native 13-bin command table.
    """
    if any(type(v) is not int or not 0 <= v < 78 for v in (reference, correction)):
        raise ValueError("reference and correction must be integer action78 indices")
    if type(offset) is not int or not 0 <= offset < 5:
        raise ValueError("one of five physical tick offsets required")
    steer = min(12, max(0, reference // 6 + correction // 6 - 6))
    mode = correction % 6
    if mode == 3:
        longitudinal = reference % 6
    else:
        longitudinal = {0: 0, 1: 1, 2: int(offset == 0), 4: 3, 5: 3 + int(offset == 0)}[mode]
    return steer * 6 + longitudinal


class ReferenceControls:
    """Literal commands bound to the same map binary as the factory geometry."""

    def __init__(self, binding: Mapping[str, Any], geometry: BoundaryGeometry) -> None:
        if binding.get("after_reference") != "neutral-base-until-episode-end/v1":
            raise ValueError("explicit post-reference neutral command policy required")
        path = Path(binding["path"])
        if not path.is_absolute() or path.resolve(strict=True) != path:
            raise ValueError("reference controls require a canonical absolute path")
        if path.stat().st_size > 16 * 1024**2:
            raise ValueError("reference controls exceed 16 MiB")
        with path.open("rb") as source:
            raw = source.read(16 * 1024**2 + 1)
        if len(raw) > 16 * 1024**2:
            raise ValueError("reference controls exceed 16 MiB")
        if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
            raise ValueError("reference controls SHA-256 differs")
        validate_npz_size(raw, max_uncompressed=16 * 1024**2)
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            if (
                str(data["schema"].item()) != CONTROL_SCHEMA
                or str(data["map_uid"].item()) != geometry.map_uid
                or str(data["map_sha256"].item()) != geometry.map_sha256
            ):
                raise ValueError("reference controls schema/map binding differs")
            times, actions = data["race_time_ms"], data["native_action_ids"]
            self.first_reader_index = int(data["first_reader_index"].item())
            if (
                times.dtype.kind not in "iu"
                or actions.dtype.kind not in "iu"
                or times.ndim != 1
                or len(times) < 2
                or actions.shape != times.shape
                or np.any(times > np.iinfo(np.int64).max)
                or not np.array_equal(
                    times.astype(np.int64), np.arange(len(times), dtype=np.int64) * 10
                )
                or np.any((actions < 0) | (actions >= 78))
                or self.first_reader_index < 0
            ):
                raise ValueError("literal consecutive 10ms action78 reference required")
            self.actions = tuple(int(action) for action in actions)

    def at(self, reader_index: int) -> int:
        index = reader_index - self.first_reader_index
        if 0 <= index < len(self.actions):
            return self.actions[index]
        if index < 0:
            raise NativeLifecycleError("native input precedes the reference control domain")
        # This is an explicitly configured COMMAND after the expert tape ends,
        # not an extrapolated expert sample or fabricated observation. Corrections
        # still select steering/throttle/braking; all physics must be observed.
        # The base provider and reward retain the full 150-second episode limit.
        return 36


class ReferenceResidualProvider(NativeFullQrProvider):
    def __init__(
        self,
        settings: NativeFullQrProviderSettings,
        events: FullQrEventSource,
        controls: ReferenceControls,
    ) -> None:
        super().__init__(settings, events)
        self.controls = controls
        self._program: tuple[int, ...] = ()
        self.reference_ticks_in_plan = 0

    def advance(self, action: int) -> FullQrStep | FullQrTerminalStep:
        if type(action) is not int or not 0 <= action < 78:
            raise ValueError("one source-guided residual action78 is required")
        if self._session is None or self._before is None or self._closed:
            raise NativeLifecycleError("an active reference residual reset is required")
        if self._physics_delivery()[0] != 1:
            raise NativeLifecycleError(
                "reference residual controls require per-tick event coverage"
            )
        try:
            first = self._before["currentRecordIndex"]
            expected = self.controls.first_reader_index + (self._before["nativeT"] - 2200) // 10 + 1
            if first != expected:
                raise NativeLifecycleError("reference input index and actual native clock disagree")
            self.reference_ticks_in_plan = sum(
                0 <= first + offset - self.controls.first_reader_index < len(self.controls.actions)
                for offset in range(5)
            )
            self._program = tuple(
                residual_tick_action(self.controls.at(first + offset), action, offset)
                for offset in range(5)
            )
            return super().advance(action)
        except BaseException as error:
            self._fail(error)
            raise

    def _decision_steering(self, action: int) -> float:
        # This task's issued steering is explicitly the five-command mean;
        # historical absolute-action behavior is unchanged in the base provider.
        return sum(_issued_steering(command) for command in self._program) / 5

    def _control_for_tick(self, action: int, offset: int) -> int:
        return self._program[offset]


class SourceGuidedShortcutFactory(FullQrEnvironmentFactory):
    """Separate task class path makes the action meaning part of the run identity."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        if config.get("control_schema") != CONTROL_SCHEMA or not config.get("reward_reference"):
            raise ValueError(
                "explicit residual control schema and shortcut reward reference required"
            )
        super().__init__(config)
        self._controls = ReferenceControls(
            config["reference_controls"], BoundaryGeometry(self._geometry_path)
        )

    def create(
        self, *, seed: int, evaluation_map: EvaluationMapSpec | None = None
    ) -> FullQrEnvironment:
        if evaluation_map is not None:
            if (
                evaluation_map.expected_map_uid != self._map_uid
                or evaluation_map.geometry_path is None
                or evaluation_map.geometry_path.resolve() != self._geometry_path.resolve()
            ):
                raise ValueError("source-guided evaluation requires the exact pinned map/geometry")
            BoundaryGeometry(self._geometry_path).validate_map(evaluation_map.map_path)
        return super().create(seed=seed)

    def _create_provider(
        self, settings: NativeFullQrProviderSettings, events: FullQrEventSource
    ) -> NativeFullQrProvider:
        return ReferenceResidualProvider(settings, events, self._controls)

    def _create_environment(
        self,
        provider: NativeFullQrProvider,
        pipeline: NativeObservationPipeline,
        settings: FullQrEnvironmentSettings,
    ) -> FullQrEnvironment:
        return SourceGuidedShortcutEnvironment(provider, pipeline, settings)


class SourceGuidedShortcutEnvironment(FullQrEnvironment):
    """Expose hybrid controls and real Goal time separately from sampled physics."""

    def _record_step_info(
        self, step: FullQrStep | FullQrTerminalStep, result: RewardResult
    ) -> dict[str, Any]:
        info = super()._record_step_info(step, result)
        provider = self.provider
        if not isinstance(provider, ReferenceResidualProvider):
            raise NativeLifecycleError("source-guided environment requires its own provider")
        info.update(
            control_schema=CONTROL_SCHEMA,
            reference_guided=True,
            native_tick_actions=list(provider._program),
            reference_ticks_in_plan=provider.reference_ticks_in_plan,
        )
        if result.reason == "finished":
            finish = getattr(provider.events, "_finish", None)
            if finish is None:
                raise NativeLifecycleError("finished shortcut requires an actual Goal reader")
            goals = [fact for fact in finish.reader.waypoints if fact.endpoint_requested]
            if not goals:
                raise NativeLifecycleError("finished shortcut has no actual Goal fact")
            info["actual_goal_race_time_ms"] = goals[-1].event_native_t - 2200
        return info


class SourceGuidedShortcutEvaluator(TrackmaniaEvaluator):
    """Standard SDK evaluation with the actual Goal clock for this hybrid task."""

    def _episode_result(self, outcome: _EpisodeOutcome) -> EvaluationResult:
        result = super()._episode_result(outcome)
        if not result.finished:
            return result
        environment = outcome.request.environment
        if not isinstance(environment, SourceGuidedShortcutEnvironment):
            raise NativeLifecycleError("shortcut evaluation requires the source-guided environment")
        provider = environment.provider
        finish = (
            getattr(provider.events, "_finish", None)
            if isinstance(provider, ReferenceResidualProvider)
            else None
        )
        if finish is None:
            raise NativeLifecycleError("shortcut evaluation requires its actual Goal reader")
        goals = [fact for fact in finish.reader.waypoints if fact.endpoint_requested]
        if not goals:
            raise NativeLifecycleError("shortcut evaluation has no actual Goal fact")
        return replace(
            result,
            finish_time_s=(goals[-1].event_native_t - 2200) / 1000,
            finish_time_source="native_goal_event",
        )
