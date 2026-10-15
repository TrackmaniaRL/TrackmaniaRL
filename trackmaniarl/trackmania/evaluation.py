"""First-party evaluation of a configured TrackMania environment."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from math import isfinite
from numbers import Real
from pathlib import Path
from statistics import median
from time import perf_counter, sleep
from typing import Any, cast

from trackmaniarl.core.contracts import (
    EnvironmentFactory,
    EvaluatorRuntimeRequest,
    Policy,
    PolicyMode,
)
from trackmaniarl.core.environment_errors import EnvironmentPausedError
from trackmaniarl.core.spec import EvaluationMapSpec, EvaluationSuiteSpec
from trackmaniarl.experiments.evaluation import EvaluationResult, aggregate_results
from trackmaniarl.trackmania.diagnostics import (
    ProgressBinDiagnostics,
    ProgressDiagnosticRecord,
    aggregate_progress_bins,
)
from trackmaniarl.trackmania.evaluation_observability import EvaluationObservability
from trackmaniarl.trackmania.geometry import BoundaryGeometry
from trackmaniarl.trackmania.session import PLUGIN_PROTOCOL_VERSION

_TRIAL_FIELDS = (
    "map_id",
    "map_uid",
    "trial_index",
    "steps",
    "finished",
    "finish_time_s",
    "finish_time_source",
    "termination_reason",
    "crashed",
    "reward",
    "action_latency_ms",
    "action_latency_ms_p95",
    "vision_capture_ms",
    "vision_capture_ms_p95",
    "vision_capture_measurement_count",
    "vision_pairing_delay_ms",
    "vision_pairing_delay_ms_p95",
    "vision_pairing_measurement_count",
    "controller_apply_ms",
    "telemetry_wait_ms",
    "control_brake_tap_fraction",
    "step_race_time_ms_p99",
    "step_race_time_ms_max",
    "step_race_time_measurement_count",
    "step_race_time_measurements_valid",
    "telemetry_skipped_frames_total",
    "telemetry_skipped_frames_mean",
    "telemetry_skipped_frames_max",
    "telemetry_steps_with_skipped_frames_fraction",
    "throughput_fps",
    "progress_pct",
    "telemetry_error",
    "controller_error",
    "progress_bins",
)


class EvaluationCancelledError(RuntimeError):
    """The operator stopped an incomplete suite; retained trials are not a full benchmark."""


@dataclass(frozen=True, slots=True)
class _EpisodeRequest:
    policy: Policy
    map_spec: EvaluationMapSpec
    trial_index: int
    seed: int
    environment: Any


@dataclass(slots=True)
class _EpisodeState:
    observability: EvaluationObservability = field(default_factory=EvaluationObservability)
    reward_sum: float = 0.0
    steps: int = 0
    finished: bool = False
    crashed: bool = False
    finish_time_s: float | None = None
    progress_pct: float = 0.0
    telemetry_error: str | None = None
    termination_reason: str = "unknown"


@dataclass(slots=True)
class _EpisodeContext:
    environment: Any
    diagnostics: ProgressBinDiagnostics
    started: float


@dataclass(frozen=True, slots=True)
class _EpisodeOutcome:
    request: _EpisodeRequest
    context: _EpisodeContext
    state: _EpisodeState


@dataclass(slots=True)
class _EpisodeLoop:
    policy: Policy
    environment: Any
    diagnostics: ProgressBinDiagnostics
    observation: Any
    prepared: Any


@dataclass(frozen=True, slots=True)
class _EvaluatedStep:
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    action: Any
    action_duration_ms: float


def _validate_evaluator_request(request: EvaluatorRuntimeRequest) -> None:
    if not isinstance(request.suite, EvaluationSuiteSpec):
        raise TypeError("TrackmaniaEvaluator requires an EvaluationSuiteSpec")
    if request.environment_factory is None:
        raise ValueError("TrackmaniaEvaluator requires components.environment")
    if request.max_episode_steps < 1:
        raise ValueError("max_episode_steps must be positive")


class TrackmaniaEvaluator:
    """Evaluate a policy over the declared seeds and episode budget."""

    def __init__(self, request: EvaluatorRuntimeRequest) -> None:
        _validate_evaluator_request(request)
        self.suite = cast(EvaluationSuiteSpec, request.suite)
        self.environment_factory = cast(EnvironmentFactory, request.environment_factory)
        self.feature_pipeline = request.feature_pipeline
        self.max_episode_steps = request.max_episode_steps
        self.run_dir = Path(request.run_dir) if request.run_dir is not None else None
        self.checkpoint: str | None = None
        self.checkpoint_sha256: str | None = None
        self._stop_requested: Callable[[], bool] = lambda: False

    def set_stop_requested(self, stop_requested: Callable[[], bool]) -> None:
        """Set an optional cancellation check, including while the game is unavailable."""

        self._stop_requested = stop_requested

    def _check_stop_requested(self) -> None:
        if self._stop_requested():
            raise EvaluationCancelledError("Evaluation stopped; the suite is incomplete")

    def set_checkpoint(self, checkpoint: str | Path) -> None:
        """Attach the exact policy checkpoint to the next versioned evaluation artifact."""

        self.checkpoint = str(checkpoint)
        self.checkpoint_sha256 = _checkpoint_sha256(Path(checkpoint))

    def evaluate(self, policy: Policy) -> dict[str, float]:
        """Run the fixed suite and return the standard comparable metric set."""

        results: list[EvaluationResult] = []
        try:
            for result in self._evaluation_results(policy):
                results.append(result)
            self._check_stop_requested()
        except EvaluationCancelledError:
            if self.run_dir is not None:
                metrics = self._evaluation_metrics(results) if results else {}
                self._write_artifact(results, metrics, status="cancelled")
            raise
        metrics = self._evaluation_metrics(results)
        if self.run_dir is not None:
            self._write_artifact(results, metrics)
        return metrics

    def _evaluation_results(self, policy: Policy) -> Iterator[EvaluationResult]:
        for map_spec in self.suite.maps:
            self._check_stop_requested()
            yield from self._map_results(policy, map_spec)

    def _map_results(
        self, policy: Policy, map_spec: EvaluationMapSpec
    ) -> Iterator[EvaluationResult]:
        from trackmaniarl.trackmania.reward_points import RewardPoints

        geometry = (
            RewardPoints(map_spec.reward_points_path, expected_map_uid=map_spec.expected_map_uid)
            if map_spec.reward_points_path is not None
            else BoundaryGeometry(
                map_spec.geometry_path, expected_map_uid=map_spec.expected_map_uid
            )
        )
        geometry.validate_map(map_spec.map_path)
        self._set_evaluation_map(map_spec)
        environment = self._create_environment(map_spec, seed=0)
        try:
            for index in range(self.suite.trials_per_map):
                self._check_stop_requested()
                yield self._evaluate_episode(
                    _EpisodeRequest(policy, map_spec, index, 0, environment)
                )
        finally:
            self._close_environment(environment)

    def _set_evaluation_map(self, map_spec: EvaluationMapSpec) -> None:
        set_evaluation_map = getattr(self.feature_pipeline, "set_evaluation_map", None)
        if callable(set_evaluation_map):
            set_evaluation_map(map_spec)

    def _evaluation_metrics(self, results: list[EvaluationResult]) -> dict[str, float]:
        metrics = dict(aggregate_results(results))
        metrics["eval/median_finish_time_s"] = self._median_finished(results)
        metrics.update(
            {
                key: value
                for key, value in aggregate_progress_bins(
                    result.progress_bins for result in results if result.progress_bins is not None
                ).items()
                if value is not None
            }
        )
        return metrics

    @staticmethod
    def _median_finished(results: list[EvaluationResult]) -> float:
        times = [
            result.finish_time_s for result in results if result.finished and result.finish_time_s
        ]
        return float(median(times)) if times else 0.0

    def _evaluate_episode(self, request: _EpisodeRequest) -> EvaluationResult:
        context = self._episode_context(request)
        state = _EpisodeState()
        try:
            if request.trial_index == 0:
                self._prewarm_policy(request)
            self._write_trial_event(request, "start")
            self._run_episode(request, context, state)
        except EnvironmentPausedError:
            # Retain this interrupted trial as a DNF; never retry away a failure.
            state.termination_reason = "capture_interruption"
            state.finished = False
            state.finish_time_s = None
        except EvaluationCancelledError:
            if state.steps == 0:
                # A cancelled reset has not produced a policy-controlled trial.
                self._write_trial_event(request, "cancelled")
                raise
            state.termination_reason = "operator_interruption"
            state.finished = False
            state.finish_time_s = None
        except (TimeoutError, ConnectionError) as exc:
            state.telemetry_error = f"{type(exc).__name__}: {exc}"
            state.termination_reason = "telemetry_error"
            state.finished = False
            state.finish_time_s = None
        result = self._episode_result(_EpisodeOutcome(request, context, state))
        self._write_trial_event(request, "end", result)
        return result

    def _write_trial_event(
        self, request: _EpisodeRequest, event: str, result: EvaluationResult | None = None
    ) -> None:
        if self.run_dir is None:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with (self.run_dir / "evaluation-timeline.jsonl").open("a", encoding="utf-8") as file:
            json.dump(
                {
                    "utc": datetime.now(UTC).isoformat(),
                    "event": event,
                    "map_id": request.map_spec.id,
                    "trial_index": request.trial_index,
                    "result": None if result is None else asdict(result),
                },
                file,
            )
            file.write("\n")

    def _episode_context(self, request: _EpisodeRequest) -> _EpisodeContext:
        diagnostics = ProgressBinDiagnostics(_policy_action_count(request.policy), bin_count=20)
        return _EpisodeContext(request.environment, diagnostics, perf_counter())

    def _run_episode(
        self, request: _EpisodeRequest, context: _EpisodeContext, state: _EpisodeState
    ) -> None:
        loop = self._start_episode(request, context)
        for _ in range(self.max_episode_steps):
            self._check_stop_requested()
            step = self._take_step(loop)
            if self._record_step(loop, state, step):
                break
        else:
            state.termination_reason = "max_steps"

    def _start_episode(self, request: _EpisodeRequest, context: _EpisodeContext) -> _EpisodeLoop:
        observation, _ = self._reset_available(request)
        # Window availability and reset latency are not policy-controlled driving.
        context.started = perf_counter()
        self._reset_component(self.feature_pipeline)
        self._reset_component(request.policy)
        prepared = self.feature_pipeline.transform_observation(observation)
        return _EpisodeLoop(
            request.policy, context.environment, context.diagnostics, observation, prepared
        )

    def _prewarm_policy(self, request: _EpisodeRequest) -> None:
        observation, _ = self._reset_available(request)
        self._reset_component(self.feature_pipeline)
        self._reset_component(request.policy)
        prepared = self.feature_pipeline.transform_observation(observation)
        policy_observation = (
            observation if getattr(request.policy, "requires_raw_observation", False) else prepared
        )
        try:
            request.policy.act(policy_observation, PolicyMode.EVALUATION)
        finally:
            self._reset_component(self.feature_pipeline)
            self._reset_component(request.policy)

    def _reset_available(self, request: _EpisodeRequest) -> Any:
        while True:
            self._check_stop_requested()
            try:
                return request.environment.reset(seed=request.seed)
            except EnvironmentPausedError:
                sleep(0.5)

    @staticmethod
    def _reset_component(component: Any) -> None:
        reset_episode = getattr(component, "reset_episode", None)
        if callable(reset_episode):
            reset_episode()

    def _take_step(self, loop: _EpisodeLoop) -> _EvaluatedStep:
        observation = (
            loop.observation
            if getattr(loop.policy, "requires_raw_observation", False)
            else loop.prepared
        )
        started = perf_counter()
        action = loop.policy.act(observation, PolicyMode.EVALUATION)
        action_duration_ms = (perf_counter() - started) * 1_000.0
        next_observation, reward, terminated, truncated, info = loop.environment.step(action)
        return _EvaluatedStep(
            next_observation,
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
            action,
            action_duration_ms,
        )

    def _record_step(self, loop: _EpisodeLoop, state: _EpisodeState, step: _EvaluatedStep) -> bool:
        state.observability.record(step.info, step.action_duration_ms)
        record = ProgressDiagnosticRecord(
            float(step.info.get("progress_pct", state.progress_pct)),
            step.action,
            loop.policy,
            (
                None
                if "race_time_ms" in step.info and _race_time_seconds(step.info) is None
                else step.info
            ),
        )
        loop.diagnostics.record(record)
        loop.observation = step.observation
        loop.prepared = self.feature_pipeline.transform_observation(step.observation)
        state.reward_sum += step.reward
        state.steps += 1
        self._record_outcome(state, step.info)
        if (step.terminated or step.truncated) and not step.info.get("termination_reason"):
            state.termination_reason = "truncated" if step.truncated else "terminated"
        return step.terminated or step.truncated

    @staticmethod
    def _record_outcome(state: _EpisodeState, info: dict[str, Any]) -> None:
        termination_reason = str(info.get("termination_reason", ""))
        if termination_reason:
            state.termination_reason = termination_reason
        state.progress_pct = float(info.get("progress_pct", state.progress_pct))
        state.finished = termination_reason == "finished"
        collision = bool(info.get("collision", False)) or bool(
            info.get("collision_detected", False)
        )
        state.crashed = (
            state.crashed
            or collision
            or termination_reason
            in {
                "crashed",
                "off_track",
            }
        )
        if state.finished:
            state.finish_time_s = _race_time_seconds(info)

    def _episode_result(self, outcome: _EpisodeOutcome) -> EvaluationResult:
        context, state = outcome.context, outcome.state
        elapsed_s = perf_counter() - context.started
        result = EvaluationResult(
            finished=state.finished,
            finish_time_s=(state.finish_time_s or elapsed_s) if state.finished else None,
            crashed=state.crashed,
            reward=state.reward_sum,
            action_latency_ms=state.observability.mean_action_latency_ms(state.steps),
            throughput_fps=state.steps / elapsed_s if elapsed_s > 0.0 else 0.0,
            progress_pct=state.progress_pct,
            termination_reason=state.termination_reason,
            finish_time_source=(
                ("race_clock" if state.finish_time_s is not None else "elapsed_fallback")
                if state.finished
                else None
            ),
        )
        return self._annotated_result(result, outcome)

    def _annotated_result(
        self, result: EvaluationResult, outcome: _EpisodeOutcome
    ) -> EvaluationResult:
        request, context, state = outcome.request, outcome.context, outcome.state
        identified = replace(
            result,
            map_id=request.map_spec.id,
            map_uid=request.map_spec.expected_map_uid,
            trial_index=request.trial_index,
            telemetry_error=state.telemetry_error,
            progress_bins=context.diagnostics.summary(),
            steps=state.steps,
        )
        return state.observability.annotated_result(identified, state.steps)

    @staticmethod
    def _close_environment(environment: Any) -> None:
        close = getattr(environment, "close", None)
        if callable(close):
            close()

    def _create_environment(self, map_spec: EvaluationMapSpec, *, seed: int) -> Any:
        factory = cast(Any, self.environment_factory)
        return factory.create(seed=seed, evaluation_map=map_spec)

    def _write_artifact(
        self,
        results: list[EvaluationResult],
        metrics: dict[str, float],
        *,
        status: str = "complete",
    ) -> None:
        assert self.run_dir is not None
        target = self.run_dir / "evaluation.json"
        payload = self._artifact_payload(results, metrics)
        payload["status"] = status
        payload["expected_trials"] = len(self.suite.maps) * self.suite.trials_per_map
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, target)

    def _artifact_payload(
        self, results: list[EvaluationResult], metrics: dict[str, float]
    ) -> dict[str, Any]:
        if (
            self.checkpoint is not None
            and _checkpoint_sha256(Path(self.checkpoint)) != self.checkpoint_sha256
        ):
            raise RuntimeError("Checkpoint changed during evaluation; refusing a false binding")
        return {
            "schema_version": "1",
            "plugin_protocol_version": PLUGIN_PROTOCOL_VERSION,
            "suite": {"name": self.suite.name, "version": self.suite.version},
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "metrics": metrics,
            "trials": [self._trial_payload(result) for result in results],
        }

    @staticmethod
    def _trial_payload(result: EvaluationResult) -> dict[str, Any]:
        return {name: getattr(result, name) for name in _TRIAL_FIELDS}


def _policy_action_count(policy: Policy) -> int:
    model = getattr(policy, "model", None)
    count = getattr(policy, "action_count", getattr(model, "action_count", 78))
    if not isinstance(count, int) or count < 2:
        raise ValueError("TrackMania policy must expose at least two actions")
    return count


def _checkpoint_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _race_time_seconds(info: dict[str, Any]) -> float | None:
    value = info.get("race_time_ms")
    if isinstance(value, Real) and not isinstance(value, bool):
        milliseconds = float(value)
        if isfinite(milliseconds) and milliseconds > 0.0:
            return milliseconds / 1_000.0
    return None
