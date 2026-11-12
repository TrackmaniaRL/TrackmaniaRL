"""Deterministic evaluation episodes run by distributed actors."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from time import monotonic
from typing import Any, cast

import trackmaniarl.distributed.actor_watchdog as actor_watchdog
from trackmaniarl.core.contracts import PolicyMode, ReplicablePolicy
from trackmaniarl.core.environment_errors import EnvironmentPausedError
from trackmaniarl.core.pytree import policy_observation_snapshot, tree_snapshot
from trackmaniarl.distributed.actor_collection import reset_pipeline, reset_policy
from trackmaniarl.distributed.actor_metrics import EpisodeMetrics
from trackmaniarl.distributed.actor_protocols import CollectionRuntime
from trackmaniarl.distributed.actor_requests import (
    EnvironmentReset,
    EvaluationEpisodeRequest,
    SpoolRequest,
    TelemetryFailure,
)
from trackmaniarl.distributed.evaluation_assets import evaluation_assets_digest

logger = logging.getLogger(__name__)


class _EvaluationStoppedError(Exception):
    """An operator stopped the actor before the suite could finish."""


@dataclass(slots=True)
class EvaluationPlan:
    runtime: CollectionRuntime
    environment: Any
    pipeline: Any
    policy: ReplicablePolicy
    version: int


def evaluate(runtime: CollectionRuntime, environment: Any, pipeline: Any) -> None:
    suite = runtime.spec.evaluation
    trials = 1 if suite is None else suite.trials_per_map
    policy, version = runtime._evaluation_policy()
    _verify_assigned_assets(runtime)
    plan = EvaluationPlan(runtime, environment, pipeline, policy, version)
    if suite is not None and len(getattr(suite, "maps", ())) > 1:
        # Check again at the actual evaluation boundary: custom runtimes may
        # supply a policy by another path. Never switch before it is isolated.
        _isolated_policy(policy)
    try:
        summaries = (
            _multi_map_trials(plan, suite)
            if suite is not None and len(getattr(suite, "maps", ())) > 1
            else _evaluation_trials(plan, trials)
        )
    except _EvaluationStoppedError:
        logger.info(
            "Actor %s stopped during evaluation; leaving assignment pending", runtime.actor_id
        )
        return
    _verify_assigned_assets(runtime)
    assignment_id = getattr(runtime, "_active_evaluation_assignment_id", "")
    if assignment_id:
        summaries = [
            {**summary, "evaluation_assignment_id": assignment_id} for summary in summaries
        ]
    assets = getattr(runtime, "_active_evaluation_assets_sha256", "")
    if assets:
        summaries = [{**summary, "evaluation_assets_sha256": assets} for summary in summaries]
    # Identified assignments retain the loaded state from before actor-local
    # exploration settings, warmup and trial inference can mutate it. Unassigned
    # and legacy evaluations preserve their original post-trial export behavior.
    snapshot = (
        runtime._active_evaluation_initial_snapshot
        if assignment_id
        else runtime.codec.encode(dict(policy.export_state()))
    )
    if not snapshot:
        raise RuntimeError("identified evaluation is missing its initial policy snapshot")
    runtime._spool(SpoolRequest([], [], version, summaries, snapshot))


def _multi_map_trials(plan: EvaluationPlan, suite: Any) -> list[dict[str, Any]]:
    """Evaluate each map and restore collection before publishing any result.

    The factory is responsible for refusing restoration after an indeterminate
    switch and for confirming the training map before returning. Exceptions
    propagate out of the actor; they must never permit collection to resume.
    """
    factory = getattr(plan.runtime, "_multi_map_factory", None)
    if factory is None:
        raise RuntimeError("multi-map evaluation requires a verified map-switching factory")
    # Each evaluation map consumes one operation ID and restoration consumes
    # one more. Check the entire suite before staging or leaving training.
    factory.require_switch_capacity(plan.environment, len(suite.maps) + 1)
    summaries: list[dict[str, Any]] = []
    pipelines: list[Any] = []
    restore_required = False
    assets = plan.runtime._active_evaluation_assets_sha256
    with factory.staged_suite(suite, plan.runtime.base_dir, assets) as staged:
        try:
            for map_spec in staged.maps:
                if plan.runtime.stop.is_set():
                    raise _EvaluationStoppedError
                pipeline = factory.evaluation_pipeline(plan.pipeline, map_spec)
                if pipeline is plan.pipeline or any(pipeline is previous for previous in pipelines):
                    raise RuntimeError("multi-map evaluation requires an isolated pipeline per map")
                pipelines.append(pipeline)
                # Entering a context may dispatch a switch and then fail. Mark it
                # before calling the factory so even that path invokes restoration.
                restore_required = True
                with factory.evaluation_map(plan.environment, map_spec) as environment:
                    map_plan = EvaluationPlan(
                        plan.runtime, environment, pipeline, plan.policy, plan.version
                    )
                    results = _evaluation_trials(map_plan, suite.trials_per_map)
                    factory.verify_evaluation_map(plan.environment, map_spec)
                    summaries.extend(
                        {
                            **result,
                            "map_id": map_spec.id,
                            "map_uid": map_spec.expected_map_uid,
                            "trial_index": index,
                        }
                        for index, result in enumerate(results)
                    )
        finally:
            if restore_required:
                factory.restore_training(plan.environment)
    return summaries


def _verify_assigned_assets(runtime: CollectionRuntime) -> None:
    expected = getattr(runtime, "_active_evaluation_assets_sha256", "")
    suite = runtime.spec.evaluation
    if suite is not None and len(getattr(suite, "maps", ())) > 1:
        if not expected:
            raise RuntimeError("multi-map evaluation requires an assigned asset digest")
        if not getattr(runtime, "_active_evaluation_assignment_id", ""):
            raise RuntimeError("multi-map evaluation requires an assignment ID")
    if expected:
        actual = evaluation_assets_digest(suite, runtime.base_dir)
        if actual != expected:
            raise RuntimeError("evaluation assets differ from the assigned suite")


def _evaluation_trials(plan: EvaluationPlan, trials: int) -> list[dict[str, Any]]:
    if plan.runtime.stop.is_set():
        raise _EvaluationStoppedError
    # Prewarm and inference can mutate exported policy state. Reload the
    # assigned snapshot for every trial instead of assuming reset_episode
    # restores the policy's weights or other persistent state.
    prewarm = EvaluationPlan(
        plan.runtime, plan.environment, plan.pipeline, _trial_policy(plan), plan.version
    )
    if not _prewarm_evaluation_policy(prewarm):
        return _evaluation_failures(plan, trials)
    summaries: list[dict[str, Any]] = []
    for trial in range(trials):
        if plan.runtime.stop.is_set():
            raise _EvaluationStoppedError
        request = EvaluationEpisodeRequest(
            plan.environment, plan.pipeline, _trial_policy(plan), plan.version
        )
        summary = plan.runtime._evaluate_episode(request)
        summaries.append(summary)
        if summary["termination"] == "telemetry_error":
            summaries.extend(_evaluation_failures(plan, trials - trial - 1))
            break
    return summaries


def _trial_policy(plan: EvaluationPlan) -> ReplicablePolicy:
    runtime = plan.runtime
    suite = getattr(runtime.spec, "evaluation", None)
    if (
        not getattr(runtime, "_active_evaluation_assignment_id", "")
        or suite is None
        or len(getattr(suite, "maps", ())) <= 1
    ):
        return plan.policy
    snapshot = runtime._active_evaluation_initial_snapshot
    if not snapshot:
        raise RuntimeError("identified evaluation is missing its initial policy snapshot")
    state = runtime.codec.decode(snapshot)
    if not isinstance(state, Mapping):
        raise ValueError("evaluation policy snapshot must decode to a mapping")
    policy = _isolated_policy(plan.policy)
    policy.load_state(state)
    return policy


def _isolated_policy(policy: ReplicablePolicy) -> ReplicablePolicy:
    if not _supports_multimap_policy(policy):
        raise RuntimeError("multi-map evaluation policy or selector is not supported")
    try:
        copy = deepcopy(policy)
    except Exception as error:
        raise RuntimeError("assigned evaluation policy cannot be isolated") from error
    selector = getattr(policy, "action_selector", None)
    if copy is policy or (
        selector is not None and getattr(copy, "action_selector", None) is selector
    ):
        raise RuntimeError("assigned evaluation policy cannot be isolated")
    _validate_model_isolation(policy, copy)
    return copy


def _supports_multimap_policy(policy: ReplicablePolicy) -> bool:
    """Only admit the reviewed, resettable first-party value-policy components."""
    from trackmaniarl.algorithms.value_based.policy import DiscreteValuePolicy
    from trackmaniarl.trackmania.actions import TrackmaniaActionSelector

    if type(policy) is not DiscreteValuePolicy:
        return False
    selector = policy.action_selector
    if selector is not None and type(selector) is not TrackmaniaActionSelector:
        return False
    return all(
        type(module).__module__.startswith(("trackmaniarl.models.", "torch.nn."))
        for module in policy.model.modules()
    )


def _validate_model_isolation(original: ReplicablePolicy, copied: ReplicablePolicy) -> None:
    """Reject shared Torch modules and parameter/buffer storage after deepcopy."""
    import torch

    selector = getattr(original, "action_selector", None)
    if selector is not None and hasattr(selector, "weights"):
        before = selector.weights
        after = copied.action_selector.weights  # type: ignore[attr-defined]
        if (
            not isinstance(before, torch.Tensor)
            or not isinstance(after, torch.Tensor)
            or (
                before.numel()
                and before.untyped_storage().data_ptr() == after.untyped_storage().data_ptr()
            )
        ):
            raise RuntimeError("assigned evaluation policy cannot be isolated")
    source = getattr(original, "model", None)
    target = getattr(copied, "model", None)
    if not isinstance(source, torch.nn.Module) or not isinstance(target, torch.nn.Module):
        raise RuntimeError("assigned evaluation policy cannot be isolated")
    if {id(module) for module in source.modules()} & {id(module) for module in target.modules()}:
        raise RuntimeError("assigned evaluation policy cannot be isolated")

    def storage_ids(model: Any) -> set[tuple[str, int]]:
        tensors = (*model.parameters(), *model.buffers())
        return {
            (str(value.device), value.untyped_storage().data_ptr())
            for value in tensors
            if value.numel() and value.device.type != "meta"
        }

    if storage_ids(source) & storage_ids(target):
        raise RuntimeError("assigned evaluation policy cannot be isolated")


def _evaluation_failures(plan: EvaluationPlan, count: int) -> list[dict[str, Any]]:
    failure = TelemetryFailure(plan.version)
    return [plan.runtime._evaluation_telemetry_failure(failure) for _ in range(count)]


def _prewarm_evaluation_policy(plan: EvaluationPlan) -> bool:
    reset = EnvironmentReset(
        plan.environment,
        1_000_000 + plan.runtime._evaluation_index,
        attempts=1,
        stop_on_failure=False,
    )
    observation = plan.runtime._reset_environment(reset)
    if observation is None:
        if plan.runtime.stop.is_set():
            raise _EvaluationStoppedError
        return False
    if plan.runtime.stop.is_set():
        raise _EvaluationStoppedError
    reset_pipeline(plan.pipeline)
    reset_policy(plan.policy)
    try:
        prepared = plan.pipeline.transform_observation(observation)
        plan.policy.act(policy_observation_snapshot(prepared), PolicyMode.EVALUATION)
    finally:
        reset_pipeline(plan.pipeline)
        reset_policy(plan.policy)
    return True


def evaluation_policy(runtime: CollectionRuntime) -> tuple[ReplicablePolicy, int]:
    with runtime._evaluation_request_lock:
        request = runtime._evaluation_request
        runtime._evaluation_request = None
    if request is None:
        runtime._active_evaluation_assignment_id = ""
        runtime._active_evaluation_assets_sha256 = ""
        runtime._active_evaluation_initial_snapshot = b""
        policy, _, version = runtime._policy()
        return policy, version
    snapshot, version, assignment_id = request[:3]
    runtime._active_evaluation_assets_sha256 = request[3] if len(request) == 4 else ""
    runtime._active_evaluation_assignment_id = assignment_id
    runtime._active_evaluation_initial_snapshot = b""
    return _policy_from_request(runtime, (snapshot, version))


def _policy_from_request(
    runtime: CollectionRuntime, request: tuple[bytes, int]
) -> tuple[ReplicablePolicy, int]:
    snapshot, version = request
    state = runtime.codec.decode(snapshot)
    if not isinstance(state, Mapping):
        raise ValueError("evaluation policy snapshot must decode to a mapping")
    policy = runtime._new_policy()
    suite = runtime.spec.evaluation
    if (
        getattr(runtime, "_active_evaluation_assignment_id", "")
        and suite is not None
        and len(getattr(suite, "maps", ())) > 1
    ):
        policy = _isolated_policy(policy)
    policy.load_state(state)
    if getattr(runtime, "_active_evaluation_assignment_id", ""):
        runtime._active_evaluation_initial_snapshot = runtime.codec.encode(
            dict(policy.export_state())
        )
    return policy, version


@dataclass(slots=True)
class EvaluationContext:
    runtime: CollectionRuntime
    environment: Any
    pipeline: Any
    policy: Any
    version: int
    prepared: Any
    metrics: EpisodeMetrics


@dataclass(slots=True)
class EvaluationStep:
    index: int
    action: Any
    inference_s: float
    observation: Any
    reward: float
    info: Mapping[str, Any]
    stopped: bool
    interrupted: bool


@dataclass(frozen=True, slots=True)
class EvaluatedAction:
    value: Any
    inference_s: float


def evaluate_episode(
    runtime: CollectionRuntime, request: EvaluationEpisodeRequest
) -> dict[str, Any]:
    plan = EvaluationPlan(
        runtime,
        request.environment,
        request.pipeline,
        request.policy,
        request.version,
    )
    return _evaluate_episode(plan)


def _evaluate_episode(plan: EvaluationPlan) -> dict[str, Any]:
    runtime = plan.runtime
    reset = EnvironmentReset(
        plan.environment,
        1_000_000 + runtime._evaluation_index,
        attempts=1,
        stop_on_failure=False,
    )
    observation = runtime._reset_environment(reset)
    if observation is None:
        if runtime.stop.is_set():
            raise _EvaluationStoppedError
        return runtime._evaluation_telemetry_failure(TelemetryFailure(plan.version))
    context = _evaluation_context(plan, observation)
    failure = _run_evaluation_episode(context)
    if failure is not None:
        return failure
    return _evaluation_summary(context)


def _evaluation_context(plan: EvaluationPlan, observation: Any) -> EvaluationContext:
    reset_pipeline(plan.pipeline)
    reset_policy(plan.policy)
    return EvaluationContext(
        plan.runtime,
        plan.environment,
        plan.pipeline,
        plan.policy,
        plan.version,
        plan.pipeline.transform_observation(observation),
        EpisodeMetrics.from_policy(plan.policy),
    )


def _evaluation_summary(
    context: EvaluationContext, *, termination: str | None = None
) -> dict[str, Any]:
    runtime = context.runtime
    transitions = context.metrics.controls.samples
    info = context.metrics.summary_info(0.0, context.version, transitions)
    if termination is not None:
        info["termination_reason"] = termination
    summary = runtime._summary(
        context.metrics.total_reward,
        info,
        transitions,
    )
    summary["deterministic"] = 1.0
    runtime._evaluation_index += 1
    return summary


def _run_evaluation_episode(context: EvaluationContext) -> dict[str, Any] | None:
    for step_index in range(context.runtime.spec.training.max_episode_steps):
        failure, stopped = _take_evaluation_step(context, step_index)
        if failure is not None:
            return failure
        if stopped:
            break
    return None


def _take_evaluation_step(
    context: EvaluationContext, step_index: int
) -> tuple[dict[str, Any] | None, bool]:
    if context.runtime.stop.is_set():
        raise _EvaluationStoppedError
    evaluated = _evaluation_action(context)
    if context.runtime.stop.is_set():
        raise _EvaluationStoppedError
    try:
        step = _evaluation_step(context, step_index, evaluated)
    except EnvironmentPausedError as exc:
        if context.runtime.stop.is_set():
            raise _EvaluationStoppedError from exc
        logger.warning(
            "Actor %s deterministic evaluation capture interrupted (%s); retaining DNF",
            context.runtime.actor_id,
            exc,
        )
        return _evaluation_summary(context, termination="capture_interruption"), True
    except (TimeoutError, ConnectionError) as exc:
        if context.runtime.stop.is_set():
            raise _EvaluationStoppedError from exc
        return _evaluation_interruption(context, exc), True
    _record_evaluation_step(context, step)
    if step.interrupted:
        raise _EvaluationStoppedError
    return None, step.stopped


def _evaluation_action(context: EvaluationContext) -> EvaluatedAction:
    started = monotonic()
    action = context.policy.act(
        policy_observation_snapshot(context.prepared), PolicyMode.EVALUATION
    )
    return EvaluatedAction(action, monotonic() - started)


def _evaluation_step(
    context: EvaluationContext, step_index: int, evaluated: EvaluatedAction
) -> EvaluationStep:
    recorded_action = tree_snapshot(evaluated.value)
    observation, reward, terminated, truncated, info = context.environment.step(evaluated.value)
    actor_watchdog.touch(context.runtime)
    interrupted = bool(not terminated and not truncated and context.runtime.stop.is_set())
    stopped = bool(terminated or truncated or interrupted)
    return EvaluationStep(
        step_index,
        recorded_action,
        evaluated.inference_s,
        observation,
        float(reward),
        cast(Mapping[str, Any], info),
        stopped,
        interrupted,
    )


def _record_evaluation_step(context: EvaluationContext, step: EvaluationStep) -> None:
    metrics = context.metrics
    metrics.record_inference(step.inference_s)
    metrics.record_policy(context.policy, step.index)
    metrics.record_diagnostics(step.action, context.policy, step.info)
    context.prepared = context.pipeline.transform_observation(step.observation)
    metrics.record_reward(step.reward, step.info)


def _evaluation_interruption(
    context: EvaluationContext, exc: TimeoutError | ConnectionError
) -> dict[str, Any]:
    runtime = context.runtime
    logger.warning(
        "Actor %s deterministic evaluation telemetry failed (%s: %s)",
        runtime.actor_id,
        type(exc).__name__,
        exc,
    )
    transitions = context.metrics.controls.samples
    failure = TelemetryFailure(
        context.version,
        transitions,
        context.metrics.total_reward,
        context.metrics.summary_info(0.0, context.version, transitions),
    )
    return runtime._evaluation_telemetry_failure(failure)


def evaluation_telemetry_failure(
    runtime: CollectionRuntime, failure: TelemetryFailure
) -> dict[str, Any]:
    return _telemetry_failure_summary(runtime, failure)


def _telemetry_failure_summary(
    runtime: CollectionRuntime, failure: TelemetryFailure
) -> dict[str, Any]:
    summary = runtime._summary(
        failure.reward,
        {
            **dict(failure.info or {}),
            "termination_reason": "telemetry_error",
            "telemetry_error": 1.0,
            "actor_epsilon": 0.0,
            "policy_version": failure.version,
        },
        failure.transitions,
    )
    summary["deterministic"] = 1.0
    runtime._evaluation_index += 1
    return summary
