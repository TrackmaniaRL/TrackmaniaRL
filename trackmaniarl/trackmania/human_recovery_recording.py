"""Live collection of model-induced incidents followed by human recovery."""

from __future__ import annotations

from time import monotonic
from typing import Any

import numpy as np

from trackmaniarl.core.contracts import PolicyMode
from trackmaniarl.core.pytree import policy_observation_snapshot
from trackmaniarl.trackmania import human_recovery_recording_types as _types
from trackmaniarl.trackmania.actions import (
    build_brake_tap_action_table,
    continuous_control_to_discrete_index,
)
from trackmaniarl.trackmania.demonstration_data import Demonstration, _control
from trackmaniarl.trackmania.demonstration_recording import (
    _advance_demonstration_frame,
    _sampling_plan,
)
from trackmaniarl.trackmania.human_recovery_data import (
    MAXIMUM_OBSERVED_PERTURBATION_BRAKE,
    MINIMUM_OBSERVED_PERTURBATION_GAS,
    MINIMUM_OBSERVED_PERTURBATION_STEER,
    MINIMUM_REQUESTED_DURATION_FRACTION,
    RecoveryDemonstration,
)
from trackmaniarl.trackmania.human_recovery_plans import (
    sample_human_recovery_plan as sample_human_recovery_plan,
)
from trackmaniarl.trackmania.human_recovery_plans import (
    sample_human_recovery_plans as sample_human_recovery_plans,
)
from trackmaniarl.trackmania.human_recovery_recording_types import (
    HumanRecoveryEpisodePlan as HumanRecoveryEpisodePlan,
)
from trackmaniarl.trackmania.human_recovery_recording_types import (
    HumanRecoveryRecordingConfig as HumanRecoveryRecordingConfig,
)
from trackmaniarl.trackmania.human_recovery_recording_types import (
    HumanRecoveryRecordingRequest as HumanRecoveryRecordingRequest,
)
from trackmaniarl.trackmania.human_recovery_recording_types import (
    RecoveryAttemptError as RecoveryAttemptError,
)
from trackmaniarl.trackmania.human_recovery_safety import (
    require_continuous_lap as _require_continuous_lap,
)
from trackmaniarl.trackmania.telemetry import TelemetryFrame


def record_human_recovery_episode(
    request: HumanRecoveryRecordingRequest, plan: HumanRecoveryEpisodePlan
) -> RecoveryDemonstration:
    """Drive deterministically to ``plan``, perturb once, then record the human recovery."""

    state = _start_capture(request)
    actual_progress = _drive_policy_to_target(request, state, plan.target_progress)
    result = _apply_perturbation_and_handover(request, state, plan)
    _record_human_finish(request, state)
    recorded = _types.RecordedRecovery(state, actual_progress, result)
    return _recovery_demonstration(request, plan, recorded)


def _recovery_demonstration(
    request: HumanRecoveryRecordingRequest,
    plan: HumanRecoveryEpisodePlan,
    recorded: _types.RecordedRecovery,
) -> RecoveryDemonstration:
    result = recorded.perturbation
    return RecoveryDemonstration(
        demonstration=_completed_demonstration(request, recorded.state),
        perturbation_start_step=result.start_step,
        perturbation_steps=result.steps,
        takeover_step=result.takeover_step,
        target_progress=plan.target_progress,
        actual_progress=recorded.actual_progress,
        perturbation_control=result.actual_control,
        requested_perturbation_duration_ms=plan.perturbation_duration_ms,
        actual_perturbation_duration_ms=result.actual_duration_ms,
        source_checkpoint_sha256=request.checkpoint_sha256,
    )


def _start_capture(request: HumanRecoveryRecordingRequest) -> _types.CaptureState:
    request.status(
        "Model is driving this lap. Keep hands ready and take over only after TAKE OVER NOW."
    )
    observation, _ = request.environment.reset(seed=0)
    _reset_episode(request.feature_pipeline)
    _reset_episode(request.policy)
    current = _raw_frame(observation)
    _, table = build_brake_tap_action_table()
    return _types.CaptureState(
        current=current,
        frames=[current.copy()],
        actions=[],
        controls=[],
        action_table=table,
        deadline=monotonic() + request.config.max_duration_s,
    )


def _drive_policy_to_target(
    request: HumanRecoveryRecordingRequest, state: _types.CaptureState, target: float
) -> float:
    prepared = request.feature_pipeline.transform_observation(state.current)
    while monotonic() < state.deadline:
        step = _policy_step(request, prepared)
        _require_continuous_lap(request, state.current, step.following)
        _append_transition(state, step.following)
        if step.ended:
            raise RecoveryAttemptError(
                f"model lap ended at {step.progress:.1%} "
                f"before the perturbation target ({step.reason})"
            )
        if step.progress >= target:
            _require_pre_perturbation_context(request, state)
            return min(1.0, max(0.0, step.progress))
        prepared = request.feature_pipeline.transform_observation(step.following)
    raise TimeoutError("model did not reach the recovery target before max_duration_s")


def _policy_step(request: HumanRecoveryRecordingRequest, prepared: Any) -> _types.PolicyStep:
    action = request.policy.act(policy_observation_snapshot(prepared), PolicyMode.EVALUATION)
    observation, _, terminated, truncated, info = request.environment.step(action)
    following = _raw_frame(observation)
    return _types.PolicyStep(
        following=following,
        progress=float(info.get("progress_pct", 0.0)) / 100.0,
        ended=terminated or truncated or bool(following[2]),
        reason=str(info.get("termination_reason", "ended")),
    )


def _require_pre_perturbation_context(
    request: HumanRecoveryRecordingRequest, state: _types.CaptureState
) -> None:
    elapsed_ms = float(state.current[3] - state.frames[0][3])
    minimum_ms = request.config.minimum_context_s * 1_000.0
    if elapsed_ms < minimum_ms:
        raise RecoveryAttemptError(
            f"target was reached after only {elapsed_ms:.0f} ms; "
            f"require {minimum_ms:.0f} ms context"
        )


def _apply_perturbation_and_handover(
    request: HumanRecoveryRecordingRequest,
    state: _types.CaptureState,
    plan: HumanRecoveryEpisodePlan,
) -> _types.PerturbationResult:
    timing = _capture_perturbation(request, state, plan)
    deadline = min(state.deadline, monotonic() + request.config.takeover_timeout_s)
    _await_virtual_release(request, state, deadline)
    capture = _observed_perturbation(state, timing)
    request.status("TAKE OVER NOW: recover with your human controls and finish the lap.")
    takeover_step = _await_human_input(request, state, deadline)
    return _types.PerturbationResult(
        capture.start_step,
        capture.steps,
        takeover_step,
        capture.actual_duration_ms,
        capture.actual_control,
    )


def _capture_perturbation(
    request: HumanRecoveryRecordingRequest,
    state: _types.CaptureState,
    plan: HumanRecoveryEpisodePlan,
) -> _types.PerturbationTiming:
    started_ms = float(state.current[3])
    control = _effective_perturbation_control(request, state, plan)
    _announce_perturbation(request, plan.perturbation_duration_ms, control)
    timing = _types.PerturbationTiming(started_ms, plan.perturbation_duration_ms, control)
    _inject_perturbation(request, state, timing)
    return timing


def _observed_perturbation(
    state: _types.CaptureState,
    timing: _types.PerturbationTiming,
) -> _types.PerturbationCapture:
    search_start = _perturbation_search_start(state, timing.started_ms)
    start, stop = _required_perturbation_run(state, search_start, timing.control)
    actual_duration_ms, actual_control = _observed_perturbation_values(state, start, stop)
    _require_sufficient_perturbation_duration(actual_duration_ms, timing.duration_ms)
    return _types.PerturbationCapture(start, stop - start, actual_duration_ms, actual_control)


def _perturbation_search_start(state: _types.CaptureState, started_ms: float) -> int:
    times_ms = np.asarray(state.frames, dtype=np.float32)[:, 3]
    return int(np.searchsorted(times_ms, started_ms, side="left"))


def _required_perturbation_run(
    state: _types.CaptureState, search_start: int, intended: np.ndarray
) -> tuple[int, int]:
    start, stop = _longest_perturbation_run(state, search_start, intended)
    if start is None or stop is None:
        raise RecoveryAttemptError("telemetry did not confirm the injected perturbation")
    return start, stop


def _observed_perturbation_values(
    state: _types.CaptureState, start: int, stop: int
) -> tuple[float, np.ndarray]:
    times_ms = np.asarray(state.frames, dtype=np.float32)[start : stop + 1, 3].astype(np.float64)
    weights = np.diff(times_ms)
    actual_duration_ms = float(weights.sum())
    controls = np.asarray(state.controls[start:stop], dtype=np.float64)
    actual_control = np.average(controls, axis=0, weights=weights).astype(np.float32)
    return actual_duration_ms, actual_control


def _require_sufficient_perturbation_duration(actual_ms: float, requested_ms: float) -> None:
    minimum_duration_ms = requested_ms * MINIMUM_REQUESTED_DURATION_FRACTION
    if actual_ms + 1.0e-6 < minimum_duration_ms:
        raise RecoveryAttemptError(
            f"telemetry confirmed only {actual_ms:.0f} ms of the requested "
            f"{requested_ms:.0f} ms perturbation"
        )


def _longest_perturbation_run(
    state: _types.CaptureState, search_start: int, intended: np.ndarray
) -> tuple[int | None, int | None]:
    matching = np.asarray(
        [_matches_perturbation(control, intended) for control in state.controls[search_start:]],
        dtype=np.bool_,
    )
    if not matching.any():
        return None, None
    padded = np.concatenate((np.asarray([False]), matching, np.asarray([False])))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    stops = np.flatnonzero(padded[:-1] & ~padded[1:])
    frames = np.asarray(state.frames, dtype=np.float32)
    durations = frames[search_start + stops, 3] - frames[search_start + starts, 3]
    best = int(np.argmax(durations))
    return search_start + int(starts[best]), search_start + int(stops[best])


def _matches_perturbation(observed: np.ndarray, intended: np.ndarray) -> bool:
    directed_steer = float(observed[2]) * float(np.sign(intended[2]))
    return bool(
        MINIMUM_OBSERVED_PERTURBATION_GAS <= float(observed[0]) <= 1.0
        and 0.0 <= float(observed[1]) <= MAXIMUM_OBSERVED_PERTURBATION_BRAKE
        and MINIMUM_OBSERVED_PERTURBATION_STEER <= directed_steer <= 1.0
    )


def _announce_perturbation(
    request: HumanRecoveryRecordingRequest, duration_ms: float, control: np.ndarray
) -> None:
    direction = "right" if control[2] > 0 else "left"
    request.status(f"Injecting {duration_ms:.0f} ms {direction} perturbation.")


def _effective_perturbation_control(
    request: HumanRecoveryRecordingRequest,
    state: _types.CaptureState,
    plan: HumanRecoveryEpisodePlan,
) -> np.ndarray:
    control = np.asarray(plan.perturbation_control, dtype=np.float32).copy()
    current_steer = float(_control(TelemetryFrame(state.current))[2])
    if abs(current_steer) > request.config.human_input_deadzone:
        control[2] = -float(np.sign(current_steer))
    return control


def _inject_perturbation(
    request: HumanRecoveryRecordingRequest,
    state: _types.CaptureState,
    timing: _types.PerturbationTiming,
) -> None:
    controller = request.environment.controller
    controller.apply(timing.control)
    try:
        _record_perturbation_frames(request, state, timing)
    finally:
        controller.apply(np.zeros(3, dtype=np.float32))


def _record_perturbation_frames(
    request: HumanRecoveryRecordingRequest,
    state: _types.CaptureState,
    timing: _types.PerturbationTiming,
) -> None:
    while float(state.current[3]) - timing.started_ms < timing.duration_ms:
        following = _next_frame(request, state)
        _require_unfinished_continuous_lap(request, state, following)
        _append_transition(state, following)


def _await_virtual_release(
    request: HumanRecoveryRecordingRequest, state: _types.CaptureState, deadline: float
) -> None:
    while monotonic() < deadline:
        if _control_is_neutral(request, state.current):
            return
        following = _next_frame(request, state)
        _require_unfinished_continuous_lap(request, state, following)
        _append_transition(state, following)
    raise RecoveryAttemptError("virtual perturbation input did not return to neutral")


def _await_human_input(
    request: HumanRecoveryRecordingRequest, state: _types.CaptureState, deadline: float
) -> int:
    while monotonic() < deadline:
        if not _control_is_neutral(request, state.current):
            return len(state.actions)
        following = _next_frame(request, state)
        _require_unfinished_continuous_lap(request, state, following)
        _append_transition(state, following)
    raise RecoveryAttemptError("no clear human input was observed before takeover timeout")


def _control_is_neutral(request: HumanRecoveryRecordingRequest, frame: np.ndarray) -> bool:
    control = _control(TelemetryFrame(frame))
    return bool(np.max(np.abs(control)) <= request.config.human_input_deadzone)


def _record_human_finish(
    request: HumanRecoveryRecordingRequest, state: _types.CaptureState
) -> None:
    while monotonic() < state.deadline:
        following = _next_frame(request, state)
        _require_continuous_lap(request, state.current, following)
        _append_transition(state, following)
        if bool(following[2]):
            request.status(f"Recovery lap finished in {float(following[3]) / 1_000.0:.3f}s.")
            return
    raise TimeoutError("human recovery did not finish before max_duration_s")


def _next_frame(request: HumanRecoveryRecordingRequest, state: _types.CaptureState) -> np.ndarray:
    if monotonic() >= state.deadline:
        raise TimeoutError("human-recovery lap exceeded max_duration_s")
    config = request.environment.config
    frame = _advance_demonstration_frame(
        request.environment.client,
        TelemetryFrame(state.current),
        _sampling_plan(config, None),
    )
    return frame.values


def _append_transition(state: _types.CaptureState, following: np.ndarray) -> None:
    control = _control(TelemetryFrame(state.current))
    state.actions.append(continuous_control_to_discrete_index(control, state.action_table))
    state.controls.append(control)
    state.frames.append(following.copy())
    state.current = following


def _require_unfinished_continuous_lap(
    request: HumanRecoveryRecordingRequest,
    state: _types.CaptureState,
    following: np.ndarray,
) -> None:
    _require_continuous_lap(request, state.current, following)
    if bool(following[2]):
        raise RecoveryAttemptError("lap finished before a human recovery label was captured")


def _completed_demonstration(
    request: HumanRecoveryRecordingRequest, state: _types.CaptureState
) -> Demonstration:
    environment = request.environment
    return Demonstration(
        map_uid=environment.geometry.map_uid,
        geometry_sha256=environment.geometry.sha256,
        action_repeat_frames=environment.config.action_repeat_frames,
        frames=np.asarray(state.frames, dtype=np.float32),
        actions=np.asarray(state.actions, dtype=np.int64),
        controls=np.asarray(state.controls, dtype=np.float32),
        finish_time_s=float(state.frames[-1][3]) / 1_000.0,
        decision_interval_ms=environment.config.decision_interval_ms,
        control_alignment="frame_start",
    )


def _reset_episode(component: Any) -> None:
    reset = getattr(component, "reset_episode", None)
    if callable(reset):
        reset()


def _raw_frame(observation: Any) -> np.ndarray:
    frame = np.asarray(observation, dtype=np.float32).reshape(-1)
    if frame.shape != (33,) or not np.isfinite(frame).all():
        raise ValueError("human recovery requires a finite raw 33-field telemetry observation")
    return frame.copy()
