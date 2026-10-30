"""Fresh native episodes with exact command receipts and honest prefix provenance."""

import json
import time
from pathlib import Path

import numpy as np

from tools.record_rl1168.closure import native_closure_clean
from tools.record_rl1168.features import RouteFeatures, shaped_reward
from tools.record_rl1168.model import ACTIONS


class InvalidProgress(ValueError):
    """A measured progress jump failed the episode-level physical guard."""


def episode(
    config,
    folder,
    geometry,
    act,
    *,
    deadline,
    training=False,
    prefix_controls=(),
    on_transition=None,
    stop_file=None,
):
    if not training and (len(prefix_controls) or on_transition is not None):
        raise ValueError("evaluation forbids teacher prefixes and training callbacks")
    from environment import FallEnvironment

    folder = Path(folder)
    env = None
    report = {
        "status": "FAILED",
        "actual_goal_ms": None,
        "prefix_ticks": len(prefix_controls),
        "full_start": len(prefix_controls) == 0,
        "training": training,
        "policy_steps": 0,
    }
    started = time.monotonic()
    decisions = []
    try:
        env = FallEnvironment(config, folder)
        route = RouteFeatures(geometry)
        raw = env.reset()
        state, progress, lateral = route.observe(raw, reset=True)
        stalled = 0
        furthest = progress
        for step in range(3000):
            if time.monotonic() >= deadline or (stop_file and Path(stop_file).exists()):
                report["status"] = "STOPPED_BUDGET_OR_OPERATOR"
                break
            prefix = step < len(prefix_controls)
            action = None if prefix else act(state)
            command = prefix_controls[step] if prefix else ACTIONS[action]
            following, _, goal, truncated, info = env.step(command)
            next_state, next_progress, lateral = route.observe(following)
            if not np.isclose((following[0] - raw[0]) * 30000, 10, atol=0.005):
                raise RuntimeError("native10ms observation discontinuity")
            stalled = 0 if next_progress > furthest + 0.005 else stalled + 1
            furthest = max(furthest, next_progress)
            failed = not goal and (truncated or stalled >= 200 or lateral > 20)
            dt = 0.01
            if goal:
                dt = (info["actual_goal_ms"] - round(float(raw[0]) * 30000)) / 1000
                if not 0 < dt <= 0.01:
                    raise RuntimeError("Goal outside measured physics interval")
            try:
                reward = shaped_reward(progress, next_progress, dt=dt, goal=goal, failure=failed)
            except ValueError as error:
                # Reject this measured attempt, never invent a reward or a finish.
                # Training retains its exception/discard path. Evaluation records
                # the failure and advances only after the normal closure checks.
                if str(error) != "unphysical progress jump; discard transition":
                    raise
                if training:
                    raise InvalidProgress(str(error)) from error
                report.update(
                    status="INVALID_PROGRESS",
                    actual_goal_ms=None,
                    error=repr(error),
                    race_ms=info["race_ms"],
                    previous_progress_m=progress,
                    rejected_progress_m=next_progress,
                    measured_dt=dt,
                )
                break
            if not prefix:
                report["policy_steps"] += 1
                decisions.append(
                    {
                        "race_ms": info["race_ms"],
                        "action": action,
                        "reward": reward,
                        "goal": goal,
                        "failure": failed,
                        "progress_m": next_progress,
                        "lateral_m": lateral,
                    }
                )
                if on_transition:
                    on_transition(state, action, reward, next_state, goal or failed)
            state, raw, progress = next_state, following, next_progress
            if goal or failed:
                report.update(
                    status="NATIVE_GOAL" if goal else "DRIVING_FAILURE",
                    actual_goal_ms=info["actual_goal_ms"],
                    race_ms=info["race_ms"],
                    progress_m=progress,
                    lateral_m=lateral,
                )
                break
        else:
            # Reaching the safety limit is not a terminal physics observation.
            report["status"] = "STEP_LIMIT_INCOMPLETE"
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        if env:
            cleanup_errors = []
            try:
                env.close()
            except Exception as error:
                cleanup_errors.append({"stage": "env.close", "error": repr(error)})
            try:
                close = json.loads((env.settings.trace_dir / "native-supervisor.json").read_text())
                report["closure"] = close
            except Exception as error:
                cleanup_errors.append({"stage": "supervisor", "error": repr(error)})
                close = None
            report["trace_dir"] = str(env.settings.trace_dir)
            report["cleanupErrors"] = cleanup_errors
            report["clean_closure"] = not cleanup_errors and native_closure_clean(close)
        report["elapsed_seconds"] = time.monotonic() - started
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "decisions.json").write_text(json.dumps(decisions))
        (folder / "result.json").write_text(json.dumps(report, indent=2))
        # Closure errors must remain fatal even while propagating InvalidProgress.
        if env and not report.get("clean_closure"):
            raise RuntimeError("native episode closure failed")
    if not report.get("clean_closure"):
        raise RuntimeError("native episode closure failed")
    return report
