"""Record-attempt actor timing, separate from the fresh full-QR campaign profile."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trackmaniarl.distributed.actor import ActorRuntime
from trackmaniarl.distributed.actor_collection import summary


def native_goal_summary(reward: float, info: Mapping[str, Any], transitions: int) -> dict[str, Any]:
    """Preserve physical timing; score a finished lap from its qualified Goal fact."""
    result = summary(reward, info, transitions)
    if info.get("termination_reason") != "finished":
        return result
    goal = info.get("actual_goal_race_time_ms")
    if type(goal) is not int or not 0 < goal <= 150_000:
        raise ValueError("finished shortcut requires its actual native Goal timestamp")
    physical_ms = float(info["race_time_ms"])
    if not abs(physical_ms - goal) < 50:
        raise ValueError("native Goal must belong to the actual final physical interval")
    result.update(
        physical_race_time_s=physical_ms / 1000,
        race_time_s=goal / 1000,
        finish_time_s=goal / 1000,
        finish_time_source="native_goal_event",
        actual_goal_race_time_ms=goal,
    )
    return result


class NativeGoalActor(ActorRuntime):
    """Ordinary SDK actor, with precise Goal scoring for training and evaluation."""

    _summary = staticmethod(native_goal_summary)
