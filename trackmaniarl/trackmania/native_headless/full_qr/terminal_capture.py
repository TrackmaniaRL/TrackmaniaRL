"""Decode an actual partial final capture without inventing a 50 ms boundary.

This is source evidence only. It supplies neither motor coverage nor a paired
UI qualification, and cannot be passed off as a regular graph/reward frame.
"""

from dataclasses import dataclass

from trackmaniarl.trackmania.native_headless.full_qr.final_capture import FinalCollectorBoundary
from trackmaniarl.trackmania.native_headless.full_qr.goal_delivery import (
    NativeGoalBoundary,
    NativeGoalContext,
    NativeWaypointFact,
    decode_goal_boundary,
    decode_goal_facts,
)
from trackmaniarl.trackmania.native_headless.full_qr.vehicle_capture import (
    VehicleCapture,
    VehicleCaptureContext,
    decode_vehicle_capture,
)


@dataclass(frozen=True, slots=True)
class PartialGoalCapture:
    previous_native_t: int
    vehicle: VehicleCapture
    goal: NativeGoalBoundary

    def __post_init__(self) -> None:
        end = self.vehicle.native_t
        event = self.goal.last_goal_native_t
        if (
            type(self.previous_native_t) is not int
            or not 2200 <= self.previous_native_t < end <= 152200
            or (self.previous_native_t - 2200) % 50
            or end % 10
            or not 0 < end - self.previous_native_t < 50
            or self.goal.native_t != end
            or not self.goal.goal_active
            or event is None
            or not self.previous_native_t < event <= end
        ):
            raise ValueError("actual partial interval with an occurred matching Goal required")

    @property
    def observed_interval_ms(self) -> int:
        return self.vehicle.native_t - self.previous_native_t


@dataclass(frozen=True, slots=True)
class CommittedGoalCapture:
    """Distinct clocks for a committed score after the last sampled physics.

    This is diagnostic source evidence, not an RL terminal authorization. The
    original partial-capture guard still requires an already occurred Goal.
    No state at the score timestamp or at the next physics tick is invented.
    """

    previous_native_t: int
    vehicle: VehicleCapture
    goal: NativeGoalBoundary
    fact: NativeWaypointFact
    final_boundary: FinalCollectorBoundary | None = None

    def __post_init__(self) -> None:
        sampled = self.vehicle.native_t
        event = self.fact.event_native_t
        if (
            type(self.previous_native_t) is not int
            or not 2200 <= self.previous_native_t <= sampled < event <= 152200
            or (self.previous_native_t - 2200) % 50
            or sampled % 10
            or not 0 <= sampled - self.previous_native_t <= 50
            or not 0 < event - sampled < 10
            or self.goal.native_t != sampled
            or self.goal.goal_active
            or self.goal.goals != 1
            or self.goal.last_goal_native_t != event
            or not self.fact.endpoint_requested
            or self.fact.entry_name != "Goal"
            or self.fact.dispatch_native_t != sampled
            or self.fact.callback_native_t != sampled - 10
        ):
            raise ValueError(
                "matching committed Goal, dispatch and actual physical sample required"
            )
        if sampled - self.previous_native_t == 50 and (
            type(self.final_boundary) is not FinalCollectorBoundary
            or self.final_boundary.native_t != sampled
            or self.final_boundary.goal_native_t != event
        ):
            raise ValueError("actual collector proof required for a completed final physical step")


def decode_committed_goal_capture(  # noqa: PLR0913 - independent primary records and clocks.
    vehicle_raw: bytes,
    goal_facts_raw: bytes,
    goal_receipt_raw: bytes,
    *,
    context: NativeGoalContext,
    previous_native_t: int,
    held_native_t: int,
) -> CommittedGoalCapture:
    """Decode completed native entry/store pairs without claiming future physics."""
    meta = context.meta
    vehicle = decode_vehicle_capture(
        vehicle_raw,
        VehicleCaptureContext(
            context.pid,
            meta["participant"],
            meta["vehicle"],
            meta["module_slide"],
            held_native_t,
        ),
    )
    facts = decode_goal_facts(goal_facts_raw, context)
    goal = decode_goal_boundary(goal_receipt_raw, facts, context)
    endpoints = tuple(fact for fact in facts if fact.endpoint_requested)
    if len(endpoints) != 1:
        raise ValueError("one complete committed Goal writer pair required")
    return CommittedGoalCapture(previous_native_t, vehicle, goal, endpoints[0])


def decode_partial_goal_capture(  # noqa: PLR0913 - three source records, independent binding and two observed clocks.
    vehicle_raw: bytes,
    goal_facts_raw: bytes,
    goal_receipt_raw: bytes,
    *,
    context: NativeGoalContext,
    previous_native_t: int,
    held_native_t: int,
) -> PartialGoalCapture:
    """Require matching actual vehicle/Goal clocks after an occurred endpoint.

    Context must come from the current episode's independently held binding.
    An EOF or a computed validator time alone is deliberately insufficient.
    """
    if (
        type(previous_native_t) is not int
        or type(held_native_t) is not int
        or not 2200 <= previous_native_t < held_native_t <= 152200
        or (previous_native_t - 2200) % 50
        or held_native_t % 10
        or not 0 < held_native_t - previous_native_t < 50
    ):
        raise ValueError("one actual partial interval after a regular 50ms boundary required")
    meta = context.meta
    vehicle = decode_vehicle_capture(
        vehicle_raw,
        VehicleCaptureContext(
            context.pid,
            meta["participant"],
            meta["vehicle"],
            meta["module_slide"],
            held_native_t,
        ),
    )
    facts = decode_goal_facts(goal_facts_raw, context)
    goal = decode_goal_boundary(goal_receipt_raw, facts, context)
    event = goal.last_goal_native_t
    if (
        goal.native_t != held_native_t
        or not goal.goal_active
        or event is None
        or not previous_native_t < event <= held_native_t
    ):
        raise ValueError("actual occurred Goal and matching held final capture required")
    return PartialGoalCapture(previous_native_t, vehicle, goal)
