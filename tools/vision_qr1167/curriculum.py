"""Training-only random starts and bounded progress rewards for tmrl-test.

Snapshot restoration and nearest-path projection belong to the environment.
Neither a curriculum finish nor shaped reward is a full-lap record.
"""
from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class Start:
    snapshot: str | None
    perturb_steer: float
    perturb_ticks: int


def choose_start(snapshots: list[str], rng: random.Random, *, evaluation=False) -> Start:
    if evaluation:
        return Start(None, 0.0, 0)
    # Keep learning the actual start throughout training, not just short suffixes.
    snapshot = None if not snapshots or rng.random() < 0.5 else rng.choice(snapshots)
    return Start(snapshot, rng.uniform(-0.12, 0.12), rng.randint(1, 5))


class ProgressDiscontinuity(ValueError):
    """Reject this transition and reset the episode, never invent its reward."""


class ProgressReward:
    """Pay once for new path distance; no reward for revisiting checkpoints.

Projection must follow ordered checkpoint topology (not global nearest XYZ).
Reject physically impossible jumps, including accidental mid-episode restores.
"""
    def __init__(self, start_distance: float, *, corridor_m: float = 12.0):
        if not math.isfinite(start_distance) or corridor_m <= 0:
            raise ValueError("invalid reward configuration")
        self.maximum = start_distance
        self.corridor = corridor_m
        self.done = False

    def step(self, distance: float, lateral_m: float, dt: float, *, goal=False) -> float:
        if self.done:
            raise RuntimeError("reset reward after terminal transition")
        if not all(map(math.isfinite, (distance, lateral_m, dt))) or not 0 < dt <= 0.1:
            raise ValueError("invalid physics sample")
        advance = max(0.0, distance - self.maximum)
        if advance > 200 * dt + 1:
            raise ProgressDiscontinuity(f"progress discontinuity: advance={advance}, limit={200 * dt + 1}; refuse reward")
        # Corridor is a reward hint, not an enforced imitation action sequence.
        # Update maximum outside it too, so crossing the boundary cannot farm reward.
        self.maximum = max(self.maximum, distance)
        reward = (advance / 10 if abs(lateral_m) <= self.corridor else 0.0) - dt
        if goal:
            self.done = True
            reward += 10
        return reward


def validate_step(before: dict, after: dict, requested_ticks: int, map_uid: str) -> None:
    """Admission contract: producer evidence, not host wall-clock estimates."""
    if not 1 <= requested_ticks <= 5:
        raise ValueError("invalid tick count")
    if before['map_uid'] != map_uid or after['map_uid'] != map_uid:
        raise ValueError("map changed")
    if before['episode_id'] != after['episode_id']:
        raise ValueError("unexpected reset")
    if after['race_tick'] - before['race_tick'] != requested_ticks:
        raise ValueError("wrong physics step")
    if not after['paused'] or after['render_tick'] != after['race_tick']:
        raise ValueError("image is not paired with paused physics")
    if after['applied_action_id'] != after['requested_action_id']:
        raise ValueError("action acknowledgement mismatch")
