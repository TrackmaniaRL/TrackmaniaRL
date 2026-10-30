"""Pure, deterministic single-edit candidates in native 10 ms motor ticks."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    kind: str
    start: int
    stop: int
    ticks: tuple[int, ...]

    @property
    def digest(self) -> str:
        return hashlib.sha256(bytes(self.ticks)).hexdigest()


def validate(ticks: tuple[int, ...], active: int) -> None:
    if not ticks or len(ticks) % 5 or not 0 < active <= len(ticks):
        raise ValueError("full five-tick programs and explicit active prefix required")
    if any(type(a) is not int or not 0 <= a < 78 or a % 3 == 2 for a in ticks):
        raise ValueError("canonical constant-per-tick SGB IDs only; reject pulse aliases")


def candidates(ticks: tuple[int, ...], active: int) -> list[Candidate]:
    """No time warp: shift one boundary, or replace one 1/2-tick local interval.

    Padding after the observed source input prefix is never changed. The qualified
    source tick offset is handled by the preparer, never adjusted by this search.
    """
    validate(ticks, active)
    result: list[Candidate] = []
    seen = {hashlib.sha256(bytes(ticks)).hexdigest()}

    def add(kind: str, start: int, replacement: tuple[int, ...]) -> None:
        stop = start + len(replacement)
        if start < 0 or stop > active:
            return
        edited = ticks[:start] + replacement + ticks[stop:]
        item = Candidate(kind, start, stop, edited)
        if item.digest not in seen:
            seen.add(item.digest)
            result.append(item)

    for boundary in range(1, active):
        if ticks[boundary] == ticks[boundary - 1]:
            continue
        for width in (1, 2):
            # Never cross a second boundary: genuinely move one existing edge.
            if boundary >= width and len(set(ticks[boundary - width:boundary])) == 1:
                add("edge_earlier", boundary - width, (ticks[boundary],) * width)
            if boundary + width <= active and len(set(ticks[boundary:boundary + width])) == 1:
                add("edge_later", boundary, (ticks[boundary - 1],) * width)
    for start in range(active):
        steer, variant = divmod(ticks[start], 6)
        gas, brake = divmod(variant, 3)
        alternatives = [("gas_tap", steer * 6 + (1 - gas) * 3 + brake),
                        ("brake_tap", steer * 6 + gas * 3 + 1 - brake)]
        alternatives.extend(("steer_step", s * 6 + variant)
                            for s in (steer - 1, steer + 1) if 0 <= s <= 12)
        for width in (1, 2):
            if start + width > active or len(set(ticks[start:start + width])) != 1:
                continue
            for kind, action in alternatives:
                add(kind, start, (action,) * width)
    return result


def select(pool: list[Candidate], budget: int, seed: int) -> list[Candidate]:
    """Interleave mutation families, shuffle reproducibly across the whole lap."""
    if budget <= 0:
        raise ValueError("positive evaluation budget required")
    rng = random.Random(seed)
    groups = {kind: [c for c in pool if c.kind == kind]
              for kind in sorted({c.kind for c in pool})}
    for group in groups.values():
        rng.shuffle(group)
    chosen: list[Candidate] = []
    while len(chosen) < budget and any(groups.values()):
        for group in groups.values():
            if group and len(chosen) < budget:
                chosen.append(group.pop())
    return chosen


def confirmed_improvement(goals: Sequence[int | None], incumbent_ms: int) -> bool:
    """Five exact repeat finishes, strictly faster. DNF is never a finish time.

    Caller must first verify native Goal provenance and closure; this function
    only decides the time gate, and cannot certify a public record or an agent.
    """
    return (len(goals) == 5 and all(type(g) is int and 0 < g < incumbent_ms for g in goals)
            and len(set(goals)) == 1)
