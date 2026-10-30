"""Digital native-tick neighbourhoods and strict native Goal scoring."""
from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator, Mapping
from typing import Any

UID = 'oqIJ5rQDRrNwLPTh9H2p_W4tLof'
ACTIVE = 1771
LENGTH = 1775


def digest(ticks: tuple[int, ...]) -> str:
    return hashlib.sha256(bytes(ticks)).hexdigest()


def validate(ticks: tuple[int, ...]) -> None:
    if len(ticks) != LENGTH or ticks[ACTIVE:] != (36,) * 4:
        raise ValueError('Fixed observed clock and unchanged padding required')
    if any(type(x) is not int or not 0 <= x < 78 or x % 3 == 2
           or x // 6 not in (0, 6, 12) for x in ticks):
        raise ValueError('Exact digital per-tick controls only')


def native_goal(row: Mapping[str, Any]) -> int | None:
    if row.get('MapUid') != UID:
        raise ValueError('Wrong map in native result')
    result = row.get('ValidatedResult')
    if result is None:
        return None
    if not isinstance(result, dict):
        raise ValueError('Malformed native result')
    time = result.get('Time')
    if (result.get('NbCheckpoints') != 1 or result.get('NbRespawns') != 0
            or type(time) is not int or not 0 < time < 150000):
        raise ValueError('Native Goal with one checkpoint, zero respawns and finite time required')
    # IsValid can be false solely because the unmodified declared result is faster.
    # Only the engine's computed ValidatedResult supplies fitness.
    return time


def edges(ticks: tuple[int, ...]) -> list[int]:
    return [i for i in range(1, ACTIVE) if ticks[i] != ticks[i - 1]]


def neighbourhood(ticks: tuple[int, ...], rng: random.Random, scale: int = 1
                  ) -> Iterator[tuple[str, tuple[int, ...]]]:
    """Whole-lap boundary shifts, preserving the fixed 10ms execution clock."""
    validate(ticks)
    positions = edges(ticks)
    rng.shuffle(positions)
    widths = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]
    if scale > 1:
        widths += [64, 80]
    # Shuffle globally rather than stopping at the first small improvement.
    moves = [(i, w, direction) for i in positions for w in widths for direction in (-1, 1)]
    rng.shuffle(moves)
    for i, width, direction in moves:
        start, stop, value = ((i - width, i, ticks[i]) if direction == -1
                              else (i, i + width, ticks[i - 1]))
        if start < 0 or stop > ACTIVE or len(set(ticks[start:stop])) != 1:
            continue
        yield 'edge', ticks[:start] + (value,) * width + ticks[stop:]


def mutate(ticks: tuple[int, ...], rng: random.Random, count: int, scale: int
           ) -> tuple[int, ...]:
    edited = list(ticks)
    for _ in range(count):
        boundaries = edges(tuple(edited))
        width = rng.choice([1, 2, 3, 4, 5, 8, 12, 16] + ([24, 32] if scale > 1 else []))
        if boundaries and rng.random() < .65:
            start = rng.choice(boundaries) + rng.randint(-width, width)
        else:
            start = rng.randrange(ACTIVE)
        start = max(0, min(ACTIVE - width, start))
        old = edited[start]
        steer, variant = divmod(old, 6)
        gas, brake = divmod(variant, 3)
        kind = rng.randrange(4)
        if kind < 2:
            steer = rng.choice([s for s in (0, 6, 12) if s != steer])
        elif kind == 2:
            gas = 1 - gas
        else:
            brake = 1 - brake
        edited[start:start + width] = [steer * 6 + gas * 3 + brake] * width
    return tuple(edited)
