"""Windows SGB conversion and bounded host-time execution of model macroactions.

Host deadlines are not physics acknowledgements. Literal timing is retained so
an attempted client run is never presented as qualified native-tick parity.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np


class Pad(Protocol):
    def apply(self, action: np.ndarray) -> None: ...


class ExactDigitalPad:
    """Delegate only exact digital controls; never quantize analog model output."""

    def __init__(self, pad: Pad) -> None:
        self.pad = pad

    def apply(self, action: np.ndarray) -> None:
        if (action.shape != (3,) or action[0] not in (0, 1) or action[1] not in (0, 1)
                or action[2] not in (-1, 0, 1)):
            raise ValueError('keyboard cannot execute fractional steer exactly')
        self.pad.apply(action)


def control(action: int) -> np.ndarray:
    """Canonical action78 SGB -> Windows controller GBS, no curve fabrication."""
    if type(action) is not int or not 0 <= action < 78:
        raise ValueError('native action78 command required')
    steer, variant = divmod(action, 6)
    gas, brake = divmod(variant, 3)
    # Native macro provider calls session.advance(command, 1): mode2 is the
    # first tick of action_sgb(command), so brake is1 for this literal tick.
    return np.asarray([gas, min(brake, 1), (steer - 6) / 6], dtype=np.float32)


def execute(  # noqa: PLR0913 -- explicit controls, guard and optional shared host epoch
    program: Sequence[int], pad: Pad, guard: Callable[[], None],
    *, anchor: float | None = None,
) -> list[dict[str, float | int]]:
    if len(program) != 5:
        raise ValueError('exactly five motor commands required')
    commands = [control(action) for action in program]
    anchor = time.perf_counter() if anchor is None else anchor
    receipts: list[dict[str, float | int]] = []
    for offset, command in enumerate(commands):
        due = anchor + offset * 0.01
        while time.perf_counter() < due:
            guard()
            time.sleep(min(0.001, max(0, due - time.perf_counter())))
        guard()
        before = time.perf_counter()
        if before - due > 0.03:
            raise RuntimeError('motor scheduling more than30ms late')
        pad.apply(command)
        receipts.append({'offset': offset, 'native_command': program[offset],
                         'host_before_s': before, 'host_after_s': time.perf_counter(),
                         'host_lateness_ms': (before - due) * 1000})
    while time.perf_counter() < anchor + 0.05:
        guard()
        time.sleep(0.001)
    return receipts
