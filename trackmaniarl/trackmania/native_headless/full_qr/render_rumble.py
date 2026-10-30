"""Explicit virtual 150-FPS input-clock reconstruction for qualification.

This is a declared simulation policy, not a record of measured client clocks.
It supplies no reward watermark or collision authority by itself.
"""

from dataclasses import dataclass

from trackmaniarl.trackmania.native_headless.full_qr.gameplay_rumble import (
    interpolate_visual_rumble,
    unity_gameplay_pulse,
)
from trackmaniarl.trackmania.native_headless.full_qr.motor_feedback import queued_motor_words
from trackmaniarl.trackmania.native_headless.full_qr.physical_rumble import PhysicalRumbleStep
from trackmaniarl.trackmania.native_headless.full_qr.rumble_producer import RumbleState
from trackmaniarl.trackmania.native_headless.full_qr.rumble_queue import (
    RumblePulse,
    evaluate_rumble_queue,
)


@dataclass(frozen=True, slots=True)
class ReconstructedMotorChange:
    sequence: int
    race_time_ms: int
    large_word: int
    small_word: int

    @property
    def large_motor(self) -> int:
        return self.large_word >> 8


class Virtual150RumbleClock:
    """Process every virtual input frame after complete actual physical ticks.

    Retains both words for change detection, uses strict queue expiry, and
    preserves interpolation's one-physics-tick visual delay. A failed advance
    does not commit any clock, queue, sequence or waveform history.
    """

    def __init__(self, bootstrap: RumbleState) -> None:
        if type(bootstrap) is not RumbleState:
            raise ValueError("actual bootstrap waveform required")
        self.native_t = 2190
        self._history = {2190: bootstrap}
        self._next_frame = 0
        self._pulses: tuple[RumblePulse, ...] = ()
        self._words = (0, 0)
        self._sequence = 0
        self.rendered_through_native_t = 2190
        self._finished = False

    @property
    def rendered_frames(self) -> int:
        return self._next_frame

    def advance(self, step: PhysicalRumbleStep) -> tuple[ReconstructedMotorChange, ...]:
        if self._finished:
            raise ValueError("committed terminal render stream is already closed")
        if type(step) is not PhysicalRumbleStep or step.native_t != self.native_t + 10:
            raise ValueError("consecutive actual ten-ms physical rumble steps required")
        if not 2200 <= step.native_t <= 152200:
            raise ValueError("physical step outside the full150s race domain")
        history = {**self._history, step.native_t: step.state}
        result = self._render_through(step.native_t, history)
        self.native_t = step.native_t
        self._history = {
            clock: value for clock, value in history.items() if clock >= step.native_t - 20
        }
        return result

    def finish_committed(self, event_native_t: int) -> tuple[ReconstructedMotorChange, ...]:
        """Render to a committed score using only already captured endpoints.

        The interpolation delay uses the previous/current physical samples,
        so a score less than one tick after the current sample requires no
        new physical state. This method supplies no Goal authority itself.
        """
        if (
            self._finished
            or type(event_native_t) is not int
            or not self.native_t < event_native_t < self.native_t + 10
            or event_native_t > 152200
        ):
            raise ValueError("one bounded committed terminal render interval required")
        result = self._render_through(event_native_t, self._history)
        self._finished = True
        return result

    def _render_through(
        self, through: int, history: dict[int, RumbleState]
    ) -> tuple[ReconstructedMotorChange, ...]:
        frame, pulses, words, sequence = (
            self._next_frame,
            self._pulses,
            self._words,
            self._sequence,
        )
        changes = []
        while True:
            race = frame * 1000 // 150
            native = 2200 + race
            if native > through:
                break
            current = native // 10 * 10
            previous = current - 10
            if previous not in history or current not in history:
                raise ValueError("both actual physical interpolation endpoints required")
            visual = interpolate_visual_rumble(
                history[previous], history[current], (native - current) / 10.0
            )
            delta = race - (frame - 1) * 1000 // 150
            pulse = unity_gameplay_pulse(visual.amplitude, visual.frequency, 1.0, race, delta)
            retained = evaluate_rumble_queue(pulses, race).retained
            if pulse is not None:
                retained += (pulse,)
            evaluation = evaluate_rumble_queue(retained, race)
            new_words = queued_motor_words(*evaluation.channels)
            if new_words != words:
                if race == 0:
                    raise ValueError("unexpected nonzero reset motor notification")
                sequence += 1
                changes.append(ReconstructedMotorChange(sequence, race, *new_words))
            pulses, words = evaluation.retained, new_words
            frame += 1
        self._next_frame, self._pulses = frame, pulses
        self._words, self._sequence = words, sequence
        self.rendered_through_native_t = through
        return tuple(changes)
