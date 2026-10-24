"""Motor bridge/latch calibrated on the pinned Windows XInput/ViGEm stack.

This does not derive rumble from physics. A native producer must be qualified
before its output can enter a full-QR reward transition.
"""

from math import isfinite
from struct import pack, unpack
from threading import RLock


def _channel_to_word(channel: float) -> int:
    if type(channel) is not float or not isfinite(channel):
        raise ValueError("finite queued float channel required")
    bounded = unpack("<f", pack("<f", min(1.0, max(0.0, channel))))[0]
    product = unpack("<f", pack("<f", bounded * 65535.0))[0]
    return int(product)


def queued_motor_words(first: float, second: float) -> tuple[int, int]:
    """Match the pinned Windows device-buffer fragment, in word storage order.

    The second queue channel becomes the first word. Both clamping and the
    float32 product precede truncation. This component does not prove the
    protected device call, physics source, or notification timing.
    """
    return _channel_to_word(second), _channel_to_word(first)


def callback_large_motor(xinput_word: int) -> int:
    """Return the callback byte; all 256 bucket endpoints were measured."""
    if type(xinput_word) is not int or not 0 <= xinput_word <= 65535:
        raise ValueError("unsigned 16-bit XInput motor word required")
    return xinput_word >> 8


class MotorFeedbackLatch:
    """Match the historical strict >100 callback latch and consume semantics."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._pending = False

    def record_callback(self, large_motor: int) -> None:
        if type(large_motor) is not int or not 0 <= large_motor <= 255:
            raise ValueError("unsigned 8-bit ViGEm callback motor required")
        if large_motor > 100:
            with self._lock:
                self._pending = True

    def record_xinput(self, xinput_word: int) -> None:
        self.record_callback(callback_large_motor(xinput_word))

    def consume_collision(self) -> bool:
        with self._lock:
            result = self._pending
            self._pending = False
        return result
