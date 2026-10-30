"""Compare staged motor feedback against the actual historical controller methods."""

import struct
from threading import RLock
from types import SimpleNamespace
from typing import cast

import pytest

from trackmaniarl.trackmania.control import GamepadController
from trackmaniarl.trackmania.native_headless.full_qr.motor_feedback import (
    MotorFeedbackLatch,
    callback_large_motor,
    queued_motor_words,
)


def test_all_callback_bytes_match_historical_controller_without_creating_device() -> None:
    reference = cast(
        GamepadController, SimpleNamespace(_collision_lock=RLock(), _collision_detected=False)
    )
    actual = MotorFeedbackLatch()
    for large in range(256):
        GamepadController._record_vibration(reference, large)
        actual.record_callback(large)
        assert actual.consume_collision() == GamepadController.consume_collision(reference)
        assert actual.consume_collision() == GamepadController.consume_collision(reference)
    for large in (101, 0, 100):
        GamepadController._record_vibration(reference, large)
        actual.record_callback(large)
    assert actual.consume_collision() == GamepadController.consume_collision(reference) is True
    assert actual.consume_collision() is False


def test_measured_motor_word_threshold_and_single_consumption() -> None:
    latch = MotorFeedbackLatch()
    latch.record_xinput(25855)
    assert latch.consume_collision() is False
    latch.record_xinput(25856)
    latch.record_xinput(0)
    assert latch.consume_collision() is True
    assert latch.consume_collision() is False
    assert callback_large_motor(65535) == 255


@pytest.mark.parametrize("value", [-1, 65536, 1.5, True])
def test_invalid_xinput_word_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="16-bit"):
        callback_large_motor(cast(int, value))


@pytest.mark.parametrize("value", [-1, 256, 1.5, True])
def test_invalid_callback_byte_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="8-bit"):
        MotorFeedbackLatch().record_callback(cast(int, value))


def test_queued_channels_follow_original_windows_word_storage_order() -> None:
    assert queued_motor_words(0.0, 1.0) == (65535, 0)
    assert queued_motor_words(1.0, 0.0) == (0, 65535)
    assert queued_motor_words(-0.5, 10.0) == (65535, 0)
    assert queued_motor_words(0.5, 0.5) == (32767, 32767)


@pytest.mark.parametrize(
    ("bits", "expected"),
    [
        (1053425353, 25854),
        (1053425354, 25855),
        (1053425355, 25855),
        (1053425865, 25855),
        (1053425866, 25856),
        (1053425867, 25856),
    ],
)
def test_measured_native_float32_boundaries(bits: int, expected: int) -> None:
    # Expected values came from the unchanged Windows machine-code fragment.
    value = struct.unpack("<f", struct.pack("<I", bits))[0]
    assert queued_motor_words(0.0, value) == (expected, 0)


@pytest.mark.parametrize("channel", [float("nan"), float("inf"), float("-inf"), True, 1])
def test_nonfinite_or_untyped_queue_channel_is_rejected(channel: object) -> None:
    with pytest.raises(ValueError, match="finite queued float"):
        queued_motor_words(0.0, cast(float, channel))
