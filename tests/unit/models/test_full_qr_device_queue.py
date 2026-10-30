"""Device queue contract using independently constructed synthetic entries."""

from struct import pack_into

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.gameplay_rumble import decode_device_queue
from trackmaniarl.trackmania.native_headless.full_qr.motor_feedback import queued_motor_words
from trackmaniarl.trackmania.native_headless.full_qr.rumble_queue import evaluate_rumble_queue


def synthetic_queue() -> bytes:
    raw = bytearray(196)
    pack_into("<I", raw, 0, 3)
    pack_into("<Iff", raw, 4, 40, 0.75, 0.25)  # expired
    pack_into("<Iff", raw, 16, 50, 0.25, 0.5)  # equality remains active
    pack_into("<Iff", raw, 28, 0, 0.5, 1.0)  # indefinite
    return bytes(raw)


def test_synthetic_queue_expiry_and_motor_channel_order() -> None:
    evaluation = evaluate_rumble_queue(decode_device_queue(synthetic_queue()), 50)
    assert len(evaluation.retained) == 2
    assert evaluation.channels == (0.5, 1.0)
    assert queued_motor_words(*evaluation.channels) == (65535, 32767)


def test_partial_queue_is_rejected() -> None:
    with pytest.raises(ValueError, match="complete actual"):
        decode_device_queue(synthetic_queue()[:-1])


def test_synthetic_queue_ignores_unused_slots_and_accepts_full_capacity() -> None:
    raw = bytearray(196)
    pack_into("<I", raw, 0, 16)
    for i in range(16):
        pack_into("<Iff", raw, 4 + i * 12, 0, i / 16, 0.0)
    assert len(decode_device_queue(bytes(raw))) == 16
    pack_into("<I", raw, 0, 0)
    assert decode_device_queue(bytes(raw)) == ()


def test_synthetic_queue_rejects_nonfinite_channels() -> None:
    raw = bytearray(synthetic_queue())
    pack_into("<f", raw, 8, float("nan"))
    with pytest.raises(ValueError, match="finite actual"):
        decode_device_queue(bytes(raw))


def test_corrupt_count_cannot_be_treated_as_empty_queue() -> None:
    raw = bytearray(synthetic_queue())
    raw[:4] = (17).to_bytes(4, "little")
    with pytest.raises(ValueError, match="exceeds source capacity"):
        decode_device_queue(bytes(raw))
