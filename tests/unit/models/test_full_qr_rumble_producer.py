"""Regressions at discontinuities measured in the original Windows block."""

from struct import pack

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.rumble_producer import (
    RumbleState,
    combine_contact_rumble,
)


@pytest.mark.parametrize(
    ("main", "frequency", "amplitude"),
    [
        (0.4999999701976776, 0.0, 0.0),
        (0.5, 2.0, 0.6600000262260437),
        (0.5000000596046448, 4.0, 0.6600000858306885),
        (1.0, 4.0, 1.0),
    ],
)
def test_actual_native_thresholds(main: float, frequency: float, amplitude: float) -> None:
    result = combine_contact_rumble(RumbleState(0.0, 0.0), (main, 0.0, 0.0))
    assert pack("<ff", result.frequency, result.amplitude) == pack("<ff", frequency, amplitude)


def test_contact_override_keeps_larger_existing_channels() -> None:
    result = combine_contact_rumble(RumbleState(20.0, 1.0), (0.5, 0.0, 0.0))
    assert result == RumbleState(20.0, 1.0)


@pytest.mark.parametrize("main", [float("nan"), float("inf"), -0.01, 1.01, True])
def test_unqualified_numeric_domain_rejected(main: float) -> None:
    with pytest.raises(ValueError, match="contact inputs"):
        combine_contact_rumble(RumbleState(0.0, 0.0), (main, 0.0, 0.0))
