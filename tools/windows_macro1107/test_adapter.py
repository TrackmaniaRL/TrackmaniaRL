"""Pure conversion and timing safety tests, no Windows pad or game access."""

import numpy as np
import pytest

from .adapter import ExactDigitalPad, control, execute


def test_native_sgb_to_windows_gbs() -> None:
    assert np.array_equal(control(39), [1, 0, 0])
    assert np.array_equal(control(75), [1, 0, 1])
    assert np.array_equal(control(3), [1, 0, -1])
    assert np.array_equal(control(40), [1, 1, 0])
    assert np.array_equal(control(41), [1, 1, 0])


def test_invalid_command() -> None:
    for value in (-1, 78):
        with pytest.raises(ValueError, match='action78'):
            control(value)


def test_macro_receipts_and_guard() -> None:
    class Pad:
        def __init__(self) -> None:
            self.commands: list[np.ndarray] = []

        def apply(self, action: np.ndarray) -> None:
            self.commands.append(action.copy())

    pad = Pad()
    receipts = execute((39, 75, 3, 40, 36), pad, lambda: None)
    assert len(pad.commands) == len(receipts) == 5
    assert [r['native_command'] for r in receipts] == [39, 75, 3, 40, 36]
    assert all(r['host_after_s'] >= r['host_before_s'] for r in receipts)


def test_guard_aborts_before_control() -> None:
    class NoPad:
        def apply(self, action: np.ndarray) -> None:
            pytest.fail('guard must abort before input')

    def stop() -> None:
        raise TimeoutError('test stop')

    with pytest.raises(TimeoutError, match='test stop'):
        execute((39,) * 5, NoPad(), stop)


def test_keyboard_never_rounds_fractional_steering() -> None:
    class NoPad:
        def apply(self, action: np.ndarray) -> None:
            pytest.fail('fractional input must not reach the keyboard')

    with pytest.raises(ValueError, match='fractional'):
        ExactDigitalPad(NoPad()).apply(control(45))


def test_schedule_does_not_hide_excessive_lateness() -> None:
    import time

    class NoPad:
        def apply(self, action: np.ndarray) -> None:
            pytest.fail('late schedule must fail before sending control')

    with pytest.raises(RuntimeError, match='scheduling'):
        execute((39,) * 5, NoPad(), lambda: None, anchor=time.perf_counter() - 0.1)
