from __future__ import annotations

from collections.abc import Callable, Iterator
from ctypes import wintypes
from pathlib import Path
from threading import RLock

import pytest

from experiments.tmrl_test_comparison import preflight
from trackmaniarl.trackmania import keyboard_control as keyboard
from trackmaniarl.trackmania.control import GamepadController


class _WinFunction:
    def __init__(self, callback: Callable[..., object]) -> None:
        self.callback = callback
        self.argtypes: list[object] | None = None
        self.restype: object = None
        self.calls: list[tuple[object, ...]] = []

    def __call__(self, *args: object) -> object:
        self.calls.append(args)
        return self.callback(*args)


class _User32:
    def __init__(self, window: int | None, foreground: Iterator[int | None], result: int) -> None:
        last: list[int | None] = [None]

        def read_foreground() -> int | None:
            last[0] = next(foreground, last[0])
            return last[0]

        self.FindWindowW = _WinFunction(lambda *_: window)
        self.GetForegroundWindow = _WinFunction(read_foreground)
        self.SetForegroundWindow = _WinFunction(lambda *_: result)


@pytest.fixture
def fake_windows(monkeypatch: pytest.MonkeyPatch) -> list[keyboard.KeyboardKeyEvent]:
    events: list[keyboard.KeyboardKeyEvent] = []
    elapsed = [0.0]

    def pause(seconds: float) -> None:
        elapsed[0] += seconds

    monkeypatch.setattr(keyboard.sys, "platform", "win32")
    monkeypatch.setattr(keyboard, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(keyboard, "sleep", pause)
    monkeypatch.setattr(keyboard.KeyboardController, "_windows_key_event", events.append)
    return events


@pytest.mark.parametrize("result", [0, 1])
@pytest.mark.parametrize(
    "action", [keyboard.restart_trackmania_race, keyboard.confirm_trackmania_finish]
)
def test_unverified_foreground_never_sends_keys(  # noqa: PLR0913
    monkeypatch: pytest.MonkeyPatch,
    fake_windows: list[keyboard.KeyboardKeyEvent],
    result: int,
    action: Callable[[], None],
) -> None:
    user32 = _User32(42, iter([7, 7]), result)
    monkeypatch.setattr(keyboard, "_windows_dll", lambda _: user32)

    with pytest.raises(
        RuntimeError, match=f"SetForegroundWindow={result}, target=42, foreground=7"
    ):
        action()

    assert not fake_windows


def test_already_foreground_does_not_request_activation(
    monkeypatch: pytest.MonkeyPatch, fake_windows: list[keyboard.KeyboardKeyEvent]
) -> None:
    window = 0x1234567887654321
    user32 = _User32(window, iter([window]), 0)
    monkeypatch.setattr(keyboard, "_windows_dll", lambda _: user32)

    keyboard.restart_trackmania_race()

    assert not user32.SetForegroundWindow.calls
    assert user32.FindWindowW.restype is wintypes.HWND
    assert user32.FindWindowW.argtypes == [wintypes.LPCWSTR, wintypes.LPCWSTR]
    assert user32.GetForegroundWindow.restype is wintypes.HWND
    assert user32.GetForegroundWindow.argtypes == []
    assert user32.SetForegroundWindow.restype is wintypes.BOOL
    assert user32.SetForegroundWindow.argtypes == [wintypes.HWND]
    assert fake_windows == [
        keyboard.KeyboardKeyEvent(0x2E, True),
        keyboard.KeyboardKeyEvent(0x2E, False),
    ]


@pytest.mark.parametrize("result", [0, 1])
def test_actual_foreground_is_authoritative_even_if_activation_returned_zero(
    monkeypatch: pytest.MonkeyPatch, fake_windows: list[keyboard.KeyboardKeyEvent], result: int
) -> None:
    window = 0x1234567887654321
    user32 = _User32(window, iter([7, window]), result)
    monkeypatch.setattr(keyboard, "_windows_dll", lambda _: user32)

    keyboard.confirm_trackmania_finish()

    assert user32.SetForegroundWindow.calls == [(window,)]
    assert fake_windows == [
        keyboard.KeyboardKeyEvent(0x0D, True),
        keyboard.KeyboardKeyEvent(0x0D, False),
    ]


def test_transient_activation_denial_recovers_before_sending_reset(
    monkeypatch: pytest.MonkeyPatch, fake_windows: list[keyboard.KeyboardKeyEvent]
) -> None:
    user32 = _User32(42, iter([7, 7, 7, 42]), 0)
    results = iter([0, 1])

    def activate(*_: object) -> int:
        assert not fake_windows
        return next(results)

    user32.SetForegroundWindow.callback = activate
    monkeypatch.setattr(keyboard, "_windows_dll", lambda _: user32)

    keyboard.restart_trackmania_race()

    assert user32.SetForegroundWindow.calls == [(42,), (42,)]
    assert fake_windows == [
        keyboard.KeyboardKeyEvent(0x2E, True),
        keyboard.KeyboardKeyEvent(0x2E, False),
    ]


def test_user_returning_game_during_denied_activation_wait_recovers(
    monkeypatch: pytest.MonkeyPatch, fake_windows: list[keyboard.KeyboardKeyEvent]
) -> None:
    user32 = _User32(42, iter([7, 7, 42]), 0)
    monkeypatch.setattr(keyboard, "_windows_dll", lambda _: user32)

    keyboard.confirm_trackmania_finish()

    assert user32.SetForegroundWindow.calls == [(42,)]
    assert fake_windows == [
        keyboard.KeyboardKeyEvent(0x0D, True),
        keyboard.KeyboardKeyEvent(0x0D, False),
    ]


def test_persistent_denial_has_a_local_deadline_without_a_guard(
    monkeypatch: pytest.MonkeyPatch, fake_windows: list[keyboard.KeyboardKeyEvent]
) -> None:
    user32 = _User32(42, iter([7]), 0)
    monkeypatch.setattr(keyboard, "_windows_dll", lambda _: user32)

    with pytest.raises(RuntimeError, match=r"timeout_s=5\.0"):
        keyboard.restart_trackmania_race()

    assert keyboard.monotonic() == pytest.approx(5.0)
    assert len(user32.SetForegroundWindow.calls) == 20
    assert not fake_windows


@pytest.mark.parametrize("failure", ["stop", "deadline"])
def test_guard_interrupts_foreground_wait_before_any_reset_key(  # noqa: PLR0913
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_windows: list[keyboard.KeyboardKeyEvent],
    failure: str,
) -> None:
    user32 = _User32(42, iter([7]), 0)
    monkeypatch.setattr(keyboard, "_windows_dll", lambda _: user32)
    stop = tmp_path / "STOP"
    elapsed = [0.0]
    budget = preflight._Budget(0.1 if failure == "deadline" else 10.0, (stop,))
    monkeypatch.setattr(preflight, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(keyboard, "monotonic", lambda: elapsed[0])

    def pause(seconds: float) -> None:
        elapsed[0] += seconds
        if failure == "stop":
            stop.touch()

    monkeypatch.setattr(preflight, "sleep", pause)
    guard = keyboard.ResetGuard(check=budget.remaining, wait=budget.pause)

    with pytest.raises((preflight._PreflightCancelledError, TimeoutError)):
        keyboard.restart_trackmania_race(guard=guard)

    assert elapsed[0] <= 0.25
    assert user32.SetForegroundWindow.calls == [(42,)]
    assert not fake_windows


def test_focus_lost_after_activation_never_sends_key_down_or_key_up(
    monkeypatch: pytest.MonkeyPatch, fake_windows: list[keyboard.KeyboardKeyEvent]
) -> None:
    user32 = _User32(42, iter([7, 42, 7]), 1)
    monkeypatch.setattr(keyboard, "_windows_dll", lambda _: user32)

    with pytest.raises(RuntimeError, match="foreground changed before reset key"):
        keyboard.restart_trackmania_race()

    assert not fake_windows


def test_window_replaced_during_foreground_wait_aborts_without_reset_keys(
    monkeypatch: pytest.MonkeyPatch, fake_windows: list[keyboard.KeyboardKeyEvent]
) -> None:
    user32 = _User32(42, iter([7]), 0)
    windows = iter([42, 42, 99])
    user32.FindWindowW.callback = lambda *_: next(windows)
    monkeypatch.setattr(keyboard, "_windows_dll", lambda _: user32)

    with pytest.raises(RuntimeError, match="window changed while waiting"):
        keyboard.restart_trackmania_race()

    assert user32.SetForegroundWindow.calls == [(42,)]
    assert not fake_windows


def test_keyboard_controller_rechecks_foreground_before_delete(
    monkeypatch: pytest.MonkeyPatch, fake_windows: list[keyboard.KeyboardKeyEvent]
) -> None:
    user32 = _User32(42, iter([42, 7]), 0)
    monkeypatch.setattr(keyboard, "_windows_dll", lambda _: user32)
    controller = keyboard.KeyboardController()

    with pytest.raises(RuntimeError, match="foreground changed before reset key"):
        controller.reset()

    assert not fake_windows


def test_keyboard_controller_releases_delete_if_hold_is_interrupted(
    monkeypatch: pytest.MonkeyPatch, fake_windows: list[keyboard.KeyboardKeyEvent]
) -> None:
    user32 = _User32(42, iter([42]), 0)
    monkeypatch.setattr(keyboard, "_windows_dll", lambda _: user32)
    controller = keyboard.KeyboardController()

    def interrupted_wait(_: float) -> None:
        raise TimeoutError("interrupted key hold")

    monkeypatch.setattr(keyboard, "sleep", interrupted_wait)
    with pytest.raises(TimeoutError, match="interrupted key hold"):
        controller.reset()

    assert fake_windows == [
        keyboard.KeyboardKeyEvent(0x2E, True),
        keyboard.KeyboardKeyEvent(0x2E, False),
    ]


def test_missing_window_stops_editor_restart_before_any_key(
    monkeypatch: pytest.MonkeyPatch, fake_windows: list[keyboard.KeyboardKeyEvent]
) -> None:
    user32 = _User32(None, iter([]), 0)
    monkeypatch.setattr(keyboard, "_windows_dll", lambda _: user32)

    with pytest.raises(RuntimeError, match="window was not found"):
        keyboard.restart_trackmania_editor_validation()

    assert not fake_windows
    assert not user32.SetForegroundWindow.calls


def test_focus_loss_between_editor_restart_and_confirmation_stops_enter(
    monkeypatch: pytest.MonkeyPatch, fake_windows: list[keyboard.KeyboardKeyEvent]
) -> None:
    user32 = _User32(42, iter([42, 42, 7, None]), 0)
    monkeypatch.setattr(keyboard, "_windows_dll", lambda _: user32)

    with pytest.raises(RuntimeError, match="foreground=None"):
        keyboard.restart_trackmania_editor_validation()

    assert fake_windows == [
        keyboard.KeyboardKeyEvent(0x2E, True),
        keyboard.KeyboardKeyEvent(0x2E, False),
    ]


@pytest.mark.parametrize("phase", ["delete_hold", "before_enter"])
@pytest.mark.parametrize("failure", ["stop", "deadline"])
def test_guarded_gamepad_editor_reset_releases_delete_but_never_sends_enter(  # noqa: PLR0913
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_windows: list[keyboard.KeyboardKeyEvent],
    phase: str,
    failure: str,
) -> None:
    user32 = _User32(42, iter([42, 42]), 0)
    monkeypatch.setattr(keyboard, "_windows_dll", lambda _: user32)
    stop = tmp_path / "STOP"
    elapsed = [0.0]
    stop_at = 0.05 if phase == "delete_hold" else 0.2
    budget = preflight._Budget(stop_at if failure == "deadline" else 1.0, (stop,))
    monkeypatch.setattr(preflight, "monotonic", lambda: elapsed[0])

    def pause(seconds: float) -> None:
        elapsed[0] += seconds
        if failure == "stop" and elapsed[0] >= stop_at:
            stop.touch()

    monkeypatch.setattr(preflight, "sleep", pause)
    neutral: list[str] = []

    class Pad:
        def reset(self) -> None:
            neutral.append("reset")

        def update(self) -> None:
            neutral.append("update")

    controller = object.__new__(GamepadController)
    controller._gamepad = Pad()
    controller._tap_lock = RLock()
    controller._restart_input = "editor_validation"
    monkeypatch.setattr(controller, "consume_collision", lambda: False)
    guard = keyboard.ResetGuard(check=budget.remaining, wait=budget.pause)

    with pytest.raises((preflight._PreflightCancelledError, TimeoutError)):
        controller.reset(guard=guard)

    assert neutral == ["reset", "update"]
    assert fake_windows == [
        keyboard.KeyboardKeyEvent(0x2E, True),
        keyboard.KeyboardKeyEvent(0x2E, False),
    ]


def test_guarded_confirmation_always_releases_enter_when_wait_fails(
    monkeypatch: pytest.MonkeyPatch, fake_windows: list[keyboard.KeyboardKeyEvent]
) -> None:
    user32 = _User32(42, iter([42]), 0)
    monkeypatch.setattr(keyboard, "_windows_dll", lambda _: user32)

    def interrupted_wait(_: float) -> None:
        raise TimeoutError("deadline during Enter hold")

    guard = keyboard.ResetGuard(check=lambda: None, wait=interrupted_wait)
    with pytest.raises(TimeoutError, match="Enter hold"):
        keyboard.confirm_trackmania_finish(guard=guard)

    assert fake_windows == [
        keyboard.KeyboardKeyEvent(0x0D, True),
        keyboard.KeyboardKeyEvent(0x0D, False),
    ]
