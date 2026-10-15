"""Windows keyboard control for TrackMania."""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass
from enum import Enum
from threading import RLock, Timer
from time import monotonic, sleep

import numpy as np

from trackmaniarl.trackmania.actions import BRAKE_TAP_DURATION_S, BRAKE_TAP_SENTINEL

_FOCUS_TIMEOUT_S = 5.0
_FOCUS_POLL_S = 0.25


class _MouseInput(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouse_data", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("extra_info", ctypes.c_size_t),
    )


class _KeyboardInput(ctypes.Structure):
    _fields_ = (
        ("virtual_key", wintypes.WORD),
        ("scan_code", wintypes.WORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("extra_info", ctypes.c_size_t),
    )


class _HardwareInput(ctypes.Structure):
    _fields_ = (
        ("message", wintypes.DWORD),
        ("parameter_low", wintypes.WORD),
        ("parameter_high", wintypes.WORD),
    )


class _InputValue(ctypes.Union):
    _fields_ = (("mouse", _MouseInput), ("keyboard", _KeyboardInput), ("hardware", _HardwareInput))


class _Input(ctypes.Structure):
    _anonymous_ = ("value",)
    _fields_ = (("kind", wintypes.DWORD), ("value", _InputValue))


class _KeyState(Enum):
    PRESSED = "pressed"
    RELEASED = "released"


@dataclass(frozen=True, slots=True)
class KeyboardKeyEvent:
    key: int
    pressed: bool


@dataclass(frozen=True, slots=True)
class ResetGuard:
    """Optional cancellation/deadline checks for a supervised keyboard reset."""

    check: Callable[[], object]
    wait: Callable[[float], None]


@dataclass(frozen=True, slots=True)
class _KeyStroke:
    key: int
    scan_code: int
    state: _KeyState


def _windows_dll(name: str) -> object:
    if sys.platform != "win32":
        raise RuntimeError(f"{name} is only available on Windows")
    loader = getattr(ctypes, "WinDLL", None)
    if loader is None:
        raise RuntimeError("Windows ctypes support is unavailable")
    return loader(name, use_last_error=True)


def _windows_call(library: object, name: str, *args: object) -> object:
    function = getattr(library, name)
    if not callable(function):
        raise RuntimeError(f"Windows API function is unavailable: {name}")
    return function(*args)


def _windows_int_call(library: object, name: str, *args: object) -> int:
    result = _windows_call(library, name, *args)
    if not isinstance(result, int):
        raise RuntimeError(f"Windows API function returned a non-integer result: {name}")
    return result


@dataclass(frozen=True, slots=True)
class _FocusedWindow:
    user32: object
    handle: object

    def verify(self) -> None:
        """Recheck the same window immediately before sending a reset key."""

        current = _windows_call(self.user32, "FindWindowW", None, "Trackmania")
        foreground = _windows_call(self.user32, "GetForegroundWindow")
        if current != self.handle or foreground != self.handle:
            raise RuntimeError(
                "Trackmania foreground changed before reset key: "
                f"target={self.handle}, current={current}, foreground={foreground}"
            )


def _focus_trackmania(*, guard: ResetGuard | None = None) -> _FocusedWindow | None:
    """Wait briefly for lawful activation, without bypassing Windows focus policy.

    Windows may deny a background process's activation request, even for an open
    game. Retry for a bounded interval so a transient denial or the user returning
    to the game need not abort training. Persistent denial still fails before any
    reset key, and a supplied guard can cancel every wait or attempt.
    """

    if guard is not None:
        guard.check()
    if sys.platform != "win32":
        return None
    user32 = _windows_dll("user32")
    # HWND is pointer-sized; ctypes' default int result truncates it on Win64.
    for name, argument_types, result_type in (
        ("FindWindowW", [wintypes.LPCWSTR, wintypes.LPCWSTR], wintypes.HWND),
        ("GetForegroundWindow", [], wintypes.HWND),
        ("SetForegroundWindow", [wintypes.HWND], wintypes.BOOL),
    ):
        function = getattr(user32, name)
        function.argtypes = argument_types
        function.restype = result_type
    window = _windows_call(user32, "FindWindowW", None, "Trackmania")
    if not window:
        return None
    deadline = monotonic() + _FOCUS_TIMEOUT_S
    activated = 0
    attempts = 0
    foreground = _windows_call(user32, "GetForegroundWindow")
    while foreground != window:
        if guard is not None:
            guard.check()
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise RuntimeError(
                "Trackmania foreground verification failed: "
                f"SetForegroundWindow={activated}, target={window}, foreground={foreground}, "
                f"attempts={attempts}, timeout_s={_FOCUS_TIMEOUT_S}"
            )
        if _windows_call(user32, "FindWindowW", None, "Trackmania") != window:
            raise RuntimeError("Trackmania window changed while waiting for foreground")
        activated = _windows_int_call(user32, "SetForegroundWindow", window)
        attempts += 1
        if guard is not None:
            guard.check()
        foreground = _windows_call(user32, "GetForegroundWindow")
        if foreground == window:
            break
        remaining = deadline - monotonic()
        (sleep if guard is None else guard.wait)(max(0.0, min(_FOCUS_POLL_S, remaining)))
        if guard is not None:
            guard.check()
        foreground = _windows_call(user32, "GetForegroundWindow")
    return _FocusedWindow(user32, window)


def _tap_reset_key(key: int, guard: ResetGuard | None, window: _FocusedWindow) -> None:
    if guard is not None:
        guard.check()
    window.verify()
    try:
        KeyboardController._windows_key_event(KeyboardKeyEvent(key, True))
        (sleep if guard is None else guard.wait)(0.1)
    finally:
        # Releasing a pressed key remains mandatory after cancellation or timeout.
        KeyboardController._windows_key_event(KeyboardKeyEvent(key, False))


def confirm_trackmania_finish(*, guard: ResetGuard | None = None) -> None:
    if sys.platform == "linux":
        from trackmaniarl.trackmania.linux_reset import tap_reset_key

        tap_reset_key("Return", guard=guard)
        return
    window = _focus_trackmania(guard=guard)
    if window is None:
        raise RuntimeError("Trackmania window was not found for keyboard confirmation")
    _tap_reset_key(0x0D, guard, window)


def restart_trackmania_race(*, guard: ResetGuard | None = None) -> None:
    if sys.platform == "linux":
        from trackmaniarl.trackmania.linux_reset import tap_reset_key

        tap_reset_key("Delete", guard=guard)
        return
    window = _focus_trackmania(guard=guard)
    if window is None:
        raise RuntimeError("Trackmania window was not found for keyboard restart")
    _tap_reset_key(0x2E, guard, window)


def restart_trackmania_editor_validation(*, guard: ResetGuard | None = None) -> None:
    restart_trackmania_race(guard=guard)
    (sleep if guard is None else guard.wait)(0.5)
    confirm_trackmania_finish(guard=guard)


class KeyboardController:
    """Digital W/S/A/D controller matching keyboard-recorded demonstrations."""

    _GAS = 0x57
    _BRAKE = 0x53
    _LEFT = 0x41
    _RIGHT = 0x44
    _RESTART = 0x2E
    _EXTENDED_KEYS = frozenset({0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E})
    _KEY_EXTENDED = 0x0001
    _KEY_UP = 0x0002
    _KEY_SCAN_CODE = 0x0008
    _STEERING_DEADZONE = 0.25

    def __init__(self, key_event: Callable[[KeyboardKeyEvent], None] | None = None) -> None:
        if key_event is None and sys.platform != "win32":
            raise RuntimeError("KeyboardController requires Windows")
        self._requires_focus = key_event is None
        self._key_event = key_event or self._windows_key_event
        self._pressed: set[int] = set()
        self._lock = RLock()
        self._tap_timer: Timer | None = None

    @classmethod
    def _windows_key_event(cls, command: KeyboardKeyEvent) -> None:
        user32 = _windows_dll("user32")
        scan_code = _windows_int_call(user32, "MapVirtualKeyW", command.key, 0)
        state = _KeyState.PRESSED if command.pressed else _KeyState.RELEASED
        event = cls._keyboard_input(_KeyStroke(command.key, scan_code, state))
        sent = _windows_int_call(user32, "SendInput", 1, ctypes.byref(event), ctypes.sizeof(_Input))
        if sent != 1:
            error_code = _windows_int_call(_windows_dll("kernel32"), "GetLastError")
            raise OSError(error_code, "SendInput failed")

    @classmethod
    def _keyboard_input(cls, stroke: _KeyStroke) -> _Input:
        return _Input(
            kind=1,
            keyboard=_KeyboardInput(
                virtual_key=0,
                scan_code=stroke.scan_code,
                flags=cls._keyboard_flags(stroke),
                time=0,
                extra_info=0,
            ),
        )

    @classmethod
    def _keyboard_flags(cls, stroke: _KeyStroke) -> int:
        flags = cls._KEY_SCAN_CODE
        if stroke.key in cls._EXTENDED_KEYS:
            flags |= cls._KEY_EXTENDED
        if stroke.state is _KeyState.RELEASED:
            flags |= cls._KEY_UP
        return flags

    def apply(self, action: np.ndarray) -> None:
        control = self._digital_control(action)
        self._validate_control(control)
        self._apply_target(self._key_target(control))

    def apply_discrete(self, action: np.ndarray) -> None:
        control = np.asarray(action, dtype=np.float32).copy()
        self._validate_control(control)
        if float(control[1]) != BRAKE_TAP_SENTINEL:
            self.apply(control)
            return
        control = self._digital_control_with_brake_tap(control)
        control[1] = 1.0
        with self._lock:
            self._cancel_tap_unlocked()
            self._set_pressed(self._key_target(control))
            self._tap_timer = Timer(BRAKE_TAP_DURATION_S, self._release_brake)
            self._tap_timer.daemon = True
            self._tap_timer.start()

    def _apply_target(self, target: set[int]) -> None:
        with self._lock:
            self._cancel_tap_unlocked()
            self._set_pressed(target)

    def _key_target(self, control: np.ndarray) -> set[int]:
        gas, brake, steer = (float(value) for value in control)
        target = set()
        if gas > 0.5:
            target.add(self._GAS)
        if brake > 0.5:
            target.add(self._BRAKE)
        if steer > 0.5:
            target.add(self._RIGHT)
        elif steer < -0.5:
            target.add(self._LEFT)
        return target

    @staticmethod
    def _validate_control(control: np.ndarray) -> None:
        if control.shape != (3,):
            raise ValueError("keyboard control must be [gas, brake, steer]")

    def _release_brake(self) -> None:
        with self._lock:
            self._tap_timer = None
            self._set_pressed(self._pressed - {self._BRAKE})

    def _cancel_tap_unlocked(self) -> None:
        if self._tap_timer is not None:
            self._tap_timer.cancel()
            self._tap_timer = None

    def _set_pressed(self, target: set[int]) -> None:
        for key in sorted(self._pressed - target):
            self._key_event(KeyboardKeyEvent(key, False))
        for key in sorted(target - self._pressed):
            self._key_event(KeyboardKeyEvent(key, True))
        self._pressed = target

    @classmethod
    def _digital_control(cls, action: np.ndarray) -> np.ndarray:
        return cls._quantized_control(action, cls._threshold_brake)

    @classmethod
    def _digital_control_with_brake_tap(cls, action: np.ndarray) -> np.ndarray:
        return cls._quantized_control(action, cls._preserve_brake_tap)

    @classmethod
    def _quantized_control(
        cls, action: np.ndarray, brake_value: Callable[[float], float]
    ) -> np.ndarray:
        control = np.nan_to_num(np.asarray(action, dtype=np.float32).reshape(-1))
        cls._validate_control(control)
        gas, brake, steer = (float(value) for value in control)
        digital_steer = 0.0
        if abs(steer) >= cls._STEERING_DEADZONE:
            digital_steer = float(np.sign(steer))
        return np.asarray([float(gas > 0.5), brake_value(brake), digital_steer], dtype=np.float32)

    @staticmethod
    def _threshold_brake(brake: float) -> float:
        return float(brake > 0.5)

    @staticmethod
    def _preserve_brake_tap(brake: float) -> float:
        if brake == BRAKE_TAP_SENTINEL:
            return BRAKE_TAP_SENTINEL
        return float(brake > 0.5)

    def consume_collision(self) -> bool:
        return False

    def reset(self) -> None:
        with self._lock:
            self._cancel_tap_unlocked()
            self._set_pressed(set())
            if self._requires_focus:
                window = _focus_trackmania()
                if window is None:
                    raise RuntimeError("Trackmania window was not found for keyboard control")
                window.verify()
            try:
                self._key_event(KeyboardKeyEvent(self._RESTART, True))
                sleep(0.1)
            finally:
                self._key_event(KeyboardKeyEvent(self._RESTART, False))

    def confirm_finish(self) -> None:
        confirm_trackmania_finish()

    def close(self) -> None:
        with self._lock:
            self._cancel_tap_unlocked()
            self._set_pressed(set())
