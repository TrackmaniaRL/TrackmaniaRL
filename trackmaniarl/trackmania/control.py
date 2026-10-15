"""Controller backends for TrackMania; optional drivers are imported lazily."""

from __future__ import annotations

from inspect import Parameter, Signature
from threading import RLock
from time import sleep
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

import numpy as np

from trackmaniarl.trackmania.actions import BRAKE_TAP_DURATION_S, BRAKE_TAP_SENTINEL
from trackmaniarl.trackmania.keyboard_control import (
    KeyboardController as KeyboardController,
)
from trackmaniarl.trackmania.keyboard_control import (
    ResetGuard,
    restart_trackmania_editor_validation,
    restart_trackmania_race,
)
from trackmaniarl.trackmania.steering_curve import SteeringCurve


@runtime_checkable
class Controller(Protocol):
    def apply(self, action: np.ndarray) -> None: ...

    def apply_discrete(self, action: np.ndarray) -> None: ...

    def consume_collision(self) -> bool: ...

    def reset(self) -> None: ...

    def confirm_finish(self) -> None: ...

    def close(self) -> None: ...


class _VibrationCallback:
    __signature__: ClassVar[Signature] = Signature(
        tuple(
            Parameter(name, Parameter.POSITIONAL_OR_KEYWORD)
            for name in (
                "client",
                "target",
                "large_motor",
                "small_motor",
                "led_number",
                "user_data",
            )
        )
    )

    def __init__(self, controller: GamepadController) -> None:
        self.controller = controller

    def __call__(self, *values: object) -> None:
        if len(values) != 6:
            raise TypeError("vgamepad vibration callback requires six values")
        large_motor = values[2]
        if not isinstance(large_motor, int):
            raise TypeError("vgamepad large-motor value must be an integer")
        self.controller._record_vibration(large_motor)


def _vgamepad_module() -> Any:
    try:
        import vgamepad
    except ImportError as exc:
        raise RuntimeError(
            "GamepadController requires the pinned vgamepad fork. Add "
            "'vgamepad @ git+https://github.com/Palamabron/vgamepad@"
            "5f3435df3f8a0e658feb58b207d9137cdb5183cd' to your project, then sync."
        ) from exc
    return vgamepad


class GamepadController:
    """Virtual XInput controller with an explicit TrackMania restart action.

    With a measured ``steering_curve`` the requested steer is what the game applies:
    the controller sends the stick position that the game's own analog curve turns
    into that steer. Without one, steer is sent as a raw stick position.
    """

    _CONFIRM_BUTTON = 0x1000  # Xbox A; menu Select binding.
    _RESTART_BUTTON = 0x2000  # Xbox B; TrackMania's default Give Up binding.

    def __init__(
        self,
        *,
        restart_input: Literal["gamepad", "keyboard", "editor_validation"] = "gamepad",
        steering_curve: SteeringCurve | None = None,
    ) -> None:
        self._gamepad = _vgamepad_module().VX360Gamepad()
        self._steering_curve = steering_curve
        self._tap_lock = RLock()
        self._collision_lock = RLock()
        self._collision_detected = False
        self._restart_input = restart_input
        self._vibration_callback: _VibrationCallback | None = _VibrationCallback(self)
        self._gamepad.register_notification(callback_function=self._vibration_callback)

    def _record_vibration(self, large_motor: int) -> None:
        if large_motor > 100:
            with self._collision_lock:
                self._collision_detected = True

    def consume_collision(self) -> bool:
        """Return and clear the most recent collision rumble event."""

        with self._collision_lock:
            collision_detected = self._collision_detected
            self._collision_detected = False
        return collision_detected

    def _open_gamepad(self) -> Any:
        if self._gamepad is None:
            raise RuntimeError("GamepadController is closed")
        return self._gamepad

    def _apply(self, action: np.ndarray) -> None:
        gamepad = self._open_gamepad()
        gas, brake, steer = np.clip(
            np.nan_to_num(action, nan=0.0), [-0.0, 0.0, -1.0], [1.0, 1.0, 1.0]
        )
        stick = float(steer)
        if self._steering_curve is not None:
            stick = self._steering_curve.stick_for(stick)
        gamepad.right_trigger_float(float(gas))
        gamepad.left_trigger_float(float(brake))
        gamepad.left_joystick_float(stick, 0.0)
        gamepad.update()

    def apply(self, action: np.ndarray) -> None:
        with self._tap_lock:
            self._apply(action)

    def apply_discrete(self, action: np.ndarray) -> None:
        """Apply a table action, releasing the brake after the explicit tap interval."""

        control = np.asarray(action, dtype=np.float32).copy()
        if control.shape != (3,):
            raise ValueError("discrete TrackMania control must be [gas, brake, steer]")
        if float(control[1]) == BRAKE_TAP_SENTINEL:
            with self._tap_lock:
                self._apply(np.asarray([control[0], 1.0, control[2]], dtype=np.float32))
                sleep(BRAKE_TAP_DURATION_S)
                self._apply(np.asarray([control[0], 0.0, control[2]], dtype=np.float32))
            return
        self.apply(control)

    def reset(self, *, guard: ResetGuard | None = None) -> None:
        """Release controls and request a TrackMania restart before an episode.

        The optional guard supervises keyboard/editor reset keys and waits.
        Native gamepad restart only checks the guard at entry.
        """

        with self._tap_lock:
            if guard is not None:
                guard.check()
            gamepad = self._open_gamepad()
            gamepad.reset()
            gamepad.update()
            match self._restart_input:
                case "keyboard":
                    if guard is None:
                        restart_trackmania_race()
                    else:
                        restart_trackmania_race(guard=guard)
                case "editor_validation":
                    if guard is None:
                        restart_trackmania_editor_validation()
                    else:
                        restart_trackmania_editor_validation(guard=guard)
                case "gamepad":
                    self._restart_with_gamepad()
        self.consume_collision()

    def _restart_with_gamepad(self) -> None:
        self._tap_button(self._RESTART_BUTTON)

    def _tap_button(self, button: int) -> None:
        gamepad = self._open_gamepad()
        gamepad.press_button(button=button)
        gamepad.update()
        sleep(0.1)
        gamepad.release_button(button=button)
        gamepad.update()

    def confirm_finish(self) -> None:
        """Confirm TrackMania menus through the active virtual controller."""

        with self._tap_lock:
            self._tap_button(self._CONFIRM_BUTTON)

    def close(self) -> None:
        with self._tap_lock:
            gamepad = self._gamepad
            if gamepad is None:
                return
            try:
                gamepad.reset()
                gamepad.update()
            finally:
                try:
                    gamepad.unregister_notification()
                finally:
                    # Break the pad/callback/controller cycle so the driver's
                    # destructor detaches the device as soon as close returns.
                    self._gamepad = None
                    self._vibration_callback = None


class RecordingController:
    """Safe controller used by tests and dry diagnostics without game input."""

    def __init__(self) -> None:
        self.actions: list[np.ndarray] = []

    def apply(self, action: np.ndarray) -> None:
        self.actions.append(np.asarray(action, dtype=np.float32).copy())

    def apply_discrete(self, action: np.ndarray) -> None:
        self.apply(action)

    def consume_collision(self) -> bool:
        return False

    def reset(self) -> None:
        self.actions.clear()

    def confirm_finish(self) -> None:
        return None

    def close(self) -> None:
        return None
