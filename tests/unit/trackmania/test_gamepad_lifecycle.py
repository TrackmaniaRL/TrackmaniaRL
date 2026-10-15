from __future__ import annotations

import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from trackmaniarl.trackmania.control import GamepadController


class _LifetimePad:
    def __init__(self, events: list[str], failure: str | None) -> None:
        self.events = events
        self.failure = failure
        self.callback: object | None = None

    def register_notification(self, *, callback_function: object) -> None:
        self.callback = callback_function

    def unregister_notification(self) -> None:
        self._record("unregister")

    def reset(self) -> None:
        self._record("reset")

    def update(self) -> None:
        self._record("update")

    def _record(self, event: str) -> None:
        self.events.append(event)
        if event == self.failure:
            raise RuntimeError(f"simulated {event} failure")

    def __del__(self) -> None:
        self.events.append("detach")


@pytest.mark.parametrize("failure", [None, "reset", "unregister"])
def test_gamepad_close_releases_callback_cycle_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    events: list[str] = []
    pads: list[weakref.ReferenceType[_LifetimePad]] = []

    def create_pad() -> _LifetimePad:
        pad = _LifetimePad(events, failure)
        pads.append(weakref.ref(pad))
        return pad

    monkeypatch.setattr(
        "trackmaniarl.trackmania.control._vgamepad_module",
        lambda: SimpleNamespace(VX360Gamepad=create_pad),
    )
    controller = GamepadController()
    if failure is None:
        controller.close()
    else:
        with pytest.raises(RuntimeError, match=f"simulated {failure} failure"):
            controller.close()
    assert controller._gamepad is None
    assert controller._vibration_callback is None
    assert "unregister" in events
    assert pads[0]() is None
    assert events[-1] == "detach"
    closed_events = list(events)
    controller.close()
    assert events == closed_events
    with pytest.raises(RuntimeError, match="GamepadController is closed"):
        controller.apply(np.zeros(3, dtype=np.float32))
    with pytest.raises(RuntimeError, match="GamepadController is closed"):
        controller.reset()
    with pytest.raises(RuntimeError, match="GamepadController is closed"):
        controller.confirm_finish()
