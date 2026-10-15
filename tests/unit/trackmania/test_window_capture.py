"""Window capture must preserve client geometry and reject unusable observations."""

# ruff: noqa: N802 -- fake functions deliberately mirror the native Windows API

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from trackmaniarl.trackmania import window_capture
from trackmaniarl.trackmania.vision_environment import (
    VisionEnvironment,
    VisionEnvironmentFactory,
    WindowFrameSource,
)
from trackmaniarl.trackmania.window_capture import (
    CaptureWindowError,
    WindowCaptureConfig,
    WindowsClientCapture,
)


class _Desktop:
    def __init__(self) -> None:
        self.left, self.top = 200, 100
        self.width, self.height = 1600, 900
        self.minimized = False
        self.foreground = 1
        self.resize_calls: list[tuple[int, int]] = []
        self.dpi_context: Any = -2

    def SetThreadDpiAwarenessContext(self, context: Any) -> Any:
        previous = self.dpi_context
        self.dpi_context = getattr(context, "value", context)
        return previous

    def IsWindow(self, handle: Any) -> bool:
        return bool(handle)

    def IsWindowVisible(self, handle: Any) -> bool:
        return bool(handle)

    def IsIconic(self, handle: Any) -> bool:
        return self.minimized

    def GetForegroundWindow(self) -> int:
        return self.foreground

    def GetClientRect(self, handle: Any, output: Any) -> bool:
        output._obj.right, output._obj.bottom = self.width, self.height
        return True

    def ClientToScreen(self, handle: Any, output: Any) -> bool:
        output._obj.x, output._obj.y = self.left, self.top
        return True

    def GetWindowRect(self, handle: Any, output: Any) -> bool:
        output._obj.left, output._obj.top = self.left - 8, self.top - 32
        output._obj.right, output._obj.bottom = (
            self.left + self.width + 8,
            self.top + self.height + 8,
        )
        return True

    def SetWindowPos(self, *args: Any) -> bool:
        self.left, self.top = args[2] + 8, args[3] + 32
        self.width, self.height = args[4] - 16, args[5] - 40
        self.resize_calls.append((self.width, self.height))
        return True

    def MonitorFromWindow(self, handle: Any, flags: Any) -> int:
        return 2

    def GetMonitorInfoW(self, handle: Any, output: Any) -> bool:
        output._obj.monitor.left = -1920
        output._obj.monitor.right = 3840
        output._obj.monitor.bottom = 2160
        output._obj.work.left = -1920
        output._obj.work.right = 3840
        output._obj.work.bottom = 2120
        return True


@pytest.fixture
def desktop(monkeypatch: pytest.MonkeyPatch) -> _Desktop:
    desktop = _Desktop()
    monkeypatch.setattr(window_capture, "_user32", lambda: desktop)
    monkeypatch.setattr(WindowsClientCapture, "_find", lambda self: 1)
    monkeypatch.setattr(window_capture, "sleep", lambda duration: None)
    return desktop


def test_episode_resize_uses_client_size_and_tracks_negative_screen_coordinates(
    desktop: _Desktop,
) -> None:
    capture = WindowsClientCapture(WindowCaptureConfig(settle_s=0))
    assert capture.prepare_episode() == {"left": 200, "top": 100, "width": 1280, "height": 720}
    assert desktop.resize_calls == [(1280, 720)]
    desktop.left = -1600
    assert capture.bounds()["left"] == -1600
    capture.prepare_episode()
    assert len(desktop.resize_calls) == 1
    desktop.width, desktop.height = 960, 540
    with pytest.raises(CaptureWindowError, match="resized during"):
        capture.bounds()
    capture.prepare_episode()
    assert desktop.resize_calls[-1] == (1280, 720)
    assert desktop.dpi_context == -2


@pytest.mark.parametrize("fault", ["minimized", "background", "offscreen"])
def test_invalid_window_fails_and_restores_thread_dpi(desktop: _Desktop, fault: str) -> None:
    capture = WindowsClientCapture(WindowCaptureConfig())
    capture.prepare_episode()
    if fault == "minimized":
        desktop.minimized = True
    elif fault == "background":
        desktop.foreground = 3
    else:
        desktop.left = 3700
    with pytest.raises(CaptureWindowError):
        capture.bounds()
    assert desktop.dpi_context == -2


def test_reset_recovers_partly_offscreen_window(desktop: _Desktop) -> None:
    desktop.left, desktop.top = 3500, 1500
    capture = WindowsClientCapture(WindowCaptureConfig())
    region = capture.prepare_episode()
    assert region["left"] + region["width"] <= 3840
    assert region["top"] + region["height"] <= 2120


def test_requested_size_must_fit_monitor(desktop: _Desktop) -> None:
    capture = WindowsClientCapture(WindowCaptureConfig(client_width=8192))
    with pytest.raises(CaptureWindowError, match="does not fit"):
        capture.prepare_episode()
    assert not desktop.resize_calls


def test_native_api_is_optional_on_other_platforms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(window_capture.sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="requires Windows"):
        WindowsClientCapture(WindowCaptureConfig())


def test_frame_source_follows_window_and_discards_a_racing_move(
    desktop: _Desktop,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    regions = []
    move_during_capture = False

    def grab(region: dict[str, int]) -> np.ndarray[Any, Any]:
        regions.append(dict(region))
        if move_during_capture:
            desktop.left += 5
        result = np.zeros((region["height"], region["width"], 4), dtype=np.uint8)
        result[:, :, 2] = 255
        return result

    import sys

    monkeypatch.setitem(
        sys.modules,
        "mss",
        SimpleNamespace(mss=lambda: SimpleNamespace(grab=grab, close=lambda: None)),
    )
    source = WindowFrameSource(WindowCaptureConfig())
    source.prepare_episode()
    desktop.left = 400
    frame = source.capture()
    assert regions[-1]["left"] == 400
    assert frame.shape == (720, 1280, 3)
    assert frame[0, 0].tolist() == [255, 0, 0]
    move_during_capture = True
    with pytest.raises(CaptureWindowError, match="moved during"):
        source.capture()
    source.close()


def test_window_validation_precedes_action_and_prepare_runs_each_reset() -> None:
    order: list[str] = []

    def reject() -> None:
        raise CaptureWindowError("minimized")

    frames = SimpleNamespace(
        prepare_episode=lambda: order.append("prepare"),
        capture=lambda: np.zeros((2, 2, 3), dtype=np.uint8),
        validate=reject,
    )
    environment = SimpleNamespace(
        reset=lambda **kw: (order.append("reset"), {}),
        step=lambda action: order.append("action"),
    )
    wrapped = VisionEnvironment(environment, frames)
    wrapped.reset()
    wrapped.reset()
    with pytest.raises(CaptureWindowError, match="minimized"):
        wrapped.step(1)
    assert order == ["prepare", "reset", "prepare", "reset"]


def test_window_mode_is_explicit_and_rejects_legacy_demonstrations() -> None:
    with pytest.raises(ValueError, match="Choose capture"):
        VisionEnvironmentFactory({}, capture={}, window_capture={})
    factory = VisionEnvironmentFactory(
        {"geometry_path": "unused.npz"}, window_capture={}, include_telemetry=True
    )
    with pytest.raises(ValueError, match="no RGB frames"):
        factory.load_demonstration("unused.npz", None)  # type: ignore[arg-type]


@pytest.mark.parametrize("during_capture", [False, True])
def test_focus_loss_releases_controls_without_returning_a_transition(
    *, during_capture: bool
) -> None:
    from trackmaniarl.trackmania.window_capture import CaptureWindowUnavailableError

    commands = []
    actions = []

    def unavailable() -> None:
        raise CaptureWindowUnavailableError("background")

    frames = SimpleNamespace(
        validate=(lambda: None) if during_capture else unavailable,
        capture=unavailable,
    )
    environment = SimpleNamespace(
        controller=SimpleNamespace(apply=lambda value: commands.append(value.copy())),
        step=lambda action: (actions.append(action), 0.0, False, False, {}),
    )
    wrapped = VisionEnvironment(environment, frames)
    with pytest.raises(CaptureWindowUnavailableError):
        wrapped.step(1)
    assert actions == ([1] if during_capture else [])
    assert len(commands) == 1
    np.testing.assert_array_equal(commands[0], np.zeros(3))
