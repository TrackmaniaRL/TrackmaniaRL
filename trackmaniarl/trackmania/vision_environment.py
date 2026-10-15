"""Synchronous screen capture paired with the Openplanet reward/control environment."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from trackmaniarl.core.contracts import FeaturePipeline
from trackmaniarl.core.data import Transition
from trackmaniarl.core.environment_errors import EnvironmentPausedError
from trackmaniarl.trackmania.environment import OpenPlanetEnvironmentFactory
from trackmaniarl.trackmania.environment_config import TrackmaniaEnvironmentConfig
from trackmaniarl.trackmania.window_capture import WindowCaptureConfig, WindowsClientCapture


class CaptureRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    left: int = 0
    top: int = 0
    width: int = Field(default=1280, ge=1)
    height: int = Field(default=720, ge=1)


class FrameSource(Protocol):
    def capture(self) -> np.ndarray[Any, Any]: ...

    def close(self) -> None: ...


class ScreenFrameSource:
    """Capture a configured desktop rectangle; the game must remain visible."""

    def __init__(self, region: CaptureRegion) -> None:
        try:
            import mss
        except ImportError as exc:
            raise ImportError("Live vision requires pip install 'trackmaniarl[vision]'") from exc
        self._screen = mss.mss()
        self._region = region.model_dump()

    def capture(self) -> np.ndarray[Any, Any]:
        bgra = np.asarray(self._screen.grab(self._region))
        return np.ascontiguousarray(bgra[:, :, 2::-1])

    def close(self) -> None:
        self._screen.close()


class VisionEnvironment:
    """Replace policy telemetry with RGB while retaining telemetry-based rewards."""

    def __init__(
        self,
        environment: Any,
        frames: FrameSource,
        *,
        include_telemetry: bool = False,
    ) -> None:
        self.environment = environment
        self.frames = frames
        self.include_telemetry = include_telemetry

    def _capture(self, info: dict[str, Any]) -> np.ndarray[Any, Any]:
        started = perf_counter()
        frame = self.frames.capture()
        info["vision/capture_ms"] = (perf_counter() - started) * 1000.0
        if isinstance(self.frames, WindowFrameSource):
            info.update(
                {f"vision/client_{key}": value for key, value in self.frames.region.items()}
            )
        return frame

    def _observation(self, telemetry: Any, info: dict[str, Any]) -> Any:
        received = perf_counter()
        images = self._capture(info)
        # A host-side lower bound, not the age of the rendered frame or telemetry packet.
        info["vision/pairing_delay_ms"] = (perf_counter() - received) * 1000.0
        if not self.include_telemetry:
            return images
        observation = {"telemetry": telemetry, "images": images}
        return observation

    def reset(self, *, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        with self._release_on_pause():
            prepare = getattr(self.frames, "prepare_episode", None)
            if callable(prepare):
                prepare()
            telemetry, info = self.environment.reset(seed=seed)
            return self._observation(telemetry, info), info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        with self._release_on_pause():
            validate = getattr(self.frames, "validate", None)
            if callable(validate):
                validate()
            telemetry, reward, terminated, truncated, info = self.environment.step(action)
            return self._observation(telemetry, info), reward, terminated, truncated, info

    @contextmanager
    def _release_on_pause(self) -> Iterator[None]:
        try:
            yield
        except EnvironmentPausedError:
            self.environment.controller.apply(np.zeros(3, dtype=np.float32))
            raise

    def close(self) -> None:
        try:
            self.environment.close()
        finally:
            self.frames.close()


class VisionEnvironmentFactory:
    def __init__(  # noqa: PLR0913 - capture and observation mode are independent keyword options
        self,
        config: TrackmaniaEnvironmentConfig | dict[str, Any],
        *,
        capture: CaptureRegion | Mapping[str, Any] | None = None,
        window_capture: WindowCaptureConfig | Mapping[str, Any] | None = None,
        include_telemetry: bool = False,
        base_dir: str | Path = ".",
    ) -> None:
        if capture is not None and window_capture is not None:
            raise ValueError("Choose capture or window_capture, not both")
        self._factory = OpenPlanetEnvironmentFactory(config, base_dir=base_dir)
        self.config = self._factory.config
        self.capture = CaptureRegion.model_validate({} if capture is None else capture)
        self.window_capture = (
            None if window_capture is None else WindowCaptureConfig.model_validate(window_capture)
        )
        self.include_telemetry = include_telemetry

    def create(self, *, seed: int, evaluation_map: Any | None = None) -> VisionEnvironment:
        frames = (
            ScreenFrameSource(self.capture)
            if self.window_capture is None
            else WindowFrameSource(self.window_capture)
        )
        try:
            environment = self._factory.create(seed=seed, evaluation_map=evaluation_map)
        except BaseException:
            frames.close()
            raise
        return VisionEnvironment(
            environment,
            frames,
            include_telemetry=self.include_telemetry,
        )

    def load_demonstration(self, path: str | Path, pipeline: FeaturePipeline) -> list[Transition]:
        raise ValueError("Telemetry-only demonstrations have no RGB frames for vision training")


class WindowFrameSource(ScreenFrameSource):
    """Capture the client area; reject geometry changes during an individual grab."""

    def __init__(self, config: WindowCaptureConfig) -> None:
        self._window = WindowsClientCapture(config)
        super().__init__(CaptureRegion(width=config.client_width, height=config.client_height))

    @property
    def region(self) -> dict[str, int]:
        return dict(self._region)

    def prepare_episode(self) -> None:
        self._region = self._window.prepare_episode()

    def validate(self) -> None:
        self._region = self._window.bounds()

    def capture(self) -> np.ndarray[Any, Any]:
        from trackmaniarl.trackmania.window_capture import (
            CaptureWindowError,
            CaptureWindowUnavailableError,
        )

        with self._window.physical_pixels():
            self.validate()
            region = self.region
            result = super().capture()
            if self._window.bounds() != region:
                raise CaptureWindowUnavailableError(
                    "Window moved during frame capture; frame discarded"
                )
        if result.shape[:2] != (region["height"], region["width"]):
            raise CaptureWindowError("Captured frame dimensions differ from the game client")
        return result
