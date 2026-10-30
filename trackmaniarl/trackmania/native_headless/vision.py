"""Native physics-paired schematic images, with episode-local vision history."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from math import prod
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from gymnasium import spaces

from trackmaniarl.builtins.features import GymnasiumObservationCollator
from trackmaniarl.core.data import Transition
from trackmaniarl.trackmania.geometry import BoundaryGeometry
from trackmaniarl.trackmania.native_headless.full_qr.committed_terminal import (
    NativeCommittedGoalFrame,
)
from trackmaniarl.trackmania.native_headless.full_qr.native_features import (
    NativeGoalTerminalFrame,
    NativeGraphFrame,
)
from trackmaniarl.trackmania.native_headless.ribbon_renderer import (
    RibbonCameraConfig,
    RibbonRenderer,
)
from trackmaniarl.trackmania.vision import VisionConfig, VisionFeaturePipeline
from trackmaniarl.trackmania.vision_models import VisionSensorEncoder

type RibbonStorage = Literal["uint8", "base3"]
_TRIT_WEIGHTS = np.array([1, 3, 9, 27, 81], dtype=np.uint16)


def _storage_shape(config: VisionConfig, storage: RibbonStorage) -> tuple[int, ...]:
    shape = (config.frame_stack, config.height, config.width)
    if storage == "uint8":
        return shape
    if storage == "base3":
        return ((prod(shape) + 4) // 5,)
    raise ValueError("ribbon storage must be uint8 or base3")


def _pack_pixels(images: torch.Tensor) -> torch.Tensor:
    """Encode five palette IDs per byte, with zero padding in the last group."""
    pixels = images.numpy().reshape(-1)
    if pixels.size % 5:
        pixels = np.pad(pixels, (0, 5 - pixels.size % 5))
    groups = pixels.reshape(-1, 5)
    encoded = (groups * _TRIT_WEIGHTS).sum(axis=-1, dtype=np.uint16).astype(np.uint8)
    return torch.from_numpy(encoded)


class NativeSchematicVisionPipeline:
    """Render the observed road ribbon, not the game's camera or a future state.

    Use as the native factory's observation_pipeline. Rewards, Goal evidence,
    applied actions and the engine's source guards remain owned by that factory.
    The SDK feature component is PreparedVisionFeaturePipeline, which unwraps
    images without advancing frame history a second time.
    """

    def __init__(  # noqa: PLR0913 - asset identity, camera and preprocessing are explicit contracts
        self,
        geometry_path: str | Path,
        expected_map_uid: str,
        *,
        camera: RibbonCameraConfig | Mapping[str, Any] | None = None,
        vision: VisionConfig | Mapping[str, Any] | None = None,
        compact: bool = False,
        storage: RibbonStorage = "uint8",
    ) -> None:
        self.geometry = BoundaryGeometry(geometry_path, expected_map_uid=expected_map_uid)
        self.renderer = RibbonRenderer(self.geometry, camera)
        self.features = VisionFeaturePipeline(vision)
        if type(compact) is not bool:
            raise TypeError("compact must be an explicit bool")
        self.compact = compact
        _storage_shape(self.features.config, storage)
        if storage != "uint8" and not compact:
            raise ValueError("base3 storage requires compact ribbon pixels")
        self.storage = storage
        self._indexed_history: deque[torch.Tensor] = deque(maxlen=self.features.config.frame_stack)
        if (self.renderer.config.width, self.renderer.config.height) != (
            self.features.config.width,
            self.features.config.height,
        ):
            raise ValueError("render directly at the configured vision width and height")
        self.reset_episode()

    def reset_episode(self) -> None:
        self.features.reset_episode()
        self._indexed_history.clear()
        self.last_frame_time_ms: float | None = None
        self._last_rgb: np.ndarray | None = None
        self._last_indexed: np.ndarray | None = None
        self._last_channels: tuple[float, ...] | None = None
        self._observation: dict[str, torch.Tensor] | None = None
        self._ended = False

    @property
    def last_rgb(self) -> np.ndarray | None:
        """Expand compact pixels only when diagnostics request RGB."""
        if self._last_rgb is None and self._last_indexed is not None:
            self._last_rgb = np.asarray(self.renderer.rgb_palette, dtype=np.uint8)[
                self._last_indexed
            ]
            self._last_rgb.setflags(write=False)
        return self._last_rgb

    def transform_observation(self, observation: NativeGraphFrame) -> dict[str, torch.Tensor]:
        if type(observation) is not NativeGraphFrame:
            raise TypeError("vision requires a typed actual native graph frame")
        expected = 0.0 if self.last_frame_time_ms is None else self.last_frame_time_ms + 50.0
        if self._ended or observation.channels[0] != expected:
            raise ValueError("vision episode must start at 0 and advance exactly 50ms")
        return self._render(observation.channels)

    def transform_terminal(self, observation: NativeGoalTerminalFrame) -> dict[str, torch.Tensor]:
        if type(observation) is not NativeGoalTerminalFrame:
            raise TypeError("vision requires a typed partial Goal frame")
        self._check_terminal(observation.capture.previous_native_t)
        result = self._render(observation.channels)
        self._ended = True
        return result

    def transform_committed_terminal(
        self, observation: NativeCommittedGoalFrame
    ) -> dict[str, torch.Tensor]:
        if type(observation) is not NativeCommittedGoalFrame:
            raise TypeError("vision requires a typed committed Goal frame")
        self._check_terminal(observation.capture.previous_native_t)
        if observation.channels[0] == self.last_frame_time_ms:
            if observation.channels != self._last_channels or self._observation is None:
                raise ValueError("same-clock committed Goal must preserve its actual observation")
            result = self._observation
        else:
            result = self._render(observation.channels)
        self._ended = True
        return result

    def _check_terminal(self, previous_native_t: int) -> None:
        if self._ended or self.last_frame_time_ms != previous_native_t - 2200:
            raise ValueError("terminal vision must follow this episode's last regular frame")

    def _render(self, channels: tuple[float, ...]) -> dict[str, torch.Tensor]:
        values = np.asarray(channels, dtype=np.float64)
        self._last_rgb, self._last_indexed = None, None
        if self.compact:
            image = self.renderer.render_indexed(values[1:4], values[7:10])
            frame = torch.from_numpy(image)
            if not self._indexed_history:
                self._indexed_history.extend([frame] * self.features.config.frame_stack)
            else:
                self._indexed_history.append(frame)
            prepared = torch.stack(tuple(self._indexed_history))
            if self.storage == "base3":
                prepared = _pack_pixels(prepared)
            self._last_indexed = image
        else:
            image = self.renderer.render(values[1:4], values[7:10])
            prepared = self.features.transform_observation(image)
            self._last_rgb = image
        observation = {"images": prepared}
        image.setflags(write=False)
        self.last_frame_time_ms = float(values[0])
        self._last_channels, self._observation = channels, observation
        return observation


class RibbonVisionFeaturePipeline:
    """Keep lossless uint8 palette-index stacks in replay and transport.

    No quantization of normalized grayscale takes place. The matching encoder
    expands the three actual visualization colors at the device boundary.
    """

    def __init__(
        self,
        config: VisionConfig | Mapping[str, Any] | None = None,
        *,
        storage: RibbonStorage = "uint8",
    ) -> None:
        self.config = VisionConfig.model_validate({} if config is None else config)
        self.storage = storage
        self._shape = _storage_shape(self.config, storage)
        self._maximum = 2 if storage == "uint8" else 242
        remainder = (self.config.frame_stack * self.config.height * self.config.width) % 5
        self._last_maximum = 3**remainder - 1 if remainder else 242
        self.observation_space = spaces.Box(0, self._maximum, shape=self._shape, dtype=np.uint8)
        self._collator = GymnasiumObservationCollator(self.observation_space)

    def reset_episode(self) -> None:
        pass

    def transform_observation(self, observation: Any) -> torch.Tensor:
        if not isinstance(observation, Mapping) or set(observation) != {"images"}:
            raise ValueError("ribbon vision requires exactly the prepared images sensor")
        images = observation["images"]
        if (
            not isinstance(images, torch.Tensor)
            or images.dtype != torch.uint8
            or tuple(images.shape) != self._shape
            or bool((images > self._maximum).any())
        ):
            raise ValueError(
                f"ribbon images require uint8 values 0..{self._maximum} and the storage shape"
            )
        if self.storage == "base3" and bool(images[-1] > self._last_maximum):
            raise ValueError("base3 ribbon padding must be zero")
        return images

    def collate(self, transitions: list[Transition]) -> dict[str, Any]:
        return dict(self._collator.collate_transitions(transitions))

    def synthetic_observation(self) -> dict[str, torch.Tensor]:
        return {"images": torch.zeros(self._shape, dtype=torch.uint8)}


class RibbonVisionSensorEncoder(VisionSensorEncoder):
    """The same CNN, with exact palette expansion on the inference/training device."""

    _palette: torch.Tensor
    _packed_palette: torch.Tensor
    fingerprint_parameters: tuple[str, ...] = ("frame_stack", "grayscale", "representation")
    representation = RibbonRenderer.representation

    def __init__(  # noqa: PLR0913 - image layout and CNN dimensions are explicit contracts
        self,
        config: VisionConfig | Mapping[str, Any] | None = None,
        output_dim: int = 256,
        hidden_dim: int = 64,
        *,
        storage: RibbonStorage = "uint8",
    ) -> None:
        settings = VisionConfig.model_validate({} if config is None else config)
        super().__init__(settings.channels, output_dim, hidden_dim)
        self.frame_stack = settings.frame_stack
        self.grayscale = settings.grayscale
        self.storage = storage
        self.image_shape = (settings.frame_stack, settings.height, settings.width)
        self._storage_shape = _storage_shape(settings, storage)
        if storage == "base3":
            self.representation = f"{RibbonRenderer.representation}/base3-v1"
            self.fingerprint_parameters = (*self.fingerprint_parameters, "image_shape")
        palette = torch.tensor(RibbonRenderer.rgb_palette, dtype=torch.float32).div_(255.0)
        if settings.grayscale:
            palette = (palette * palette.new_tensor([0.299, 0.587, 0.114])).sum(dim=1, keepdim=True)
        # Derived constant; no new learned parameters or checkpoint migration.
        self.register_buffer("_palette", palette, persistent=False)
        digits = (
            torch.arange(243)[:, None].div(torch.tensor([1, 3, 9, 27, 81]), rounding_mode="floor")
            % 3
        )
        self.register_buffer("_packed_palette", palette[digits], persistent=False)

    def decode(self, images: torch.Tensor) -> torch.Tensor:
        if self.storage == "base3":
            if images.dtype != torch.uint8 or images.shape[-1:] != self._storage_shape:
                raise ValueError("base3 ribbon encoder requires the configured uint8 byte vector")
            pixels = self._packed_palette[images.long()].flatten(-3, -2)
            pixels = pixels[..., : prod(self.image_shape), :].reshape(
                *images.shape[:-1], *self.image_shape, self._palette.shape[-1]
            )
            return pixels.movedim(-1, -3).flatten(-4, -3).contiguous()
        if images.dtype != torch.uint8 or images.ndim < 3 or images.shape[-3] != self.frame_stack:
            raise ValueError("ribbon encoder requires (..., frame_stack, height, width) uint8 IDs")
        return self._palette[images.long()].movedim(-1, -3).flatten(-4, -3).contiguous()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return super().forward(self.decode(images))
