"""Lazy camera history for feed-forward policies and episode-safe replay windows."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from typing import Any, cast

import torch

from trackmaniarl.core.pytree import PyTree
from trackmaniarl.models.composite import (
    BatchLayout,
    CompositeModules,
    CompositeValueModel,
    FrameBatchAdapter,
    history_padding,
)
from trackmaniarl.models.encoders.driving_stack import DrivingStackEncoder
from trackmaniarl.models.factory import CompositeValueModelFactory
from trackmaniarl.models.temporal.identity import IdentityTemporalCore


def stack_sequence(images: torch.Tensor, length: int) -> torch.Tensor:
    """Return B,T,(history*C),H,W, oldest first; only left-pad the episode boundary."""
    if images.ndim != 5 or not images.shape[1] or length < 1:
        raise ValueError("stack_sequence requires B,T,C,H,W and positive history length")
    padded = torch.cat((images[:, :1].expand(-1, length - 1, -1, -1, -1), images), dim=1)
    windows = padded.unfold(1, length, 1).permute(0, 1, 5, 2, 3, 4)
    return windows.flatten(2, 3)


def policy_stack(
    images: torch.Tensor, state: Any, length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if images.ndim != 4 or length < 1:
        raise ValueError("policy_stack requires B,C,H,W and positive history length")
    if state is None:
        history = images.unsqueeze(1).expand(-1, length, -1, -1, -1).clone()
    else:
        if not isinstance(state, torch.Tensor) or state.shape != (
            len(images),
            length,
            *images.shape[1:],
        ):
            raise ValueError("invalid camera history state")
        history = torch.cat((state[:, 1:], images.unsqueeze(1)), dim=1)
    return history.flatten(1, 2), history


class DrivingStackValueModel(CompositeValueModel):
    """IQN on raw-frame stacks; replay stores one frame, never four copies.

    burn_in is a context prefix, not recurrent warm-up. The first history-1
    positions of a replay window are excluded from loss because their preceding
    frames are unavailable. Target histories are supplied by the n-step sampler.
    """

    def __init__(self, modules: CompositeModules) -> None:
        super().__init__(modules)
        if not isinstance(self.encoder, DrivingStackEncoder) or not isinstance(
            self.temporal, IdentityTemporalCore
        ):
            raise ValueError("stack value model requires DrivingStackEncoder and identity core")
        self.frame_stack = self.encoder.frame_stack

    def architecture_fingerprint(self) -> str:
        identity = (
            f"driving-stack-v1:{super().architecture_fingerprint()}:"
            f"{self.encoder.channels}:{self.frame_stack}:{self.encoder.shift_padding}"
        )
        return sha256(identity.encode()).hexdigest()

    def encode_sequence(
        self, observation: PyTree, layout: BatchLayout, burn_in: int
    ) -> torch.Tensor:
        values = cast(Mapping[str, torch.Tensor], observation)
        if layout is not BatchLayout.SEQUENCE:
            raise ValueError("stack value training requires contiguous replay history")
        FrameBatchAdapter.flatten(observation, layout)
        images, vehicle = values["images"], values["vehicle"]
        if not self.frame_stack - 1 <= burn_in < images.shape[1]:
            raise ValueError("stack context prefix must cover frame_stack-1 frames")
        stacks = stack_sequence(images, self.frame_stack)[:, burn_in:]
        frames = {"images": stacks.flatten(0, 1), "vehicle": vehicle[:, burn_in:].flatten(0, 1)}
        return self.encode_frames(frames).reshape(images.shape[0], -1, self.encoder.output_dim)

    def encode_masked_sequence(  # noqa: PLR0913 - sequence layout, burn-in and validity are independent
        self,
        observation: PyTree,
        layout: BatchLayout,
        burn_in: int,
        *,
        masks: torch.Tensor | None,
    ) -> torch.Tensor:
        if masks is None:
            return self.encode_sequence(observation, layout, burn_in)
        if layout is not BatchLayout.SEQUENCE:
            raise ValueError("stack value training requires contiguous replay history")
        batch = FrameBatchAdapter.flatten(observation, layout)
        padding = history_padding(masks, (batch.batch_size, batch.time_steps))
        values = cast(Mapping[str, torch.Tensor], observation)
        images = values["images"]
        first = images[torch.arange(batch.batch_size, device=images.device), padding]
        # Padding values are not observations. Repeat the first real frame, as at reset.
        images = torch.where(masks[:, :, None, None, None], images, first[:, None])
        return self.encode_sequence({**values, "images": images}, layout, burn_in)

    def initial_policy_state(self, batch_size: int, device: torch.device) -> PyTree:
        del batch_size, device
        return None

    def policy_step(self, observation: PyTree, state: PyTree) -> tuple[torch.Tensor, PyTree]:
        values = cast(Mapping[str, torch.Tensor], observation)
        images, history = policy_stack(values["images"], state, self.frame_stack)
        return self.encode_frames({"images": images, "vehicle": values["vehicle"]}), history


class DrivingStackValueModelFactory(CompositeValueModelFactory):
    def build(self) -> DrivingStackValueModel:
        model = super().build()
        return DrivingStackValueModel(
            CompositeModules(model.encoder, model.temporal, model.head, model.strategy)
        )
