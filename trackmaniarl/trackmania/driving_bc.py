"""Camera behavior cloning with recurrent and feed-forward history baselines."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, cast

import torch
from torch import nn

from trackmaniarl.core.contracts import ModelContract
from trackmaniarl.models.driving_stack import policy_stack
from trackmaniarl.models.encoders.driving_stack import DrivingStackEncoder
from trackmaniarl.models.encoders.driving_vision import DrivingVisionEncoder
from trackmaniarl.models.temporal import GruTemporalCore
from trackmaniarl.trackmania.imitation_learning.model_contract import BehaviorCloningModel


class DrivingBehaviorCloningModel(BehaviorCloningModel):
    def __init__(self, channels: int = 1, burn_in: int = 8) -> None:
        super().__init__()
        self.action_ids = tuple(range(78))
        self.action_count = 78
        self.previous_action_conditioning = False
        self.previous_action_start = 78
        self.minimum_action_hold_steps = 1
        self.switch_logit_margin = 0.0
        self.burn_in = burn_in
        self.encoder = DrivingVisionEncoder(channels)
        self.temporal = GruTemporalCore(256, 256)
        self.head = nn.Linear(256, 78)

    def initial_policy_state(self, device: torch.device) -> Any:
        return self.temporal.initial_state(1, device)

    def policy_logits(
        self, observation: Mapping[str, torch.Tensor], state: Any
    ) -> tuple[torch.Tensor, Any]:
        features, state = self.temporal.step(self.encoder(observation), state)
        return self.head(features), state

    def forward(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        images, vehicle = observation["images"], observation["vehicle"]
        if images.ndim == 4:
            features = self.encoder(observation).unsqueeze(1)
            burn_in = 0
        else:
            batch, length = images.shape[:2]
            frames = {"images": images.flatten(0, 1), "vehicle": vehicle.flatten(0, 1)}
            features = self.encoder(frames).reshape(batch, length, 256)
            burn_in = self.burn_in
        encoded = self.temporal.unroll(features, burn_in)[:, -1]
        return cast(torch.Tensor, self.head(encoded))


class DrivingBehaviorCloningModelFactory:
    model_contract = ModelContract.CATEGORICAL_POLICY

    def __init__(self, channels: int = 1, burn_in: int = 8) -> None:
        self.channels, self.burn_in = channels, burn_in

    def build(self) -> DrivingBehaviorCloningModel:
        return DrivingBehaviorCloningModel(self.channels, self.burn_in)


class DrivingStackBehaviorCloningModel(BehaviorCloningModel):
    """Same lazy history online and offline; no recurrent hidden state."""

    def __init__(self, channels: int = 3, *, shift_padding: int = 4) -> None:
        super().__init__()
        self.action_ids = tuple(range(78))
        self.action_count = 78
        self.previous_action_conditioning = False
        self.previous_action_start = 78
        self.minimum_action_hold_steps = 1
        self.switch_logit_margin = 0.0
        self.burn_in = 0
        self.encoder = DrivingStackEncoder(channels, shift_padding=shift_padding)
        self.head = nn.Linear(256, 78)

    def initial_policy_state(self, device: torch.device) -> Any:
        del device
        return None

    def policy_logits(
        self, observation: Mapping[str, torch.Tensor], state: Any
    ) -> tuple[torch.Tensor, Any]:
        images, history = policy_stack(observation["images"], state, self.encoder.frame_stack)
        return self.head(
            self.encoder({"images": images, "vehicle": observation["vehicle"]})
        ), history

    def forward(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        images, vehicle = observation["images"], observation["vehicle"]
        if images.ndim != 5 or images.shape[1] != self.encoder.frame_stack:
            raise ValueError("stack BC requires exactly four chronological frames")
        frames = {"images": images.flatten(1, 2), "vehicle": vehicle[:, -1]}
        return cast(torch.Tensor, self.head(self.encoder(frames)))


class DrivingStackBehaviorCloningModelFactory:
    model_contract = ModelContract.CATEGORICAL_POLICY

    def __init__(self, channels: int = 3, *, shift_padding: int = 4) -> None:
        self.channels, self.shift_padding = channels, shift_padding

    def build(self) -> DrivingStackBehaviorCloningModel:
        return DrivingStackBehaviorCloningModel(self.channels, shift_padding=self.shift_padding)


def bc_configuration(rl_config: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(rl_config)
    config["run_id"] = rl_config["run_id"].replace("gru-iqn", "bc")
    config.pop("distributed", None)
    components = config["components"]
    channels = components["model_factory"]["kwargs"]["encoder"]["kwargs"]["channels"]
    components["learner"] = {
        "class_path": "trackmaniarl.trackmania.imitation_learning:BehaviorCloningLearner",
        "kwargs": {
            "learning_rate": 0.0001,
            "label_smoothing": 0.01,
            "max_steps": 3000,
            "validation_interval": 100,
            "early_stopping_patience": 10,
            "gradient_clip_norm": 5.0,
            "execution": {"device": "auto", "precision": "float32"},
        },
    }
    components["model_factory"] = {
        "class_path": "trackmaniarl.trackmania.driving_bc:DrivingBehaviorCloningModelFactory",
        "kwargs": {"channels": channels, "burn_in": 8},
    }
    components["replay_store"]["kwargs"]["capacity"] = 128
    config["training"] = {
        "batch_size": 16,
        "sequence_length": 32,
        "n_step": 1,
        "gamma": rl_config["training"]["gamma"],
        "metrics_interval_updates": 25,
    }
    if rl_config["components"]["model_factory"]["class_path"].endswith(
        ":DrivingStackValueModelFactory"
    ):
        encoder = rl_config["components"]["model_factory"]["kwargs"]["encoder"]["kwargs"]
        if encoder.get("frame_stack", 4) != 4:
            raise ValueError("camera stack BC currently requires four frames")
        config["run_id"] = rl_config["run_id"].replace("stack-iqn", "stack-bc")
        components["model_factory"] = {
            "class_path": (
                "trackmaniarl.trackmania.driving_bc:DrivingStackBehaviorCloningModelFactory"
            ),
            "kwargs": {"channels": channels, "shift_padding": encoder.get("shift_padding", 4)},
        }
        config["training"]["sequence_length"] = 4
    return config
