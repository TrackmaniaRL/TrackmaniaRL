"""Real-image QR-DQN model using the project's existing quantile learner.

The old telemetry PPO checkpoint is retained, not silently loaded into this CNN.
Optional BC needs newly paired real images and applied actions.
"""
import torch
from torch import nn
from trackmaniarl.models.composite import CompositeModules, CompositeValueModel
from trackmaniarl.models.heads import FixedQuantileHead, FixedQuantileHeadConfig
from trackmaniarl.models.strategies import FixedQuantileStrategy
from trackmaniarl.models.temporal import IdentityTemporalCore


# Absolute steering everywhere, including the early turns. No frozen baseline.
ACTIONS = tuple((steer / 8, accel, brake)
                for accel, brake in ((1, 0), (1, 1), (0, 0), (0, 1))
                for steer in range(-8, 9))


class Encoder(nn.Module):
    output_dim = 256

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU(), nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(),
        )

    def forward(self, frames):
        if frames.dtype != torch.uint8 or tuple(frames.shape[1:]) != (4, 84, 84):
            raise ValueError('expected uint8 real camera frames [B,4,84,84]')
        return self.network(frames.float() / 255)


def build_model():
    return CompositeValueModel(CompositeModules(
        Encoder(), IdentityTemporalCore(256),
        FixedQuantileHead(FixedQuantileHeadConfig(256, len(ACTIONS), 64, dueling=True)),
        FixedQuantileStrategy(64),
    ))
