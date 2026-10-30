"""All-lap absolute steering, throttle and brake from a QR network."""

import torch
from torch import nn

from tools.record_rl1168.features import OBS_DIM
from trackmaniarl.models.composite import CompositeModules, CompositeValueModel
from trackmaniarl.models.heads import FixedQuantileHead, FixedQuantileHeadConfig
from trackmaniarl.models.strategies import FixedQuantileStrategy
from trackmaniarl.models.temporal import IdentityTemporalCore

ACTIONS = tuple(
    (steer / 8, gas, brake)
    for gas, brake in ((1, 0), (1, 1), (0, 0), (0, 1))
    for steer in range(-8, 9)
)


class Encoder(nn.Module):
    output_dim = 256

    def __init__(self):
        super().__init__()
        self.register_buffer("spatial_frequencies", 2.0 ** torch.arange(11))
        self.network = nn.Sequential(
            nn.Linear(OBS_DIM + 22, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU()
        )

    def forward(self, states):
        if states.dtype != torch.float32 or states.ndim != 2 or states.shape[1] != OBS_DIM:
            raise ValueError("expected native-route float32 observations [B,47]")
        phase = states[:, -2:-1] * self.spatial_frequencies * 2 * torch.pi
        return self.network(torch.cat((states, phase.sin(), phase.cos()), dim=-1))


def build_model():
    return CompositeValueModel(
        CompositeModules(
            Encoder(),
            IdentityTemporalCore(256),
            FixedQuantileHead(FixedQuantileHeadConfig(256, len(ACTIONS), 64, dueling=True)),
            FixedQuantileStrategy(64),
        )
    )
