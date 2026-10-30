"""Historical V6 shape adapter, preserved for the full QR preset."""

from collections.abc import Mapping

import torch

from trackmaniarl.experiments.graph_iqn_v6 import IncidentGatedTrackGnnSimbaEncoderV6


class BatchedIncidentEncoder(IncidentGatedTrackGnnSimbaEncoderV6):
    """Keep V6 computation and parameters, accepting arbitrary leading axes."""

    def forward(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        leading = observation["physics"].shape[:-1]
        if len(leading) == 1:
            return super().forward(observation)
        event_dims = {"physics": 1, "track": 2, "context": 1, "recovery": 1}
        flattened = {
            key: value.reshape(-1, *value.shape[-event_dims[key] :])
            for key, value in observation.items()
        }
        return super().forward(flattened).reshape(*leading, self.output_dim)
