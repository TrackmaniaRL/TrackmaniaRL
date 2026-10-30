from __future__ import annotations

import numpy as np
import torch

from examples.native_observation_pipeline import PositionVelocityFeatures, PositionVelocityPipeline
from trackmaniarl.core.data import Transition
from trackmaniarl.models.encoders import StructuredMlpSensorEncoder
from trackmaniarl.trackmania.native_headless.full_qr.native_features import NativeGraphFrame


def test_example_exposes_six_measured_position_and_velocity_channels() -> None:
    frame = NativeGraphFrame(tuple(float(value) for value in (0, *range(1, 23))))
    pipeline = PositionVelocityPipeline()
    pipeline.reset_episode()
    for transform in (
        pipeline.transform_observation,
        pipeline.transform_terminal,
        pipeline.transform_committed_terminal,
    ):
        output = transform(frame)
        np.testing.assert_array_equal(output["kinematics"], np.arange(1, 7, dtype=np.float32))
        assert output["kinematics"].dtype == np.float32

    features = PositionVelocityFeatures()
    prepared = features.transform_observation(pipeline.transform_observation(frame))
    encoder = StructuredMlpSensorEncoder({"kinematics": [6]}, output_dim=8)
    assert isinstance(prepared["kinematics"], torch.Tensor)
    assert encoder(prepared).shape == (8,)
    transition = Transition(prepared, 39, 0.0, prepared, False, False)
    batch = features.collate([transition])
    encoded = encoder(batch["observations"])
    assert encoded.shape == (1, 8)
    assert torch.isfinite(encoded).all()
