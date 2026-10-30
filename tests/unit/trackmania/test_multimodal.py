from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

from tests.unit.trackmania._lidar_fixtures import _asset
from tests.unit.trackmania.test_vision import _Environment, _Frames
from trackmaniarl.builtins.features import GymnasiumObservationCollator
from trackmaniarl.core.data import Transition
from trackmaniarl.trackmania.multimodal import LidarVisionFeaturePipeline, LidarVisionSensorEncoder
from trackmaniarl.trackmania.vision_environment import VisionEnvironment


def test_dictionary_collation_uses_keys_not_insertion_order() -> None:
    collator = GymnasiumObservationCollator(
        spaces.Dict({"a": spaces.Box(0, 1, (1,)), "b": spaces.Box(0, 1, (1,))})
    )
    values = {"b": np.ones(1, dtype=np.float32), "a": np.zeros(1, dtype=np.float32)}
    result = collator.collate_observations([values, dict(reversed(list(values.items())))])
    torch.testing.assert_close(result["a"], torch.zeros(2, 1))
    with pytest.raises(ValueError, match="Dict keys"):
        collator.collate_observations([{"a": np.zeros(1)}])


def test_cpu_byte_image_collation_is_owned_exact_and_preserves_validation() -> None:
    collator = GymnasiumObservationCollator(spaces.Box(0, 2, shape=(2, 4), dtype=np.uint8))
    first = torch.arange(8, dtype=torch.uint8).reshape(4, 2).T % 3
    expected = torch.stack([first, first])
    result = collator.collate_observations([first, first])
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    first.fill_(2)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    result[0].fill_(1)
    torch.testing.assert_close(result[1], expected[1], rtol=0, atol=0)
    for invalid, message in (
        (torch.zeros(3, 4, dtype=torch.uint8), "shape"),
        (torch.zeros(2, 4), "dtype"),
        (torch.full((2, 4), 3, dtype=torch.uint8), "outside"),
    ):
        with pytest.raises(ValueError, match=message):
            collator.collate_observations([first, invalid])
    heterogeneous = GymnasiumObservationCollator(
        spaces.Box(np.array([0, 1], np.uint8), np.array([1, 2], np.uint8), dtype=np.uint8)
    )
    with pytest.raises(ValueError, match="outside"):
        heterogeneous.collate_observations([torch.tensor([0, 0], dtype=torch.uint8)])


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_cpu_float_collation_preserves_gradients_ownership_and_all_bounds(
    dtype: torch.dtype,
) -> None:
    numpy_dtype = torch.empty((), dtype=dtype).numpy().dtype
    collator = GymnasiumObservationCollator(
        spaces.Box(np.array([-1, 1], numpy_dtype), np.array([1, 2], numpy_dtype), dtype=numpy_dtype)
    )
    first = torch.tensor([-0.0, 1.5], dtype=dtype, requires_grad=True)
    expected = torch.stack([first, first]).detach().clone()
    result = collator.collate_observations([first, first])
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert result.detach().numpy().tobytes() == expected.numpy().tobytes()
    result.sum().backward()
    torch.testing.assert_close(first.grad, torch.full_like(first, 2), rtol=0, atol=0)
    with torch.no_grad():
        first.fill_(0)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    for values, message in (
        ([float("nan"), 1.5], "non-finite"),
        ([float("inf"), 1.5], "non-finite"),
        ([float("-inf"), 1.5], "non-finite"),
        ([-2, 1.5], "outside"),
        ([0, 2.5], "outside"),
    ):
        with pytest.raises(ValueError, match=message):
            collator.collate_observations([torch.tensor(values, dtype=dtype)])


def test_paired_environment_preserves_both_observations() -> None:
    environment = VisionEnvironment(_Environment(), _Frames(), include_telemetry=True)
    first, info = environment.reset(seed=17)
    assert first["telemetry"] is None
    assert np.all(first["images"] == 1)
    assert info["seed"] == 17
    second, reward, terminated, truncated, info = environment.step(4)
    assert second["telemetry"] is None
    assert np.all(second["images"] == 2)
    assert (reward, terminated, truncated) == (2.0, True, False)
    assert info["action"] == 4


def test_fusion_pipeline_resets_history_and_collates_without_advancing_it(tmp_path: Path) -> None:
    pipeline = LidarVisionFeaturePipeline(
        {"geometry_path": str(_asset(tmp_path))},
        {"width": 8, "height": 8, "frame_stack": 2},
    )
    raw = pipeline.synthetic_observation()
    pipeline.transform_observation(raw)
    raw["images"].fill(255)
    observation = pipeline.transform_observation(raw)
    torch.testing.assert_close(observation["images"][0], torch.zeros(8, 8))
    before = tuple(pipeline.vision._frames)
    transition = Transition(observation, 0, 1.0, observation, False, True)
    batch = pipeline.collate([transition, transition])
    assert batch["observations"]["images"].shape == (2, 2, 8, 8)
    assert tuple(pipeline.vision._frames) == before
    pipeline.reset_episode()
    torch.testing.assert_close(pipeline.transform_observation(raw)["images"], torch.ones(2, 8, 8))
    pipeline.set_evaluation_map(
        SimpleNamespace(geometry_path=_asset(tmp_path), expected_map_uid="trackmaniarl-test")
    )
    assert not pipeline.vision._frames


def test_fusion_encoder_preserves_sequence_axes_and_rejects_mismatched_sensors() -> None:
    encoder = LidarVisionSensorEncoder(
        {"telemetry_dim": 6, "hidden_dim": 8, "output_dim": 8}, channels=2, output_dim=8
    )
    observation = {
        "lidar": {
            "lidar": torch.rand(2, 3, 4, 12),
            "lidar_mask": torch.ones(2, 3, 12),
            "telemetry": torch.rand(2, 3, 6),
        },
        "images": torch.rand(2, 3, 2, 8, 8),
    }
    assert encoder(observation).shape == (2, 3, 8)
    observation["images"] = torch.rand(2, 4, 2, 8, 8)
    with pytest.raises(ValueError, match="batch and time"):
        encoder(observation)
