"""Schematic rendering and SDK integration; fixtures are not new engine evidence."""

from dataclasses import replace
from math import prod
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.unit.experiments.test_graph_iqn_v5 import _geometry
from tests.unit.models.test_full_qr_committed_terminal import _synthetic_frame
from tests.unit.models.test_full_qr_environment import frame, setup
from tests.unit.models.test_full_qr_terminal_capture import _decode
from trackmaniarl.core.data import Transition
from trackmaniarl.trackmania.geometry import BoundaryGeometry
from trackmaniarl.trackmania.native_headless.full_qr.native_features import (
    NativeGoalTerminalFrame,
    NativeGraphFrame,
)
from trackmaniarl.trackmania.native_headless.ribbon_renderer import (
    RibbonCameraConfig,
    RibbonRenderer,
)
from trackmaniarl.trackmania.native_headless.vision import (
    NativeSchematicVisionPipeline,
    RibbonStorage,
    RibbonVisionFeaturePipeline,
    RibbonVisionSensorEncoder,
)
from trackmaniarl.trackmania.vision import PreparedVisionFeaturePipeline
from trackmaniarl.trackmania.vision_models import VisionSensorEncoder


def _pipeline(tmp_path: Path, *, compact: bool = False) -> NativeSchematicVisionPipeline:
    return NativeSchematicVisionPipeline(
        _geometry(tmp_path / "road.npz"),
        "test-map",
        camera={"width": 32, "height": 32},
        vision={"width": 32, "height": 32, "frame_stack": 2},
        compact=compact,
    )


def test_perspective_is_pose_sensitive_and_does_not_draw_lidar_extension(tmp_path: Path) -> None:
    geometry = BoundaryGeometry(_geometry(tmp_path / "road.npz"))
    renderer = RibbonRenderer(geometry, {"width": 32, "height": 32, "downward_pitch_degrees": 0})
    position, forward = np.array([2, 0, 0]), np.array([1, 0, 0])
    first = renderer.render(position, forward)
    assert first.shape == (32, 32, 3)
    assert first.dtype == np.uint8
    assert np.any(np.all(first == renderer.road_rgb, axis=-1))
    # The camera is over the center of a straight symmetric strip.
    np.testing.assert_array_equal(first, first[:, ::-1])
    assert not np.array_equal(first, renderer.render(position + np.array([0, 0, 2]), forward))
    assert not np.array_equal(first, renderer.render(position, np.array([0, 0, 1])))
    assert np.all(renderer.render(np.array([620, 0, 0]), forward) == renderer.background_rgb)
    # Height is part of the recorded surface, not collapsed onto a flat map.
    geometry.left[:, 1] += 5
    geometry.right[:, 1] += 5
    raised = RibbonRenderer(geometry, renderer.config)
    assert not np.array_equal(first, raised.render(position, forward))


def test_near_clipping_depth_order_and_degenerate_pose(tmp_path: Path) -> None:
    renderer = _pipeline(tmp_path).renderer
    # Two triangles cover the same pixels at different depths. Changing their
    # submission order must not change occlusion (perspective depth test).
    far = np.array([[-4, -4, 8], [4, -4, 8], [0, 4, 8]], dtype=float)
    near = far / 2
    outputs = []
    for triangles, colors in (
        ([far, near], [[20, 0, 0], [0, 30, 0]]),
        ([near, far], [[0, 30, 0], [20, 0, 0]]),
    ):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        depth = np.zeros((32, 32))
        for triangle, color in zip(triangles, colors, strict=True):
            renderer._rasterize(triangle[None], image, depth, np.array([color]))
        outputs.append(image)
    np.testing.assert_array_equal(*outputs)
    np.testing.assert_array_equal(outputs[0][16, 16], [0, 30, 0])
    # Camera on the strip crosses the near plane; vertical orientation stays finite.
    assert renderer.render(np.zeros(3), np.array([0, 1, 0])).dtype == np.uint8
    with pytest.raises(ValueError, match="nonzero"):
        renderer.render(np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        renderer.render(np.array([np.nan, 0, 0]), np.ones(3))
    with pytest.raises(ValueError, match="near plane"):
        RibbonCameraConfig(near_m=100, far_m=1)


def test_frustum_blocks_preserve_all_pixels_including_crossings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    optimized = _pipeline(tmp_path).renderer
    reference = RibbonRenderer(BoundaryGeometry(tmp_path / "road.npz"), optimized.config)
    monkeypatch.setattr(
        reference,
        "_visible_triangles",
        lambda camera, basis: (reference._triangles, reference._colors),
    )
    for position, forward in (
        ([2, 0, 0], [1, 0, 0]),
        ([450, 2, 3], [-1, 0, 0]),
        ([100, 0, 0], [0, 1, 0]),
        ([100, -4, -2], [1, 0.3, 0.2]),
        ([700, 0, 0], [-1, 0, 0]),
    ):
        args = np.asarray(position), np.asarray(forward)
        indexed = optimized.render_indexed(*args)
        np.testing.assert_array_equal(optimized.render(*args), reference.render(*args))
        np.testing.assert_array_equal(
            optimized.render(*args), np.asarray(optimized.rgb_palette)[indexed]
        )


def test_optional_compiled_raster_preserves_clipping_depth_and_palette(tmp_path: Path) -> None:
    pytest.importorskip("numba", reason="optional native-vision accelerator")
    geometry = BoundaryGeometry(_geometry(tmp_path / "road.npz"))
    reference = RibbonRenderer(geometry, {"width": 32, "height": 32})
    compiled = RibbonRenderer(geometry, {"width": 32, "height": 32, "raster_backend": "numba"})
    for position, forward in (
        ([2, 0, 0], [1, 0, 0]),
        ([100, -3, 0], [1, 0.1, 0]),
        ([100, 0, 0], [0, 1, 0]),
        ([450, 2, 3], [-1, 0, 0]),
        ([700, 0, 0], [-1, 0, 0]),
    ):
        args = np.asarray(position), np.asarray(forward)
        np.testing.assert_array_equal(reference.render(*args), compiled.render(*args))
    triangle = np.array([[[-4, -4, 8], [4, -4, 8], [0, 4, 8]]], dtype=np.float64)
    triangles = np.concatenate([triangle, triangle / 2, triangle / 2])
    outputs = []
    for renderer in (reference, compiled):
        image, depth = np.zeros((32, 32), np.uint8), np.zeros((32, 32), np.float64)
        renderer._rasterize(triangles, image, depth, np.array([1, 1, 2], dtype=np.uint8))
        outputs.append((image, depth))
    np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])


@pytest.mark.parametrize("grayscale", [True, False])
@pytest.mark.parametrize("storage", ["uint8", "base3"])
def test_compact_pixels_decode_exactly_with_identical_cnn_outputs_and_gradients(
    tmp_path: Path,
    *,
    grayscale: bool,
    storage: RibbonStorage,
) -> None:
    path = _geometry(tmp_path / "road.npz")
    config = {"width": 32, "height": 32, "grayscale": grayscale, "frame_stack": 2}
    full = NativeSchematicVisionPipeline(
        path, "test-map", camera={"width": 32, "height": 32}, vision=config
    )
    compact = NativeSchematicVisionPipeline(
        path,
        "test-map",
        camera={"width": 32, "height": 32},
        vision=config,
        compact=True,
        storage=storage,
    )
    sdk = RibbonVisionFeaturePipeline(config, storage=storage)
    compact_encoder = RibbonVisionSensorEncoder(config, output_dim=8, hidden_dim=8, storage=storage)
    full_encoder = VisionSensorEncoder(channels=2 if grayscale else 6, output_dim=8, hidden_dim=8)
    full_encoder.load_state_dict(compact_encoder.state_dict(), strict=True)
    observations, indices = [], []
    for time_ms in (0, 50, 100):
        channels = list(frame(time_ms).channels)
        channels[3] = time_ms / 60
        native = NativeGraphFrame(tuple(channels))
        expected = full.transform_observation(native)["images"]
        ids = sdk.transform_observation(compact.transform_observation(native))
        torch.testing.assert_close(compact_encoder.decode(ids), expected, rtol=0, atol=0)
        np.testing.assert_array_equal(compact.last_rgb, full.last_rgb)
        observations.append(expected)
        indices.append(ids)
    actual = compact_encoder(torch.stack(indices))
    expected = full_encoder(torch.stack(observations))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    expected.sum().backward()
    for left, right in zip(compact_encoder.parameters(), full_encoder.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    pixel_count = prod((2, 32, 32))
    assert indices[0].numel() == (pixel_count if storage == "uint8" else (pixel_count + 4) // 5)
    compact.reset_episode()
    reset = compact.transform_observation(frame(0))["images"]
    decoded = compact_encoder.decode(reset)
    torch.testing.assert_close(decoded[: decoded.shape[0] // 2], decoded[decoded.shape[0] // 2 :])
    bad = torch.full_like(reset, 3 if storage == "uint8" else 243)
    with pytest.raises(ValueError, match=r"0\.\."):
        sdk.transform_observation({"images": bad})
    if storage == "base3":
        bad = reset.clone()
        bad[-1] = 3 ** (pixel_count % 5)
        with pytest.raises(ValueError, match="padding"):
            sdk.transform_observation({"images": bad})


def test_images_are_paired_with_exact_native_clock_and_history_is_episode_local(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(tmp_path)
    initial = pipeline.transform_observation(frame(0))["images"]
    assert pipeline.last_frame_time_ms == 0
    assert pipeline.last_rgb is not None
    assert not pipeline.last_rgb.flags.writeable
    torch.testing.assert_close(initial[0], initial[1])
    moved = list(frame(50).channels)
    moved[3] = 2.0
    following = pipeline.transform_observation(NativeGraphFrame(tuple(moved)))["images"]
    torch.testing.assert_close(following[0], initial[1])
    assert not torch.equal(following[0], following[1])
    with pytest.raises(ValueError, match="50ms"):
        pipeline.transform_observation(frame(150))
    pipeline.reset_episode()
    torch.testing.assert_close(pipeline.transform_observation(frame(0))["images"], initial)


@pytest.mark.parametrize("compact", [False, True])
def test_committed_goal_uses_physics_sample_not_score_and_refuses_more_frames(
    tmp_path: Path,
    *,
    compact: bool,
) -> None:
    pipeline = _pipeline(tmp_path, compact=compact)
    actual = _synthetic_frame()  # Synthetic terminal record, 52790ms sample / 52792ms score.
    # Advance only this unit test's preceding clock; this does not qualify a new trace.
    for time_ms in range(0, 52751, 50):
        pipeline.transform_observation(frame(time_ms))
    observation = pipeline.transform_committed_terminal(actual)
    assert pipeline.last_frame_time_ms == 52790
    assert observation["images"].shape == (2, 32, 32)
    with pytest.raises(ValueError, match="50ms"):
        pipeline.transform_observation(frame(52800))
    with pytest.raises(TypeError, match="partial Goal"):
        pipeline.transform_terminal(actual)  # type: ignore[arg-type]
    pipeline.reset_episode()
    partial = _decode()
    # Synthetic adherence binding for this unit path, not fresh event qualification.
    adherence = replace(
        actual.adherence, native_t=partial.vehicle.native_t, sequence=partial.vehicle.sequence
    )
    terminal = NativeGoalTerminalFrame(partial, adherence)
    for time_ms in range(0, partial.previous_native_t - 2199, 50):
        pipeline.transform_observation(frame(time_ms))
    pipeline.transform_terminal(terminal)
    assert pipeline.last_frame_time_ms == partial.vehicle.native_t - 2200 == 69790


@pytest.mark.parametrize("compact", [False, True])
def test_same_clock_committed_score_does_not_append_an_image(
    tmp_path: Path, *, compact: bool
) -> None:
    pipeline = _pipeline(tmp_path, compact=compact)
    actual = _synthetic_frame()
    capture = actual.capture
    committed = replace(
        actual,
        adherence=replace(actual.adherence, native_t=2200),
        capture=replace(
            capture,
            previous_native_t=2200,
            vehicle=replace(capture.vehicle, native_t=2200),
            goal=replace(capture.goal, native_t=2200, last_goal_native_t=2202),
            fact=replace(
                capture.fact, event_native_t=2202, callback_native_t=2190, dispatch_native_t=2200
            ),
        ),
    )
    first = pipeline.transform_observation(NativeGraphFrame(committed.channels))
    assert pipeline.transform_committed_terminal(committed) is first


def test_prepared_sdk_images_do_not_restack_and_rewards_are_unchanged(tmp_path: Path) -> None:
    vision_environment, _, _ = setup()
    telemetry_environment, _, _ = setup()
    vision_environment.pipeline = _pipeline(tmp_path)
    sdk = PreparedVisionFeaturePipeline({"width": 32, "height": 32, "frame_stack": 2})
    initial, _ = vision_environment.reset()
    telemetry_environment.reset()
    image, reward, terminated, truncated, info = vision_environment.step(39)
    _, expected_reward, expected_terminated, expected_truncated, expected_info = (
        telemetry_environment.step(39)
    )
    assert (reward, terminated, truncated) == (
        expected_reward,
        expected_terminated,
        expected_truncated,
    )
    assert info["race_time_ms"] == expected_info["race_time_ms"] == 50
    prepared = sdk.transform_observation(image)
    assert sdk.transform_observation(image) is prepared
    transition = Transition(
        sdk.transform_observation(initial), 39, reward, prepared, terminated, truncated
    )
    assert sdk.collate([transition])["observations"].shape == (1, 2, 32, 32)
    for invalid in (
        initial["images"],
        {"images": torch.zeros(2, 8, 8)},
        {"images": torch.full((2, 32, 32), float("nan"))},
    ):
        with pytest.raises(ValueError, match="prepared"):
            sdk.transform_observation(invalid)
    vision_environment.close()
    telemetry_environment.close()
