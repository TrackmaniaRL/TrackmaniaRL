"""Feature-computation equivalence, not a claim of Windows/native telemetry parity."""

from dataclasses import replace
from pathlib import Path
from struct import pack_into

import numpy as np
import pytest
import torch

from tests.unit.experiments.test_graph_iqn_v5 import _frame, _geometry
from trackmaniarl.experiments.graph_iqn_v5 import BoundaryGraphFeaturePipelineV5
from trackmaniarl.trackmania.native_headless.full_qr.native_features import (
    GRAPH_FIELDS,
    NativeGraphFeaturePipeline,
    NativeGraphFrame,
)
from trackmaniarl.trackmania.native_headless.full_qr.vehicle_capture import VehicleCapture


def test_native_features_preserve_all_v5_values_and_causal_reset(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path / "g.npz")
    native = NativeGraphFeaturePipeline(geometry, "test-map")
    historical = BoundaryGraphFeaturePipelineV5(geometry, "test-map")
    initial = None
    for time_ms, position, speed in [(0, (100, 0), 80), (50, (103, 1), 60), (100, (107, 2), 30)]:
        values = _frame(time_ms, position, speed)
        values[[19, 20, 21, 22, 27, 28]] = (0.3, 0.5, 0.7, 0.9, 3, 120)
        observation = NativeGraphFrame(tuple(float(values[i]) for i in GRAPH_FIELDS))
        actual = native.transform_observation(observation)
        # Poison unused API channels with arbitrary finite values. They must not
        # influence the historical computation, while native leaves them absent.
        unused = sorted(set(range(33)) - set(GRAPH_FIELDS))
        values[unused] = np.arange(len(unused), dtype=np.float32) + 1234
        expected = historical.transform_observation(values)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        if initial is None:
            initial = actual
            first = observation
    native.reset_episode()
    restored = native.transform_observation(first)
    assert initial is not None
    for key in initial:
        torch.testing.assert_close(restored[key], initial[key], rtol=0, atol=0)


def test_missing_consumed_channels_and_skipped_decisions_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="23"):
        NativeGraphFrame((0.0,))
    values = _frame(0, (100, 0), 80)
    channels = tuple(float(values[i]) for i in GRAPH_FIELDS)
    with pytest.raises(ValueError, match="finite"):
        NativeGraphFrame((float("nan"), *channels[1:]))
    pipeline = NativeGraphFeaturePipeline(_geometry(tmp_path / "g.npz"), "test-map")
    pipeline.transform_observation(NativeGraphFrame(channels))
    with pytest.raises(ValueError, match="exactly50ms"):
        pipeline.transform_observation(NativeGraphFrame((100.0, *channels[1:])))
    # Rejection precedes history mutation, so the correct next frame still works.
    pipeline.transform_observation(NativeGraphFrame((50.0, *channels[1:])))


def test_native_vendor_fields_preserved_and_unknown_adherence_is_not_defaulted() -> None:
    cache, visual = bytearray(140), bytearray(864)
    pack_into("<3f", visual, 0x5C, 3, 0, 4)
    pack_into("<fI", cache, 0x38, 8000, 3)
    pack_into("<I", cache, 0x4C, 2)
    pack_into("<I", cache, 0x5C, 120)
    capture = VehicleCapture(2200, 2, 10, bytes(cache), bytes(visual), bytes(140), 0)
    frame = NativeGraphFrame.from_capture(capture)
    named = dict(zip(GRAPH_FIELDS, frame.channels, strict=True))
    assert named[16] == 5.0
    assert named[27] == 2.0
    assert named[28] == 120.0
    assert named[29] == 1.0
    with pytest.raises(ValueError, match="adherence"):
        NativeGraphFrame.from_capture(replace(capture, global_owner=4096))
