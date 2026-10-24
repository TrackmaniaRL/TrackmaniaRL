"""Actual literal decoder fixture and explicit reader fault injection.

Synthetic faults and unused reference-array fillers are unit-test inputs only.
They are never presented as actual game telemetry or training transitions.
"""

import json
from dataclasses import asdict
from pathlib import Path
from struct import pack, unpack
from typing import Any

import numpy as np
import pytest
import torch

from experiments.full_qr.windows_graph import (
    WindowsComparisonGraphPipeline,
    WindowsGraphReader,
    WindowsObservationUnavailableError,
)
from experiments.full_qr.windows_terminal import WindowsModuleContext, WindowsTerminalReader
from tests.unit.experiments.test_graph_iqn_v5 import _geometry
from trackmaniarl.experiments.graph_iqn_v5 import BoundaryGraphFeaturePipelineV5
from trackmaniarl.trackmania.native_headless.full_qr.native_features import GRAPH_FIELDS


def fixture() -> tuple[dict[str, Any], dict[tuple[int, int], bytes], WindowsGraphReader]:
    path = Path(__file__).parents[2] / "fixtures/full_qr/windows_full_graph320.json"
    data = json.loads(path.read_text())
    reads = {(row["address"], row["bytes"]): bytes.fromhex(row["hex"]) for row in data["reads"]}
    base = data["modules"]["trackmania.exe"]["base"]
    terminal = WindowsTerminalReader(
        lambda address, size: reads[address, size],
        WindowsModuleContext(base, data["modules"]["openplanet.dll"]["base"]),
    )
    return data, reads, WindowsGraphReader(terminal)


def test_actual_full23_decoder_preserves_literal_source_values() -> None:
    data, _, reader = fixture()
    observation = reader.sample()
    assert json.loads(json.dumps(asdict(observation))) == data["observation"]
    assert observation.adherence_owner != 0  # Actual typed arena, not a null-default claim.
    unused = sorted(set(range(33)) - set(GRAPH_FIELDS))
    assert np.isnan(observation.feature_carrier()[unused]).all()


def test_same_v5_computations_preserve_historical_outputs_and_history(tmp_path: Path) -> None:
    _, _, reader = fixture()
    observation = reader.sample()
    geometry = _geometry(tmp_path / "unit-geometry.npz")
    comparison = WindowsComparisonGraphPipeline(geometry, "test-map")
    historical = BoundaryGraphFeaturePipelineV5(geometry, "test-map")
    for filler in (1234.0, -987.0):
        actual = comparison.transform_observation(observation)
        reference = observation.feature_carrier()
        reference[np.isnan(reference)] = filler  # Test-only unrelated API fields.
        expected = historical.transform_observation(reference)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_absent_arena_entity_never_falls_back_to_default_adherence() -> None:
    data, reads, reader = fixture()
    arena = data["observation"]["adherence_owner"]
    pool = unpack("<Q", reads[arena + 0x7C8, 8])[0]
    header = reads[pool + 0x8E0, 0x38]
    capacity = unpack("<I", header[4:8])[0]
    flags = unpack("<Q", header[0x20:0x28])[0]
    slot = data["observation"]["entity_id"] & (capacity - 1)
    address = flags + (slot >> 4) * 4
    word = unpack("<I", reads[address, 4])[0]
    reads[address, 4] = pack("<I", word | (2 << ((slot & 15) * 2)))
    with pytest.raises(ValueError, match="absent from the arena physics pool"):
        reader.sample()


def test_arena_lookup_requires_the_same_actual_player_body_generation() -> None:
    data, reads, reader = fixture()
    arena = data["observation"]["adherence_owner"]
    pool = unpack("<Q", reads[arena + 0x7C8, 8])[0]
    header = reads[pool + 0x8E0, 0x38]
    capacity = unpack("<I", header[4:8])[0]
    values = unpack("<Q", header[0x30:0x38])[0]
    slot = data["observation"]["entity_id"] & (capacity - 1)
    address = values + slot * 16
    reference = reads[address, 16]
    reads[address, 16] = reference[:4] + pack("<I", 99) + reference[8:]
    with pytest.raises(ValueError, match="disagrees with the current player/body"):
        reader.sample()


def test_missing_async_state_yields_no_observation() -> None:
    data, reads, reader = fixture()
    reads[data["observation"]["vis"] + 0x130, 8] = b"\0" * 8
    with pytest.raises(WindowsObservationUnavailableError, match="AsyncState is unavailable"):
        reader.sample()


def test_bounded_source_wait_rebinds_a_null_async_state_before_publication() -> None:
    data, _, reader = fixture()
    original = reader.terminal.read
    visits = 0
    guards = []

    def unavailable_once(address: int, size: int) -> bytes:
        nonlocal visits
        if address == data["observation"]["vis"] + 0x130 and size == 8:
            visits += 1
            if visits == 1:
                return b"\0" * 8
        return original(address, size)

    reader.terminal.read = unavailable_once
    observation = reader.wait_sample(guard=lambda: guards.append(True))
    assert observation.channels == tuple(data["observation"]["channels"])
    assert len(guards) == 2


def test_source_wait_never_suppresses_a_position_mismatch() -> None:
    data, reads, reader = fixture()
    player = data["observation"]["terminal"]["player"]
    cache = reads[player + 0xE24, 0xFC]
    reads[player + 0xE24, 0xFC] = b"\0" * 12 + cache[12:]
    guards = []
    with pytest.raises(ValueError, match="positions do not match"):
        reader.wait_sample(guard=lambda: guards.append(True))
    assert guards == [True]


@pytest.mark.parametrize("recovers", [True, False])
def test_container_publication_discards_entire_read_and_remains_bounded(*, recovers: bool) -> None:
    data, reads, reader = fixture()
    container = next(
        (address, size)
        for address, size in reads
        if size == 16
        and address > 4096
        and reads[address, size][:4] != bytes(4)
        and any(
            size2 == 8 and unpack("<Q", value)[0] + 0x210 == address
            for (_, size2), value in reads.items()
        )
    )
    original = reader.terminal.read
    visits = 0
    guards = []

    def changed_container(address: int, size: int) -> bytes:
        nonlocal visits
        raw = original(address, size)
        if (address, size) == container:
            visits += 1
            if visits % 2 == 0 and (not recovers or visits == 2):
                return bytes([raw[0] ^ 1]) + raw[1:]
        return raw

    reader.terminal.read = changed_container
    if recovers:
        observation = reader.wait_sample(guard=lambda: guards.append(True))
        assert observation.channels == tuple(data["observation"]["channels"])
        assert len(guards) == 2
    else:
        with pytest.raises(TimeoutError, match="complete current comparison source frame"):
            reader.wait_sample(guard=lambda: guards.append(True), timeout_s=0.001)


def test_changed_native_registration_is_refused_before_field_decoding() -> None:
    data, reads, _ = fixture()
    base = data["modules"]["trackmania.exe"]["base"]
    reads[base + 0x7261A0, 283] = b"\0" * 283
    with pytest.raises(ValueError, match="source function changed"):
        WindowsGraphReader(
            WindowsTerminalReader(
                lambda address, size: reads[address, size],
                WindowsModuleContext(base, data["modules"]["openplanet.dll"]["base"]),
            )
        )
