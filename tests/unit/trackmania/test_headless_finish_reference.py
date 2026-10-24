"""Source controls must map exactly to the existing N1 action table."""

import hashlib
import json
import struct

import pytest

from scripts import headless_finish_reference as runner
from trackmaniarl.trackmania.native_headless._runtime.features import action_sgb
from trackmaniarl.trackmania.native_headless._runtime.outcome import MAP_UID


def source() -> dict:
    return {
        "metadata": {"map_uid_literals": [MAP_UID]},
        "input_archives": [
            {
                "packets": [
                    {
                        "tick_index": index,
                        "race_time_ms": -1510 + index * 10,
                        "steer_normalized": -1.0 if index % 2 else 1.0,
                        "accel": 1,
                        "brake": 0,
                    }
                    for index in range(1923)
                ]
            }
        ],
    }


def checked_plan(monkeypatch: pytest.MonkeyPatch, document: dict) -> tuple[dict, dict]:
    raw = json.dumps(document).encode()
    # Unit fixture admission replaces only the fixture digest; the production
    # source digest and the canonical action table are never changed on disk.
    monkeypatch.setattr(runner, "EXPORT_SHA256", hashlib.sha256(raw).hexdigest())
    return runner.source_actions(raw)


def test_all_driving_packets_keep_exact_n1_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    document = source()
    plan, proof = checked_plan(monkeypatch, document)
    assert len(plan) == proof["finite_driving_source_actions"] == 1772
    assert proof["action_quantization_error"] == 0
    for index, action in plan.items():
        packet = document["input_archives"][0]["packets"][index]
        assert struct.pack("<3f", *action_sgb(action)[0]) == struct.pack(
            "<3f", packet["steer_normalized"], 1, 0
        )


def test_unsigned_or_changed_source_is_rejected_before_planning() -> None:
    with pytest.raises(ValueError, match="exact supplied Jimmy source"):
        runner.source_actions(json.dumps(source()).encode())


def test_nonrepresentable_steer_is_not_quantized(monkeypatch: pytest.MonkeyPatch) -> None:
    document = source()
    document["input_archives"][0]["packets"][152]["steer_normalized"] = 0.2
    with pytest.raises(ValueError, match="not exactly available"):
        checked_plan(monkeypatch, document)


def test_packet_clock_gap_cannot_shift_the_expert_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    document = source()
    document["input_archives"][0]["packets"][152]["race_time_ms"] += 10
    with pytest.raises(ValueError, match="consecutive"):
        checked_plan(monkeypatch, document)


def test_wrong_map_is_not_adopted(monkeypatch: pytest.MonkeyPatch) -> None:
    document = source()
    document["metadata"]["map_uid_literals"] = ["another-map"]
    with pytest.raises(ValueError, match="map UID"):
        checked_plan(monkeypatch, document)


def test_truncated_source_does_not_create_a_fabricated_neutral_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = source()
    document["input_archives"][0]["packets"].pop()
    with pytest.raises(ValueError, match="packet count"):
        checked_plan(monkeypatch, document)
