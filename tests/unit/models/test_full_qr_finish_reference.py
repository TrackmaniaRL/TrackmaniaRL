"""Actual paired UI403/validator405/Goal406 evidence; no invented event times."""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.finish_reference import (
    QualifiedGoalFinishReader,
    qualify_reference,
)


def fixture() -> tuple[dict[str, bytes], dict[str, str]]:
    root = Path(__file__).parents[2] / "fixtures/full_qr/paired_finish403"
    context = json.loads((root / "context.json").read_text())
    for name, pin in context["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == pin
    return {
        role: (root / (role + (".bin" if role == "goal" else ".json"))).read_bytes()
        for role in ("client", "observer", "plain", "observed", "goal")
    }, context["asset_pins"]


def test_actual_same_run_ui_and_original_goal_have_exact_34268_timestamp() -> None:
    evidence, pins = fixture()
    reference = qualify_reference(evidence, pins)
    assert reference.race_time_ms == reference.ui_observed_race_ms == 34268
    assert reference.race_origin == 2200
    assert reference.goal_waypoint_id == 0


@pytest.mark.parametrize("field", ["IsValid", "Inputs", "ValidatedResult"])
def test_original_validator_refusal_input_or_score_changes_cannot_qualify(field: str) -> None:
    evidence, pins = fixture()
    records = json.loads(evidence["observed"])
    replacements = {"IsValid": False, "Inputs": "different", "ValidatedResult": None}
    records[0][field] = replacements[field]
    evidence["observed"] = json.dumps(records).encode()
    with pytest.raises(ValueError, match=r"identical original|null result"):
        qualify_reference(evidence, pins)


def test_ui_timing_cannot_be_replaced_with_a_later_or_assumed_finish() -> None:
    evidence, pins = fixture()
    client = json.loads(evidence["client"])
    client["ui_timing_observations"][-1]["race_ms"] += 1
    evidence["client"] = json.dumps(client).encode()
    with pytest.raises(ValueError, match="actual UI transition"):
        qualify_reference(evidence, pins)


def test_unclosed_observer_or_foreign_process_cannot_qualify() -> None:
    evidence, pins = fixture()
    observer = json.loads(evidence["observer"])
    observer["cleanup"]["remainingOwned"] = [observer["identity"]["worker"]]
    evidence["observer"] = json.dumps(observer).encode()
    with pytest.raises(ValueError, match="clean original observer"):
        qualify_reference(evidence, pins)
    observer["cleanup"]["remainingOwned"] = []
    observer["identity"]["worker"]["pid"] += 1
    evidence["observer"] = json.dumps(observer).encode()
    with pytest.raises(ValueError, match="identity"):
        qualify_reference(evidence, pins)


def test_same_map_name_with_different_asset_cannot_use_reference() -> None:
    evidence, pins = fixture()
    reference = qualify_reference(evidence, pins)
    with pytest.raises(ValueError, match="runtime assets"):
        QualifiedGoalFinishReader(SimpleNamespace(pins={**pins, "map": "a" * 64}), reference)


@pytest.mark.skipif(os.name != "posix", reason="private Linux held stream reader")
def test_actual_later_episode_waits_for_its_own_future_goal_timestamp(tmp_path: Path) -> None:
    evidence, pins = fixture()
    reference = qualify_reference(evidence, pins)
    root = Path(__file__).parents[2] / "fixtures/full_qr/goal_delivery380"
    context = json.loads((root / "context.json").read_text())
    pid, meta = context["pid"], context["meta"]
    receipts = tmp_path / f"goal-receipt-{pid}.bin"
    facts = tmp_path / f"goal-fact-{pid}.bin"
    raw = (root / "receipts.bin").read_bytes()
    receipts.write_bytes(raw[: ((36500 - 2190) // 10 + 1) * 96])
    facts.write_bytes((root / "facts.bin").read_bytes())
    receipts.chmod(0o600)
    facts.chmod(0o600)
    reader = QualifiedGoalFinishReader(
        SimpleNamespace(pins=pins, pid=pid, meta=meta, local_posts=tmp_path), reference
    )
    assert reader.read(36500) is False  # Its event is36505, after this boundary.
    with receipts.open("ab") as stream:
        stream.write(raw[receipts.stat().st_size :])
    assert reader.read(36550) is True
