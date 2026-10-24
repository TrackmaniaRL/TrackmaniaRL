"""Synthetic native-only contracts; no filesystem, engine or external evidence I/O."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from trackmaniarl.core.data import BatchRequest
from trackmaniarl.trackmania.common_training import (
    CommonPolylineReference,
    CommonReplayStore,
    CommonSampler,
    CommonTrainingContract,
    DurationUpdateBudget,
    adapters,
    make_learner,
    native_only,
    require_paired_comparison,
)
from trackmaniarl.trackmania.common_training.contract import canonical
from trackmaniarl.trackmania.common_training.native_only import (
    NativeOnlyAdapter,
    NativeOnlyClockProof,
    native_only_contract,
)
from trackmaniarl.trackmania.native_headless._runtime.features import action_sgb


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def evidence(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    # These in-memory records exercise validation only. They are not measured
    # source artifacts and cannot satisfy the production pins without patching.
    base = Path(__file__).parent / "memory-only-fixture"
    bodies = {
        base / "server": b"synthetic server fixture",
        base / "tape": b"synthetic input tape fixture",
        base / "unit-proof": canonical(
            {
                "schema": native_only.UNIT_PROOF_SCHEMA,
                "semantics": native_only.UNIT_SEMANTICS,
                "synthetic_unit_fixture": True,
            }
        ).encode(),
    }
    server_sha, tape_sha = _sha(bodies[base / "server"]), _sha(bodies[base / "tape"])
    geometry = b'{"points":[[0,0,0],[1000,0,0]]}'
    geometry_sha = _sha(geometry)
    monkeypatch.setattr(native_only, "KNOWN_SERVER_SHA256", server_sha)
    monkeypatch.setattr(native_only, "KNOWN_TAPE_SHA256", tape_sha)
    monkeypatch.setattr(native_only, "KNOWN_REFERENCE_SHA256", geometry_sha)
    monkeypatch.setattr(adapters, "KNOWN_REFERENCE_SHA256", geometry_sha)
    manifest = {
        "schema": native_only.CLOCK_SCHEMA,
        "native_clock_qualified": True,
        "counter_source": native_only.COUNTER_SOURCE,
        "raw_interval": 10,
        "minimum_native": 2200,
        "maximum_native": native_only.LAST_CONTROLLED_NATIVE,
        "seconds_per_tick": 0.001,
        "scope": {
            "native_profile": "native-immediate-kinematics6/v7",
            "reset_profile": "neutral-race-start/v7",
            "input_schedule": adapters.SCHEDULE,
            "map_sha256": adapters.KNOWN_MAP_SHA256,
            "reference_sha256": geometry_sha,
            "server_sha256": server_sha,
            "tape_sha256": tape_sha,
        },
        "sources": {
            name: {"path": str(base / filename), "sha256": _sha(bodies[base / filename])}
            for name, filename in (
                ("server", "server"),
                ("tape", "tape"),
                ("unit_proof", "unit-proof"),
            )
        },
    }
    units = {
        "schema": native_only.UNIT_PROOF_SCHEMA,
        "semantics": native_only.UNIT_SEMANTICS,
        "synthetic_unit_fixture": True,
        "scope": copy.deepcopy(manifest["scope"]),
        "sources": {
            name: copy.deepcopy(source)
            for name, source in manifest["sources"].items()
            if name != "unit_proof"
        },
        "scale_constant": {
            "nominal_scale": {"numerator": 1, "denominator": 1000},
            "nominal_seconds_per_tick": 0.001,
            "nominal_seconds_per_interval": 0.01,
        },
        "observed_native_reference": {
            # Observed data may be shorter than the independently source-bound range.
            "observed_minimum_native": 2200,
            "observed_maximum_native": 2800,
            "controlled_maximum_from_reader_guard": native_only.LAST_CONTROLLED_NATIVE,
        },
    }
    bodies[base / "unit-proof"] = canonical(units).encode()
    manifest["sources"]["unit_proof"]["sha256"] = _sha(bodies[base / "unit-proof"])
    path = base / "clock.json"
    bodies[path] = canonical(manifest).encode()
    return {
        "manifest": manifest,
        "path": path,
        "sha256": _sha(bodies[path]),
        "bodies": bodies,
        "base": base,
        "reference": CommonPolylineReference(geometry, expected_sha256=geometry_sha),
    }


def _load(evidence: dict[str, Any], *, expected_sha256: str | None = None) -> NativeOnlyClockProof:
    # Limit the in-memory file boundary to artifact loading. Torch/pytest may
    # import their own modules later; their normal loader is not part of this mock.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "read_bytes", lambda self: evidence["bodies"][self])
        return NativeOnlyClockProof.from_artifact(
            evidence["path"], expected_sha256=expected_sha256 or evidence["sha256"]
        )


def _state(native: int, sgb: tuple[float, float, float] = (0, 0, 0)) -> dict[str, Any]:
    raw = bytearray(76)
    struct.pack_into("<10f", raw, 0, 1, 0, 0, 0, (native - 2200) / 100, 0, 0, 1, 0, 0)
    steer, gas, brake = sgb
    struct.pack_into("<4f", raw, 60, gas, brake, steer, steer)
    return {
        "schema": "tmrl.native-immediate-state/v1",
        "resetProfile": "neutral-race-start/v7",
        "inputSchedule": adapters.SCHEDULE,
        "nativeT": native,
        "native76Hex": raw.hex(),
    }


def _reset_info(proof: NativeOnlyClockProof) -> dict[str, Any]:
    return {
        "runtimeProvenance": {
            "pins": {
                "server": proof.scope["server_sha256"],
                "tape": proof.scope["tape_sha256"],
                "map": proof.scope["map_sha256"],
            }
        }
    }


def _result(  # noqa: PLR0913 -- independently varied literal endpoint fixture flags
    before: dict[str, Any], *, count: int = 5, terminal: bool = False, truncated: bool = False
) -> dict[str, Any]:
    events, after = [], before
    for index, channels in enumerate(action_sgb(39)[:count]):
        after = _state(before["nativeT"] + (index + 1) * 10, tuple(channels))
        raw = bytes.fromhex(after["native76Hex"])
        event = bytearray(256)
        event[:8] = b"TMPOST03"
        struct.pack_into("<I", event, 112, after["nativeT"])
        event[152:212], event[88:104] = raw[:60], raw[60:76]
        events.append(event.hex())
    result = {
        "observation": after,
        "rawIntervalQualified": True,
        "actualIterations": count,
        "deltaNative": count * 10,
        "terminated": terminal,
        "truncated": truncated,
        "authorization": {"nativeEventHex": events},
        "outcome": None
        if not terminal
        else {
            "qualifiedBy": "env002.full-original-validator+TMRESULT",
            "terminated": True,
            "computedResult": {"NbCheckpoints": 1},
        },
    }
    result["rawTransitionSha256"] = _sha(
        canonical(
            {"kind": "raw-transition", "before": before, "action": 39, "result": result}
        ).encode()
    )
    return result


def test_independent_native_clock_loads_all_hashes_and_extends_beyond_windows_scope(
    evidence: dict[str, Any],
) -> None:
    proof = _load(evidence)
    proof.require_interval(3000, 3050)
    proof.require_interval(42180, 42190)
    assert proof.clock_spec().elapsed(3000, 3050) == pytest.approx(0.05)
    assert proof.clock_spec().semantics == native_only.NATIVE_ONLY_CLOCK_SEMANTICS
    assert not hasattr(proof, "race_milliseconds")
    with pytest.raises(ValueError, match="scope"):
        proof.require_interval(42190, 42200)
    with pytest.raises(ValueError, match="grid"):
        proof.validate_timestamp(2201)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema", "tmrl.physical-sample-clock-alignment/v1"),
        ("native_clock_qualified", False),
        ("raw_interval", True),
        ("seconds_per_tick", 1),
        ("maximum_native", 42200),
    ],
)
def test_legacy_unqualified_or_out_of_scope_clock_is_rejected(
    evidence: dict[str, Any], key: str, value: Any
) -> None:
    changed = {**evidence["manifest"], key: value}
    raw = canonical(changed).encode()
    evidence["bodies"][evidence["path"]] = raw
    with pytest.raises(ValueError, match=r"clock|scale|scope"):
        _load(evidence, expected_sha256=_sha(raw))


def test_actual_source_byte_changes_and_missing_unit_evidence_are_rejected(
    evidence: dict[str, Any],
) -> None:
    evidence["bodies"][evidence["base"] / "server"] += b"changed"
    with pytest.raises(ValueError, match="SHA-256"):
        _load(evidence)
    del evidence["manifest"]["sources"]["unit_proof"]
    raw = canonical(evidence["manifest"]).encode()
    evidence["bodies"][evidence["path"]] = raw
    with pytest.raises(ValueError, match="unit proof"):
        _load(evidence, expected_sha256=_sha(raw))


def test_rehashed_windows_unit_document_cannot_replace_native_source_evidence(
    evidence: dict[str, Any],
) -> None:
    unit_path = evidence["base"] / "unit-proof"
    unit = b'{"schema":"tmrl.physical-sample-clock-alignment/v1"}'
    evidence["bodies"][unit_path] = unit
    evidence["manifest"]["sources"]["unit_proof"]["sha256"] = _sha(unit)
    raw = canonical(evidence["manifest"]).encode()
    evidence["bodies"][evidence["path"]] = raw
    with pytest.raises(ValueError, match="native nominal simulation unit"):
        _load(evidence, expected_sha256=_sha(raw))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (("scope", "map_sha256"), "f" * 64),
        (("sources", "server", "sha256"), "f" * 64),
        (("scale_constant", "nominal_scale", "numerator"), True),
        (("scale_constant", "nominal_scale", "denominator"), 100),
        (("scale_constant", "nominal_seconds_per_tick"), 1),
        (("scale_constant", "nominal_seconds_per_interval"), 1),
        (("observed_native_reference", "controlled_maximum_from_reader_guard"), 42200),
    ],
)
def test_rehashed_unit_evidence_cannot_contradict_outer_clock(
    evidence: dict[str, Any], field: tuple[str, ...], value: Any
) -> None:
    unit_path = evidence["base"] / "unit-proof"
    units = json.loads(evidence["bodies"][unit_path])
    target = units
    for key in field[:-1]:
        target = target[key]
    target[field[-1]] = value
    unit_raw = canonical(units).encode()
    evidence["bodies"][unit_path] = unit_raw
    evidence["manifest"]["sources"]["unit_proof"]["sha256"] = _sha(unit_raw)
    raw = canonical(evidence["manifest"]).encode()
    evidence["bodies"][evidence["path"]] = raw
    # Both artifact SHA values are deliberately updated. The semantic mismatch,
    # rather than stale bytes, must reject this newly selected synthetic manifest.
    with pytest.raises(ValueError, match=r"scope/sources|nominal scale|reader bound"):
        _load(evidence, expected_sha256=_sha(raw))


def test_native_purpose_remains_ineligible_for_paired_gate_and_old_checkpoints(
    evidence: dict[str, Any],
) -> None:
    contract = native_only_contract(_load(evidence), evidence["reference"])
    assert contract.purpose == "native-only-training"
    assert CommonTrainingContract.from_envelope(contract.envelope()) == contract
    assert all(value is None for value in vars(contract.evidence).values())
    with pytest.raises(ValueError, match="comparison"):
        require_paired_comparison((contract, contract), initial_model_sha256=("1" * 64,) * 2)
    with pytest.raises(ValueError, match="checkpoint"):
        contract.require_checkpoint(replace(contract, purpose="offline-plumbing").envelope())
    with pytest.raises(ValueError, match="independent native source clock"):
        replace(contract, clock=replace(contract.clock, synthetic=True))
    with pytest.raises(ValueError, match="independent native source clock"):
        replace(contract, clock=replace(contract.clock, semantics=adapters.CLOCK_SEMANTICS))
    with pytest.raises(ValueError, match="checkpoint"):
        contract.require_checkpoint(
            replace(contract, clock=replace(contract.clock, evidence_sha256="f" * 64)).envelope()
        )


def test_shared_adapter_validates_real_post_structure_beyond_legacy_clock_and_reset_pins(
    evidence: dict[str, Any],
) -> None:
    proof = _load(evidence)
    contract = native_only_contract(proof, evidence["reference"])
    adapter = NativeOnlyAdapter(contract, proof=proof, reference=evidence["reference"])
    before = _state(2200)
    adapter.bind_reset(before, _reset_info(proof), episode_id="synthetic-only/0")
    for _ in range(12):
        result = _result(before)
        row = adapter.transition(39, result)
        before = result["observation"]
        assert row.info["_trackmaniarl_common_interval"]["elapsed_seconds"] == pytest.approx(0.05)
    assert before["nativeT"] == 2800
    bad_info = _reset_info(proof)
    bad_info["runtimeProvenance"]["pins"]["tape"] = "f" * 64
    with pytest.raises(ValueError, match="engine/tape"):
        adapter.bind_reset(_state(2200), bad_info, episode_id="synthetic-only/1")


@pytest.mark.parametrize("terminal", [True, False])
def test_ragged_finish_or_truncation_keeps_actual_duration_and_bootstrap_semantics(
    evidence: dict[str, Any], *, terminal: bool
) -> None:
    proof = _load(evidence)
    contract = native_only_contract(proof, evidence["reference"])
    adapter = NativeOnlyAdapter(contract, proof=proof, reference=evidence["reference"])
    state = _state(2200)
    adapter.bind_reset(state, _reset_info(proof), episode_id="synthetic-only/0")
    row = adapter.transition(39, _result(state, count=1, terminal=terminal, truncated=not terminal))
    assert (row.next_observation is None) is terminal
    replay, sampler = CommonReplayStore(contract, 4), CommonSampler(contract)
    replay.append(row)
    batch = sampler.sample(replay, BatchRequest(1, gamma=contract.gamma))
    assert batch.bootstrap_discounts.item() == pytest.approx(
        0 if terminal else contract.gamma ** (0.01 / 0.05)
    )
    assert row.info["_trackmaniarl_common_interval"]["elapsed_seconds"] == pytest.approx(0.01)
    with pytest.raises(ValueError, match="reset"):
        adapter.transition(39, _result(state))


def test_native_learner_replay_sampler_and_credit_roundtrip_without_io(
    evidence: dict[str, Any],
) -> None:
    proof = _load(evidence)
    contract = native_only_contract(proof, evidence["reference"])
    adapter = NativeOnlyAdapter(contract, proof=proof, reference=evidence["reference"])
    replay, sampler, budget = (
        CommonReplayStore(contract, 16),
        CommonSampler(contract, seed=91),
        DurationUpdateBudget(contract),
    )
    for episode in range(2):
        before = _state(2200)
        adapter.bind_reset(before, _reset_info(proof), episode_id=f"synthetic-only/{episode}")
        for step in range(4):
            result = _result(before, terminal=step == 3)
            row = adapter.transition(39, result)
            replay.append(row)
            budget.credit(row)
            before = result["observation"]
    learner = make_learner(contract, seed=91)
    request = BatchRequest(4, gamma=contract.gamma)
    learner.update(sampler.sample(replay, request))
    budget.consume()
    snapshot = copy.deepcopy(
        (learner.state_dict(), replay.state_dict(), sampler.state_dict(), budget.state_dict())
    )
    restored = make_learner(contract, seed=92)
    restored_replay, restored_sampler = CommonReplayStore(contract, 16), CommonSampler(contract)
    restored_budget = DurationUpdateBudget(contract)
    restored.load_state_dict(snapshot[0])
    restored_replay.load_state_dict(snapshot[1])
    restored_sampler.load_state_dict(snapshot[2])
    restored_budget.load_state_dict(snapshot[3])
    first, second = (
        sampler.sample(replay, request),
        restored_sampler.sample(restored_replay, request),
    )
    assert first.transition_ids == second.transition_ids
    assert torch.equal(first.bootstrap_discounts, second.bootstrap_discounts)
    assert sum(row.next_observation is None for row in restored_replay.get(list(range(8)))) == 2
    assert restored_budget.integral_available == budget.integral_available == 1
    learner.update(first)
    restored.update(second)
    for actual, expected in zip(
        learner.model.parameters(), restored.model.parameters(), strict=True
    ):
        assert torch.equal(actual, expected)
