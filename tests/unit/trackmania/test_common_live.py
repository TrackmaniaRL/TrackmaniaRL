"""No-engine regression coverage for the real-rollout boundary and gates."""

import hashlib
import json
import struct
from dataclasses import asdict, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
import torch

from scripts.headless_common_live import LiveComponents, LiveOptions, run_rollout
from trackmaniarl.trackmania.common_training import (
    ClockSpec,
    CommonPolylineReference,
    QualificationEvidence,
    adapters,
    make_learner,
)
from trackmaniarl.trackmania.common_training.adapters import (
    CLOCK_SEMANTICS,
    KNOWN_MAP_SHA256,
    NativeCommonAdapter,
    ObservedControlSample,
    PhysicalClockProof,
    WindowsCommonAdapter,
    WindowsStepRecord,
    collect_contract,
    model_weights_sha256,
)
from trackmaniarl.trackmania.common_training.contract import canonical
from trackmaniarl.trackmania.native_headless._runtime.features import action_sgb
from trackmaniarl.trackmania.native_headless.config import NativeRuntimeConfig


def _write_json(path: Path, value: Any) -> str:
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _state(native: int, channels: tuple[float, float, float] = (0, 0, 0)) -> dict[str, Any]:
    raw = bytearray(76)
    struct.pack_into("<10f", raw, 0, 1, 0, 0, 0, (native - 2200) / 50, 0, 0, 1, 0, 0)
    steer, gas, brake = channels
    struct.pack_into("<4f", raw, 60, gas, brake, steer, steer)
    return {
        "schema": "tmrl.native-immediate-state/v1",
        "resetProfile": "neutral-race-start/v7",
        "inputSchedule": adapters.SCHEDULE,
        "nativeT": native,
        "native76Hex": raw.hex(),
    }


class FakeEnvironment:
    def __init__(self, config: NativeRuntimeConfig, *, stop_file: Path | None = None) -> None:
        self.config, self.stop_file = config, stop_file
        self.current, self.elapsed, self.calls = _state(2200), 0, 0
        self.closed = False
        self.terminal_after: int | None = None
        self.stop_on_step: int | None = None
        self.config.trace_dir.mkdir(parents=True)
        self.journal_path = self.config.trace_dir / "mock-only-raw-journal.jsonl"

    def reset(self, *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        del seed
        if self.stop_file is not None and self.stop_on_step is None:
            self.stop_file.touch()
        return self.current, {"runtimeProvenance": {"pins": {"map": KNOWN_MAP_SHA256}}}

    def step_native(self, action: int, *, native_iterations: int) -> dict[str, Any]:
        self.calls += 1
        before = self.current
        n = min(native_iterations, (self.config.duration_limit - self.elapsed) // 10)
        terminal = self.terminal_after == self.calls
        n = 1 if terminal else n
        events = []
        for index, channels in enumerate(action_sgb(action)[:n]):
            self.current = _state(before["nativeT"] + (index + 1) * 10, tuple(channels))
            state = bytes.fromhex(self.current["native76Hex"])
            post = bytearray(256)
            post[:8] = b"TMPOST03"
            struct.pack_into("<I", post, 112, self.current["nativeT"])
            post[152:212], post[88:104] = state[:60], state[60:76]
            events.append(post.hex())
        self.elapsed += n * 10
        truncated = not terminal and self.elapsed >= self.config.duration_limit
        result = {
            "observation": self.current,
            "rawIntervalQualified": True,
            "actualIterations": n,
            "deltaNative": n * 10,
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
        result["rawTransitionSha256"] = hashlib.sha256(
            canonical(
                {
                    "kind": "raw-transition",
                    "before": before,
                    "action": action,
                    "result": result,
                }
            ).encode()
        ).hexdigest()
        if self.stop_file is not None and self.stop_on_step == self.calls:
            self.stop_file.touch()
        return result

    def close(self) -> None:
        self.closed = True
        _write_json(
            self.config.trace_dir / "close-v7.json",
            {
                "failures": [],
                "supervisor": {"remainingOwned": [], "cleanupErrors": []},
            },
        )


class FakeFactory:
    def __init__(self, config: NativeRuntimeConfig) -> None:
        self.config, self.instances = config, []
        self.stop_file: Path | None = None
        self.terminal_after: int | None = None
        self.stop_on_step: int | None = None

    def create(self, *, seed: int) -> FakeEnvironment:
        del seed
        environment = FakeEnvironment(
            replace(
                self.config,
                trace_dir=self.config.trace_dir / str(len(self.instances)),
            ),
            stop_file=self.stop_file,
        )
        environment.terminal_after = self.terminal_after
        environment.stop_on_step = self.stop_on_step
        self.instances.append(environment)
        return environment


def _components(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LiveComponents:
    raw = json.dumps({"points": [[0, 0, 0], [100, 0, 0]]}).encode()
    digest = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(adapters, "KNOWN_REFERENCE_SHA256", digest)
    clock_path = tmp_path / "clock.json"
    clock_sha = _write_json(clock_path, {"mock_clock_evidence": True})
    proof = PhysicalClockProof(
        clock_sha, 2200, 2700, Fraction(1), Fraction(-2190), "mock observed scope", {}
    )
    reference = CommonPolylineReference(raw, expected_sha256=digest)
    config = NativeRuntimeConfig(
        runtime_root=tmp_path / "runtime",
        trace_dir=tmp_path / "trace",
        work_dir=tmp_path / "work",
        asset_manifest=tmp_path / "assets.json",
        protocol_profile="immediate-current/v7",
        reset_profile="neutral-race-start/v7",
        duration_limit=100,
        allow_native=True,
    )
    return LiveComponents(
        config=config, proof=proof, reference=reference, contract=collect_contract(proof, reference)
    )


def _options(tmp_path: Path, **changes: Any) -> LiveOptions:
    output = tmp_path / "output"
    output.mkdir()
    return LiveOptions(
        output=output,
        stop_file=output / "STOP",
        collect_only=True,
        tape="neutral",
        decisions=2,
        **changes,
    )


def test_mock_live_collect_has_genuine_cap_no_model_and_duration_credit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components, options = _components(tmp_path, monkeypatch), _options(tmp_path)
    factory = FakeFactory(components.config)
    report = run_rollout(components, options, factory)
    assert report["passed"]
    assert report["transitions"] == report["retained_rows"] == 4
    assert report["credited_simulated_seconds"] == pytest.approx(0.2)
    assert report["remaining_update_credit"] == pytest.approx(1)
    assert not report["model_executed"]
    assert report["optimizer_steps"] == 0
    assert all(episode["truncated"] for episode in report["episodes"])
    assert all(environment.closed for environment in factory.instances)
    saved = torch.load(options.output / "checkpoint.pt", weights_only=False)
    assert saved["learner"] is None
    assert saved["budget"]["transitions"] == 4


def test_mock_stop_after_reset_creates_no_transition_or_credit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components, options = _components(tmp_path, monkeypatch), _options(tmp_path)
    factory = FakeFactory(components.config)
    factory.stop_file = options.stop_file
    report = run_rollout(components, options, factory)
    assert report["passed"]
    assert report["stopped"]
    assert report["transitions"] == report["credited_transitions"] == 0
    assert report["credited_simulated_seconds"] == 0
    assert factory.instances[0].calls == 0
    assert report["owned_cleanup_verified"]


def test_collection_rejects_injected_learner_before_creating_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components, options = _components(tmp_path, monkeypatch), _options(tmp_path)
    factory = FakeFactory(components.config)
    components = replace(components, learner=make_learner(components.contract))
    with pytest.raises(ValueError, match="collection cannot accept a learner"):
        run_rollout(components, options, factory)
    assert factory.instances == []


def test_stop_preserves_earned_update_credit_without_running_optimizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components, options = _components(tmp_path, monkeypatch), _options(tmp_path)
    components = replace(
        components,
        config=replace(components.config, duration_limit=200),
        learner=make_learner(components.contract),
    )
    options = replace(
        options,
        collect_only=False,
        tape=None,
        infrastructure_smoke=True,
        episodes=1,
        decisions=4,
        batch_size=2,
    )
    factory = FakeFactory(components.config)
    factory.stop_file, factory.stop_on_step = options.stop_file, 4
    report = run_rollout(components, options, factory)
    assert report["passed"]
    assert report["stopped"]
    assert report["transitions"] == 4
    assert report["optimizer_steps"] == 0
    assert report["remaining_integral_update_credit"] == 1
    assert not report["credit_drained"]
    assert report["owned_cleanup_verified"]


def test_missing_cleanup_evidence_cannot_report_verified_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components, options = _components(tmp_path, monkeypatch), _options(tmp_path)
    monkeypatch.setattr(FakeEnvironment, "close", lambda self: None)
    report = run_rollout(components, options, FakeFactory(components.config))
    assert not report["passed"]
    assert not report["owned_cleanup_verified"]
    assert report["cleanup"][0]["verified"] is False


def test_mock_actual_eof_keeps_terminal_next_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components, options = _components(tmp_path, monkeypatch), _options(tmp_path)
    factory = FakeFactory(components.config)
    factory.terminal_after = 1
    report = run_rollout(components, options, factory)
    assert report["passed"]
    rows = [
        json.loads(line)
        for line in (options.output / "common-transitions.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    assert all(row["terminated"] and row["next_observation"] is None for row in rows)
    assert report["credited_simulated_seconds"] == pytest.approx(0.02)


def test_training_and_out_of_scope_rollouts_fail_before_engine_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components, options = _components(tmp_path, monkeypatch), _options(tmp_path)
    factory = FakeFactory(components.config)
    with pytest.raises(ValueError, match="paired evidence gate"):
        run_rollout(components, replace(options, collect_only=False, tape=None), factory)
    with pytest.raises(ValueError, match="proof scope"):
        run_rollout(
            replace(components, config=replace(components.config, duration_limit=1000)),
            replace(options, decisions=11),
            factory,
        )
    assert factory.instances == []


def test_mock_training_requires_artifact_gate_and_drains_exact_credit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components, options = _components(tmp_path, monkeypatch), _options(tmp_path)
    receipts = {}
    for name in asdict(QualificationEvidence()):
        path = tmp_path / (name + ".json")
        digest = _write_json(path, {"mock_evidence": name})
        receipts[name] = {"path": str(path), "sha256": digest}
    receipts["clock_mapping_sha256"] = {
        "path": str(tmp_path / "clock.json"),
        "sha256": components.proof.sha256,
    }
    evidence = QualificationEvidence(**{key: value["sha256"] for key, value in receipts.items()})
    native = replace(components.contract, purpose="qualified-training", evidence=evidence)
    windows = replace(
        native,
        clock=ClockSpec(
            clock_id="mock-windows-race-ms",
            unit="milliseconds",
            seconds_per_tick=0.001,
            semantics=CLOCK_SEMANTICS,
            evidence_sha256=components.proof.sha256,
        ),
    )
    learner = make_learner(native, seed=options.seed)
    model_sha = model_weights_sha256(learner.model)
    gate_path = tmp_path / "paired-gate.json"
    gate_sha = _write_json(
        gate_path,
        {
            "schema": "tmrl.common-paired-qualification/v1",
            "native_contract": native.envelope(),
            "windows_contract": windows.envelope(),
            "initial_model_sha256": {"native": model_sha, "windows": model_sha},
            "evidence": receipts,
            "scope": {
                "map_sha256": KNOWN_MAP_SHA256,
                "reference_sha256": components.reference.file_sha256,
                "clock_proof_sha256": components.proof.sha256,
                "native_iterations": 5,
                "seed": options.seed,
                "max_decisions": 2,
            },
        },
    )
    components = replace(
        components, contract=native, learner=learner, gate_path=gate_path, gate_sha256=gate_sha
    )
    report = run_rollout(
        components,
        replace(options, collect_only=False, tape=None, batch_size=2),
        FakeFactory(components.config),
    )
    assert report["passed"]
    assert report["model_executed"]
    assert report["optimizer_steps"] == 1
    assert report["credit_drained"]
    assert report["remaining_integral_update_credit"] == 0
    receipts_path = Path(receipts["reset_sha256"]["path"])
    receipts_path.write_text("changed")
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        run_rollout(
            replace(components, learner=make_learner(native, seed=options.seed)),
            replace(options, collect_only=False, tape=None),
            FakeFactory(components.config),
        )


def test_literal_action_and_raw_journal_hash_are_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = _components(tmp_path, monkeypatch)
    adapter = NativeCommonAdapter(
        components.contract, proof=components.proof, reference=components.reference
    )
    environment = FakeFactory(components.config).create(seed=1729)
    state, info = environment.reset(seed=1729)
    adapter.bind_reset(state, info, episode_id="mock")
    result = environment.step_native(39, native_iterations=5)
    with pytest.raises(ValueError, match="journal SHA-256"):
        adapter.transition(36, result)
    result["deltaNative"] = 20
    with pytest.raises(ValueError, match="journal SHA-256"):
        adapter.transition(39, result)


def test_mock_native_infrastructure_smoke_is_explicit_and_cannot_be_compared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components, options = _components(tmp_path, monkeypatch), _options(tmp_path)
    components = replace(components, learner=make_learner(components.contract))
    options = replace(
        options, collect_only=False, tape=None, infrastructure_smoke=True, batch_size=2
    )
    report = run_rollout(components, options, FakeFactory(components.config))
    assert report["passed"]
    assert report["infrastructure_smoke"]
    assert report["optimizer_steps"] == 1
    assert report["credit_drained"]
    assert report["checkpoint_roundtrip"]
    assert not report["learning_comparison_run"]
    assert components.contract.purpose == "offline-plumbing"


def test_windows_adapter_retains_observed_inputs_and_actual_duration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = _components(tmp_path, monkeypatch)
    windows = replace(
        components.contract,
        clock=ClockSpec(
            clock_id="windows-connect33.race_ms",
            unit="milliseconds",
            seconds_per_tick=0.001,
            semantics=CLOCK_SEMANTICS,
            evidence_sha256=components.proof.sha256,
        ),
    )
    adapter = WindowsCommonAdapter(windows, proof=components.proof, reference=components.reference)
    before, after = [0.0] * 33, [0.0] * 33
    after[4] = 1.0
    observed = (
        ObservedControlSample(race_ms=10, steer=0, gas=0, brake=0),
        ObservedControlSample(race_ms=30, steer=0, gas=1, brake=0),
    )
    row = adapter.transition(
        WindowsStepRecord(
            before_connect33=before,
            after_connect33=after,
            before_race_ms=10,
            after_race_ms=30,
            issued_action=39,
            observed_controls=observed,
            terminated=False,
            truncated=True,
            finished=False,
            raw_transition_sha256=hashlib.sha256(b"mock-windows-row").hexdigest(),
        )
    )
    metadata = row.info["_trackmaniarl_common_interval"]
    assert metadata["elapsed_seconds"] == pytest.approx(0.02)
    inputs = metadata["input_observation"]
    assert inputs["actual_elapsed_race_ms"] == 20
    assert inputs["issued_native_interval_sgb"][0][1] == 1
    assert inputs["observed_controls"][0]["gas"] == 0
    assert not inputs["applied_equals_issued_qualified"]
    assert row.next_observation is not None


def test_physical_clock_proof_verifies_sources_fit_heldout_and_bounds(tmp_path: Path) -> None:
    sources = {}
    for backend in ("native", "windows"):
        path = tmp_path / (backend + ".json")
        sources[backend] = {"path": str(path), "sha256": _write_json(path, {"mock": backend})}
    matches = []
    for index, native in enumerate((2200, 2250, 2300)):
        position, velocity = [float(index), 0, 0], [1, 0, 0]
        matches.append(
            {
                "native_counter": native,
                "windows_race_ms": native - 2190,
                "split": "heldout" if index == 1 else "fit",
                "position": position,
                "velocity": velocity,
                "physics6Hex": struct.pack("<6f", *position, *velocity).hex(),
            }
        )
    path = tmp_path / "proof.json"
    digest = _write_json(
        path,
        {
            "schema": "tmrl.physical-sample-clock-alignment/v1",
            "relative_clock_alignment_qualified": True,
            "fields": [
                "positionX",
                "positionY",
                "positionZ",
                "velocityX",
                "velocityY",
                "velocityZ",
            ],
            "comparison": "exact float32 bytes; no tolerance or rounding",
            "relationship": "windows_race_ms = slope * native_counter + intercept",
            "scope": "mock supplied traces only",
            "slope": {"numerator": 1, "denominator": 1},
            "intercept": {"numerator": -2190, "denominator": 1},
            "sources": sources,
            "matches": matches,
            "fit_matches": 2,
            "heldout_matches": 1,
            "sampled_race_span_ms": 100,
        },
    )
    proof = PhysicalClockProof.from_json(path, expected_sha256=digest)
    assert proof.race_milliseconds(2300) == 110
    with pytest.raises(ValueError, match="proof scope"):
        proof.require_interval(2300, 2310)
    Path(sources["native"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        PhysicalClockProof.from_json(path, expected_sha256=digest)
