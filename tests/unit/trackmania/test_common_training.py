"""Synthetic contracts verify plumbing; they never qualify either engine."""

import copy
import hashlib
import json
import struct
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from trackmaniarl.core.data import BatchRequest
from trackmaniarl.trackmania.common_training import (
    COMMON6_PROFILE,
    ClockSpec,
    CommonFeaturePipeline,
    CommonPolylineReference,
    CommonReplayStore,
    CommonSampler,
    CommonTrainingContract,
    CommonTransitionInput,
    DurationUpdateBudget,
    ProjectionLimits,
    QualificationEvidence,
    RewardMeasurement,
    RewardSpec,
    make_learner,
    make_transition,
    require_paired_comparison,
    validate_transition,
)
from trackmaniarl.trackmania.common_training.contract import sha256
from trackmaniarl.trackmania.common_training.learning import lazy_targets
from trackmaniarl.trackmania.common_training.replay import collate_rows


def contract(**changes: Any) -> CommonTrainingContract:
    base = CommonTrainingContract(
        clock=ClockSpec(
            clock_id="synthetic-fixture-clock",
            unit="milliseconds",
            seconds_per_tick=0.001,
            semantics="synthetic-simulated-time/v1",
            evidence_sha256=sha256("clock"),
            synthetic=True,
        ),
        reward=RewardSpec(reference_sha256=sha256("geometry"), projection_profile="fixture/v1"),
        reset_profile="fixture-neutral/v1",
        input_schedule="fixture-applied/v1",
        observation_semantics="fixture-position-velocity-rpm-gear/v1",
    )
    return replace(base, **changes)


def endpoint(backend: str, *, position: float = 1.0) -> dict[str, Any]:
    if backend == "windows-connect33":
        raw = np.zeros(33, dtype=np.float64)
        raw[4:10] = [position, 10, 20, 1, 2, 3]
        raw[17:19] = [200, 1]
    else:
        data = bytearray(76)
        struct.pack_into("<10f", data, 0, 1, 0, 0, 0, position, 10, 20, 1, 2, 3)
        struct.pack_into("<fI", data, 40, 200, 1)
        raw = {"native76Hex": data.hex()}
    return {"backend": backend, "raw": raw}


def source(  # noqa: PLR0913 - explicit test fixture axes
    *,
    backend: str = "windows-connect33",
    elapsed_ms: int = 50,
    terminated: bool = False,
    truncated: bool = False,
) -> CommonTransitionInput:
    return CommonTransitionInput(
        observation=endpoint(backend),
        next_observation=None if terminated else endpoint(backend, position=2),
        action=39,
        before_timestamp=1000,
        after_timestamp=1000 + elapsed_ms,
        reward_measurement=RewardMeasurement(
            progress_before_units=1,
            progress_after_units=2,
            deviation_after_units=3,
            finished=terminated,
        ),
        terminated=terminated,
        truncated=truncated,
        raw_transition_sha256=sha256([backend, elapsed_ms, terminated, truncated]),
    )


@pytest.mark.parametrize("profile", [None, COMMON6_PROFILE])
def test_both_raw_backends_share_real_tensor_reward_and_model(profile: str | None) -> None:
    selected = contract(**({"observation_profile": profile} if profile else {}))
    windows = make_transition(source(), selected)
    native = make_transition(source(backend="native-headless"), selected)
    assert torch.equal(windows.observation, native.observation)
    assert torch.equal(windows.next_observation, native.next_observation)
    assert windows.reward == native.reward == pytest.approx(0.087)
    assert windows.observation.shape == (selected.feature_dim,)
    selected.require_checkpoint(json.loads(json.dumps(selected.envelope())))


def test_common6_does_not_read_excluded_rpm_gear_fields() -> None:
    selected = contract(observation_profile=COMMON6_PROFILE)
    packet = endpoint("windows-connect33")
    packet["raw"][17:19] = [np.nan, np.nan]
    values = CommonFeaturePipeline(selected).transform_observation(packet)
    assert values.shape == (6,)
    assert torch.isfinite(values).all()
    with pytest.raises(ValueError, match="actual finite"):
        CommonFeaturePipeline(contract()).transform_observation(packet)
    native = endpoint("native-headless")
    native_packet = bytearray.fromhex(native["raw"]["native76Hex"])
    struct.pack_into("<f", native_packet, 40, float("nan"))
    native["raw"]["native76Hex"] = native_packet.hex()
    assert torch.equal(CommonFeaturePipeline(selected).transform_observation(native), values)


def test_complete_serialized_contract_roundtrip_rejects_missing_conversion() -> None:
    selected = contract(observation_profile=COMMON6_PROFILE)
    saved = json.loads(json.dumps(selected.envelope()))
    restored = CommonTrainingContract.from_envelope(saved)
    assert restored.envelope() == selected.envelope()
    del saved["clock"]["seconds_per_tick"]
    with pytest.raises(TypeError, match="seconds_per_tick"):
        CommonTrainingContract.from_envelope(saved)


def test_shared_polyline_projection_is_continuous_and_binds_full_json(tmp_path: Path) -> None:
    payload = json.dumps(
        {
            "points": [[0, 0, 0], [10, 0, 0], [10, 0, 10]],
            "source": {"fixture": True},
            "nativeTimes": [0, 10, 20],
        }
    ).encode()
    full_sha = hashlib.sha256(payload).hexdigest()
    path = tmp_path / "reference.json"
    path.write_bytes(payload)
    reference = CommonPolylineReference.from_json(path, expected_sha256=full_sha)
    result = reference.project_world([3, 4, 0])
    assert result.segment_index == 0
    assert result.segment_fraction == pytest.approx(0.3)
    assert result.progress_units == pytest.approx(3)
    assert result.deviation_units == pytest.approx(4)
    assert reference.project_world([10, 0, 0]).segment_index == 0
    assert reference.project_world([10, 0, 7]).progress_units == pytest.approx(17)
    measured = reference.measure_world(([3, 4, 0], [10, 0, 7]), finished=True)
    spec = reference.reward_spec()
    assert spec.reference_sha256 == full_sha
    assert spec.coordinate_unit == "engine-world-unit"
    assert spec.score(measured, 0.05) == pytest.approx(21.39)
    path.write_bytes(payload + b" ")
    with pytest.raises(ValueError, match="SHA-256"):
        CommonPolylineReference.from_json(path, expected_sha256=full_sha)


def test_projection_limits_reject_without_clamping_or_inferred_motion() -> None:
    raw = json.dumps([[0, 0, 0], [10, 0, 0]]).encode()
    reference = CommonPolylineReference(
        raw,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        limits=ProjectionLimits(max_deviation_units=2, max_progress_delta_units=3),
    )
    with pytest.raises(ValueError, match="deviation rejection limit"):
        reference.project_world([3, 4, 0])
    with pytest.raises(ValueError, match="progress rejection limit"):
        reference.measure_world(([0, 0, 0], [8, 0, 0]), finished=False)


def test_stationary_reference_samples_preserve_continuous_arc_and_original_index() -> None:
    raw = json.dumps([[0, 0, 0], [0, 0, 0], [10, 0, 0], [10, 0, 0], [20, 0, 0]]).encode()
    reference = CommonPolylineReference(raw, expected_sha256=hashlib.sha256(raw).hexdigest())
    projected = reference.project_world([15, 2, 0])
    assert projected.segment_index == 3
    assert projected.segment_fraction == pytest.approx(0.5)
    assert projected.progress_units == pytest.approx(15)
    assert projected.deviation_units == pytest.approx(2)


def test_both_backend_features_use_the_same_shared_projection() -> None:
    raw = json.dumps([[0, 10, 20], [100, 10, 20]]).encode()
    reference = CommonPolylineReference(raw, expected_sha256=hashlib.sha256(raw).hexdigest())
    selected = contract(observation_profile=COMMON6_PROFILE, reward=reference.reward_spec())
    pipeline = CommonFeaturePipeline(selected)
    measurements = []
    for backend in ("windows-connect33", "native-headless"):
        before = pipeline.transform_observation(endpoint(backend))
        after = pipeline.transform_observation(endpoint(backend, position=2))
        measurements.append(reference.measure_features((before, after), finished=False))
    assert measurements[0] == measurements[1]


def test_unknown_clock_mapping_is_not_inferred_from_iterations_or_timestamps() -> None:
    kwargs = asdict(contract().clock)
    for changes in (
        {"seconds_per_tick": None},
        {"evidence_sha256": None},
        {"unit": "unknown"},
        {"seconds_per_tick": 1.0},
    ):
        with pytest.raises(ValueError, match=r"required|disagree"):
            ClockSpec(**{**kwargs, **changes})
    for before, after in ((10, 10), (11, 10), (None, 50), (10, np.nan)):
        with pytest.raises(ValueError, match=r"timestamps required|finite number required"):
            make_transition(
                replace(source(), before_timestamp=before, after_timestamp=after), contract()
            )


def test_terminal_is_ragged_and_truncation_requires_actual_next() -> None:
    selected = contract()
    terminal = make_transition(source(terminated=True, elapsed_ms=20), selected)
    truncated = make_transition(source(truncated=True), selected)
    continuing = make_transition(source(elapsed_ms=10), selected)
    batch = collate_rows([terminal, truncated, continuing], [0, 1, 2], selected)
    assert batch.next_observations.shape == (2, 8)
    assert batch.data["bootstrap_row_indices"].tolist() == [1, 2]
    assert batch.bootstrap_discounts.tolist() == pytest.approx([0, 0.995, 0.995**0.2])
    calls = []

    def evaluate(actual: torch.Tensor) -> torch.Tensor:
        calls.append(actual.shape)
        return torch.full((len(actual), 32), 2.0)

    targets = lazy_targets(batch, selected, evaluate)
    assert calls == [(2, 8)]
    assert targets[0].tolist() == pytest.approx([terminal.reward] * 32)
    assert targets[1, 0] == pytest.approx(truncated.reward + 2 * 0.995)
    terminal_batch = collate_rows([terminal], [0], selected)
    lazy_targets(terminal_batch, selected, lambda _: pytest.fail("terminal NEXT was encoded"))
    for invalid in (
        replace(source(truncated=True), next_observation=None),
        replace(source(), terminated=True),
        replace(source(terminated=True), truncated=True),
    ):
        with pytest.raises(ValueError, match=r"NEXT|both"):
            make_transition(invalid, selected)


def test_reward_and_endpoint_metadata_cannot_silently_change_in_replay() -> None:
    selected = contract()
    row = make_transition(source(), selected)
    for changed in (
        replace(row, reward=123),
        replace(row, observation=torch.zeros(8)),
        replace(row, next_observation=torch.zeros(8)),
        replace(row, action=40),
    ):
        with pytest.raises(ValueError, match="differ"):
            validate_transition(changed, selected)


@pytest.mark.parametrize(
    "field",
    [
        "reset_profile",
        "input_schedule",
        "observation_semantics",
        "clock",
        "reward",
        "gamma",
        "discount_reference_seconds",
        "updates_per_reference",
        "observation_profile",
    ],
)
def test_checkpoints_bind_every_training_contract(field: str) -> None:
    selected = contract()
    replacement = {
        "clock": replace(selected.clock, evidence_sha256=sha256("other mapping")),
        "reward": replace(selected.reward, finish_bonus=21),
        "gamma": 0.99,
        "discount_reference_seconds": 0.1,
        "updates_per_reference": 0.5,
        "observation_profile": COMMON6_PROFILE,
    }.get(field, "changed")
    changed = replace(selected, **{field: replacement})
    with pytest.raises(ValueError, match="incompatible"):
        changed.require_checkpoint(selected.envelope())
    with pytest.raises(ValueError, match="incompatible"):
        selected.require_checkpoint({})


def test_duration_credit_ignores_replay_capacity_and_keeps_short_steps_fractional() -> None:
    selected = contract()
    replay, budget = CommonReplayStore(selected, capacity=2), DurationUpdateBudget(selected)
    for _ in range(20):
        row = make_transition(source(elapsed_ms=10), selected)
        replay.append(row)
        budget.credit(row)
    assert len(replay) == 2
    assert budget.transitions == 20
    assert budget.elapsed_seconds == pytest.approx(0.2)
    assert budget.integral_available == 1
    budget.consume()
    assert budget.credit_available == pytest.approx(0)
    with pytest.raises(ValueError, match="no duration-earned"):
        budget.consume()
    restored = DurationUpdateBudget(selected)
    restored.load_state_dict(budget.state_dict())
    assert restored.state_dict() == budget.state_dict()


def test_synthetic_and_incomplete_evidence_cannot_qualify_learning_comparison() -> None:
    selected = contract()
    model_hashes = (sha256("same weights"), sha256("same weights"))
    with pytest.raises(ValueError, match="offline plumbing"):
        require_paired_comparison((selected, selected), initial_model_sha256=model_hashes)
    with pytest.raises(ValueError, match="synthetic"):
        replace(selected, purpose="qualified-training")
    real_clock = replace(selected.clock, synthetic=False)
    with pytest.raises(ValueError, match="reset_sha256"):
        replace(selected, clock=real_clock, purpose="qualified-training")
    evidence = QualificationEvidence(
        **{
            name: real_clock.evidence_sha256 if name == "clock_mapping_sha256" else sha256(name)
            for name in asdict(QualificationEvidence())
        }
    )
    qualified = replace(selected, clock=real_clock, evidence=evidence, purpose="qualified-training")
    require_paired_comparison((qualified, qualified), initial_model_sha256=model_hashes)
    with pytest.raises(ValueError, match="contracts differ"):
        require_paired_comparison(
            (qualified, replace(qualified, reset_profile="other")),
            initial_model_sha256=model_hashes,
        )
    with pytest.raises(ValueError, match="identical model"):
        require_paired_comparison(
            (qualified, qualified),
            initial_model_sha256=(model_hashes[0], sha256("other weights")),
        )


@pytest.mark.parametrize("profile", [None, COMMON6_PROFILE])
def test_cpu_update_checkpoint_replay_sampler_and_optimizer_resume(
    tmp_path: Path,
    profile: str | None,
) -> None:
    selected = contract(**({"observation_profile": profile} if profile else {}))
    replay, sampler = CommonReplayStore(selected, 8), CommonSampler(selected, seed=1729)
    budget = DurationUpdateBudget(selected)
    for index in range(8):
        row = make_transition(
            source(
                backend="native-headless" if index % 2 else "windows-connect33",
                terminated=index % 3 == 0,
                truncated=index % 3 == 1,
            ),
            selected,
        )
        replay.append(row)
        budget.credit(row)
    first, twin = make_learner(selected), make_learner(selected)
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            first.model.parameters(),
            twin.model.parameters(),
            strict=True,
        )
    )
    before = [parameter.detach().clone() for parameter in first.model.parameters()]
    metrics, priorities = first.update(
        sampler.sample(replay, BatchRequest(8, gamma=selected.gamma))
    )
    budget.consume()
    assert all(np.isfinite(value) for value in metrics.values())
    assert len(priorities.priorities) == 8
    assert any(not torch.equal(a, b) for a, b in zip(before, first.model.parameters(), strict=True))
    checkpoint = tmp_path / "common.pt"
    torch.save(
        {
            "learner": first.state_dict(),
            "replay": replay.state_dict(),
            "sampler": sampler.state_dict(),
            "budget": budget.state_dict(),
        },
        checkpoint,
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    twin.load_state_dict(saved["learner"])
    other_replay, other_sampler = CommonReplayStore(selected, 8), CommonSampler(selected)
    other_replay.load_state_dict(saved["replay"])
    other_sampler.load_state_dict(saved["sampler"])
    actual = sampler.sample(replay, BatchRequest(8, gamma=selected.gamma))
    restored = other_sampler.sample(other_replay, BatchRequest(8, gamma=selected.gamma))
    assert actual.transition_ids == restored.transition_ids
    assert torch.equal(actual.bootstrap_discounts, restored.bootstrap_discounts)
    first.update(actual)
    twin.update(restored)
    assert first.update_count == twin.update_count == 2
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            first.model.parameters(),
            twin.model.parameters(),
            strict=True,
        )
    )
    wrong = make_learner(replace(selected, reset_profile="different"))
    initial = copy.deepcopy(wrong.state_dict())
    with pytest.raises(ValueError, match="incompatible"):
        wrong.load_state_dict(saved["learner"])
    assert wrong.update_count == 0
    assert initial["common_contract"] == wrong.state_dict()["common_contract"]
