import json
import struct
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from trackmaniarl.core.collector import EpisodeCollector, _EpisodeStep
from trackmaniarl.core.data import BatchRequest, Transition
from trackmaniarl.core.observation_endpoints import prepare_bootstrap_observation
from trackmaniarl.core.replay.store import InMemoryReplayStore
from trackmaniarl.distributed.coordinator_ingest import _credit_updates, replay_info_for_transition
from trackmaniarl.distributed.coordinator_validation import _validate_wire_transition
from trackmaniarl.distributed.protocol import transition_to_wire
from trackmaniarl.trackmania.native_headless.config import (
    NativeConfigurationError,
    NativeRuntimeConfig,
)
from trackmaniarl.trackmania.native_headless.training_contract import (
    NATIVE_INTERVAL_KEY,
    NATIVE_INTERVAL_SCHEMA,
    PROFILE,
    REFERENCE,
    NativeTensorFeaturePipeline,
    validate_native_transition,
)
from trackmaniarl.trackmania.native_headless.training_learning import lazy_targets
from trackmaniarl.trackmania.native_headless.training_sampler import NativeRaggedSampler, _batch
from trackmaniarl.trackmania.parity import (
    COMMON_PROFILE,
    checkpoint_contract,
    from_connect33,
    from_native,
    pending_schedule,
    require_checkpoint_contract,
    require_learning_comparison,
    verify_applied,
)


def row(*, terminal: bool = False, truncated: bool = False, dt: int = 50) -> Transition:
    record = {
        "schema": NATIVE_INTERVAL_SCHEMA,
        "profile": PROFILE,
        "referenceFingerprint": REFERENCE,
        "resetProfile": "neutral-prelude/v1",
        "inputSchedule": "pending-then-future/v1",
        "beforeNativeT": 2230,
        "afterNativeT": 2230 + dt,
        "actualIterations": dt // 10,
        "deltaNative": dt,
        "requestedIterations": 5,
        "effectiveRequestedIterations": 5,
        "pendingNotCredited": True,
        "status": "terminated" if terminal else "truncated" if truncated else "continue",
        "terminated": terminal,
        "truncated": truncated,
        "action": 39,
        "rawTransitionSha256": "a" * 64,
    }
    return Transition(
        torch.ones(44),
        39,
        2.0,
        None if terminal else torch.full((44,), 2.0),
        terminal,
        truncated,
        {NATIVE_INTERVAL_KEY: record},
        "actor/session/0",
        0,
    )


def test_native_sampler_uses_fifo_range_after_eviction(monkeypatch: pytest.MonkeyPatch) -> None:
    store = InMemoryReplayStore(capacity=3)
    for step in range(5):
        store.append(replace(row(terminal=True), episode_id=f"episode-{step}"))

    def reject_full_scan() -> list[int]:
        raise AssertionError("sampling must not allocate every replay ID")

    monkeypatch.setattr(store, "available_ids", reject_full_scan)
    sampler = NativeRaggedSampler(seed=7)
    state = sampler.state_dict()
    batch = sampler.sample(store, BatchRequest(batch_size=3, gamma=0.995))
    assert set(batch.transition_ids) == {2, 3, 4}
    sampler.load_state_dict(state)
    assert (
        sampler.sample(store, BatchRequest(batch_size=3, gamma=0.995)).transition_ids
        == batch.transition_ids
    )


def test_native_sampler_validates_custom_store_ids() -> None:
    class DuplicateIdsStore(InMemoryReplayStore):
        def available_ids(self) -> list[int]:
            return [0, 0]

    with pytest.raises(ValueError, match="unique nonnegative"):
        NativeRaggedSampler().sample(DuplicateIdsStore(), BatchRequest(batch_size=1, gamma=0.995))


@pytest.mark.parametrize("intervals", [1, 5])
@pytest.mark.parametrize("action", range(78))
def test_every_action_keeps_issued_pending_and_applied_separate(
    action: int, intervals: int
) -> None:
    pending = (0.0, 1.0, 1.0)
    plan = pending_schedule(action, pending, intervals)
    assert plan.applied_sgb[0] == pending
    assert plan.applied_sgb[1:] == plan.issued_sgb[: intervals - 1]
    assert plan.pending_sgb == plan.issued_sgb[intervals - 1]
    assert len(plan.issued_sgb) == 5


def test_tap_after_full_brake_contains_two_applied_brake_intervals() -> None:
    plan = pending_schedule(41, (0.0, 1.0, 1.0), 5)
    assert [v[2] for v in plan.applied_sgb] == [1, 1, 0, 0, 0]
    assert plan.pending_sgb[2] == 0


def test_write_receipt_does_not_replace_actual_post_evidence() -> None:
    plan = pending_schedule(39, (0.0, 0.0, 0.0), 1)
    event = bytearray(256)
    event[:8] = b"TMPOST03"
    result = {
        "actualIterations": 1,
        "terminated": False,
        "observation": {"pendingSGB": list(plan.pending_sgb)},
        "authorization": {"nativeEventHex": [event.hex()]},
    }
    verify_applied({"pendingSGB": [0.0, 0.0, 0.0]}, 39, result)
    struct.pack_into("<f", event, 88, 1.0)
    result["authorization"]["nativeEventHex"] = [event.hex()]
    with pytest.raises(ValueError, match="applied controls"):
        verify_applied({"pendingSGB": [0.0, 0.0, 0.0]}, 39, result)


def test_common_fields_have_identical_explicit_mapping_and_scaling() -> None:
    raw = bytearray(76)
    struct.pack_into("<10f", raw, 0, 1.0, 0.0, 0.0, 0.0, 496.0, 10.0, 624.0, 1.0, 2.0, 3.0)
    struct.pack_into("<fI", raw, 40, 200.0, 1)
    connect = np.full(33, np.nan)
    connect[4:10] = [496.0, 10.0, 624.0, 1.0, 2.0, 3.0]
    connect[17:19] = [200.0, 1]
    np.testing.assert_array_equal(from_native({"native76Hex": raw.hex()}), from_connect33(connect))
    assert from_connect33(connect).shape == (8,)
    connect[17] = np.nan
    with pytest.raises(ValueError, match="actual finite"):
        from_connect33(connect)
    with pytest.raises(ValueError, match="unavailable"):
        from_native({})


def test_old_checkpoints_and_changed_reset_schedule_are_rejected() -> None:
    expected = checkpoint_contract(
        reset_profile="neutral-prelude/v1", input_schedule="pending-then-future/v1"
    )
    assert expected["profile"] == COMMON_PROFILE
    require_checkpoint_contract(json.loads(json.dumps(expected)), expected)
    for changed in ({}, {**expected, "reset_profile": "legacy"}, {**expected, "profile": PROFILE}):
        with pytest.raises(ValueError, match="incompatible"):
            require_checkpoint_contract(changed, expected)
    with pytest.raises(ValueError, match="checkpoint contract"):
        NativeRaggedSampler().load_state_dict(
            {"native_ragged_contract": "tmrl.native-headless.example-ragged/v1"}
        )


def test_learning_comparison_requires_each_prerequisite() -> None:
    keys = (
        "reset_qualified",
        "applied_inputs_qualified",
        "clock_mapping_qualified",
        "observation_semantics_qualified",
        "same_model_reward_budget",
    )
    all_pass = dict.fromkeys(keys, True)
    require_learning_comparison(all_pass)
    for key in keys:
        with pytest.raises(ValueError, match=key):
            require_learning_comparison({**all_pass, key: False})


def test_terminal_endpoint_never_transformed_but_truncation_requires_real_features() -> None:
    pipeline = NativeTensorFeaturePipeline()
    assert prepare_bootstrap_observation(pipeline, None, flags=(True, False)) is None
    with pytest.raises(ValueError, match="profile"):
        prepare_bootstrap_observation(pipeline, None, flags=(False, True))
    with pytest.raises(ValueError, match="both"):
        prepare_bootstrap_observation(pipeline, None, flags=(True, True))


def test_wire_accepts_absent_terminal_only_with_opt_in_and_preserves_metadata() -> None:
    transition = row(terminal=True)
    wire = transition_to_wire(transition)
    with pytest.raises((ValueError, TypeError)):
        _validate_wire_transition(wire)
    pipeline = NativeTensorFeaturePipeline()
    _validate_wire_transition(wire, pipeline=pipeline)
    saved = replay_info_for_transition(transition.info, pipeline=pipeline)
    assert saved == transition.info
    saved[NATIVE_INTERVAL_KEY]["deltaNative"] = 10
    assert transition.info[NATIVE_INTERVAL_KEY]["deltaNative"] == 50
    with pytest.raises((ValueError, TypeError)):
        _validate_wire_transition(
            transition_to_wire(replace(transition, terminated=False)), pipeline=pipeline
        )


def test_ragged_bootstrap_uses_duration_and_preserves_truncated_endpoint() -> None:
    rows = [row(terminal=True, dt=20), row(truncated=True), row()]
    batch = _batch(rows, [0, 1, 2])
    assert batch.next_observations.shape == (2, 44)
    assert batch.data["bootstrap_row_indices"].tolist() == [1, 2]
    assert batch.bootstrap_discounts.tolist() == pytest.approx([0.0, 0.995, 0.995])
    calls = []

    def evaluate(value: torch.Tensor) -> torch.Tensor:
        calls.append(value.shape)
        return torch.full((len(value), 32), 3.0)

    targets = lazy_targets(batch, evaluate)
    assert calls == [(2, 44)]
    assert targets[0].tolist() == [2.0] * 32
    assert targets[1, 0] == pytest.approx(2 + 3 * 0.995)
    terminal_only = _batch([rows[0]], [0])
    lazy_targets(terminal_only, lambda _: pytest.fail("terminal has no next observation"))


def test_legacy_reset_cannot_enter_new_training_replay() -> None:
    transition = row()
    transition.info[NATIVE_INTERVAL_KEY]["resetProfile"] = "legacy-gas-prelude/v1"
    with pytest.raises(ValueError, match="neutral reset"):
        validate_native_transition(transition)


def test_credit_counts_experience_not_retained_capacity_and_strict_never_clips() -> None:
    counters = SimpleNamespace(transitions=100, update_credit=0.0)
    run = SimpleNamespace(
        spec=SimpleNamespace(
            training=SimpleNamespace(warmup_transitions=0, updates_per_transition=0.25),
            distributed=SimpleNamespace(strict_update_budget=True, max_update_credit=2),
        )
    )
    coordinator = SimpleNamespace(run=run, counters=counters)
    _credit_updates(coordinator, 0)
    assert counters.update_credit == 25


def test_collector_limit_cannot_manufacture_native_terminal() -> None:
    pipeline = NativeTensorFeaturePipeline()
    collector = EpisodeCollector(None, pipeline, None)
    env = SimpleNamespace(step=lambda _: ({}, 1.0, False, False, {}))
    with pytest.raises(ValueError, match="endpoint required"):
        collector._episode_outcome(_EpisodeStep(env, "0", 0, 1), 39)


def test_native_failure_closes_even_without_training_wrapper(tmp_path: Path) -> None:
    from trackmaniarl.trackmania.native_headless.environment import NativeHeadlessEnvironment

    config = NativeRuntimeConfig(
        runtime_root=tmp_path / "runtime",
        trace_dir=tmp_path / "trace",
        work_dir=tmp_path / "work",
        asset_manifest=tmp_path / "assets.json",
    )
    env = NativeHeadlessEnvironment(config)
    calls = []

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("broken transport")

    env._backend = SimpleNamespace(step=fail, close=lambda: calls.append("closed"))
    env._ready = True
    with pytest.raises(OSError, match="broken transport"):
        env.step_native(39)
    env.close()
    assert calls == ["closed"]


def test_runtime_cannot_be_placed_on_windows_mount(tmp_path: Path) -> None:
    from pathlib import Path

    with pytest.raises(NativeConfigurationError, match="outside /mnt"):
        NativeRuntimeConfig(
            runtime_root=Path("/mnt/c/runtime"),
            trace_dir=tmp_path / "trace",
            work_dir=tmp_path / "work",
            asset_manifest=tmp_path / "assets.json",
        )
