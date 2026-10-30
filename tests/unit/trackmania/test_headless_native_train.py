"""Real learner/replay, synthetic physical producer; never starts an engine."""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from scripts import headless_native_train as trainer
from trackmaniarl.trackmania.common_training import CommonPolylineReference, adapters, native_only
from trackmaniarl.trackmania.common_training.contract import RewardMeasurement
from trackmaniarl.trackmania.common_training.features import CommonTransitionInput, make_transition
from trackmaniarl.trackmania.native_headless.config import NativeRuntimeConfig

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux owned native training storage"
)


def state(native: int) -> dict[str, Any]:
    raw = bytearray(76)
    struct.pack_into("<10f", raw, 0, 1, 0, 0, 0, (native - 2200) / 50, 0, 0, 1, 0, 0)
    return {"nativeT": native, "native76Hex": raw.hex()}


class SyntheticAdapter:
    def __init__(self, contract: Any, **_: Any) -> None:
        self.contract = contract

    def bind_reset(self, before: dict[str, Any], _: Any, *, episode_id: str) -> torch.Tensor:
        self.before, self.episode_id, self.step = before, episode_id, 0
        return torch.zeros(6)

    def transition(self, action: int, result: dict[str, Any]) -> Any:
        after = result["observation"]
        row = make_transition(
            CommonTransitionInput(
                observation={"backend": "native-headless", "raw": self.before},
                next_observation={"backend": "native-headless", "raw": after},
                action=action,
                before_timestamp=self.before["nativeT"],
                after_timestamp=after["nativeT"],
                reward_measurement=RewardMeasurement(
                    progress_before_units=0,
                    progress_after_units=1,
                    deviation_after_units=0,
                    finished=False,
                ),
                terminated=False,
                truncated=result["truncated"],
                raw_transition_sha256="a" * 64,
                episode_id=self.episode_id,
                step=self.step,
            ),
            self.contract,
        )
        self.before, self.step = after, self.step + 1
        return row


class SyntheticFactory:
    def __init__(  # noqa: PLR0913 -- independent injected lifecycle faults
        self,
        *,
        stop: Path | None = None,
        fail: bool = False,
        rollover_reason: str | None = None,
        fail_reset: int | None = None,
        fail_step: int | None = None,
        stop_on_reset: int | None = None,
    ) -> None:
        self.stop, self.fail = stop, fail
        self.rollover_reason, self.fail_reset = rollover_reason, fail_reset
        self.fail_step, self.stop_on_reset = fail_step, stop_on_reset
        self.steps = self.resets = 0
        self.instances: list[SyntheticEnvironment] = []

    def __call__(self, config: NativeRuntimeConfig) -> SyntheticFactory:
        self.config = config
        return self

    def create(self, *, seed: int) -> SyntheticEnvironment:
        del seed
        env = SyntheticEnvironment(
            replace(self.config, trace_dir=self.config.trace_dir / str(len(self.instances))), self
        )
        self.instances.append(env)
        return env


class SyntheticEnvironment:
    def __init__(self, config: NativeRuntimeConfig, factory: SyntheticFactory) -> None:
        self.config, self.factory = config, factory
        config.trace_dir.mkdir(parents=True)
        self.journal_path = config.trace_dir / "synthetic.jsonl"
        self.native, self.closed = 2200, False
        self.close_calls = 0
        self.reset_durations: list[int | None] = []
        self.reuse_checks: list[dict[str, Any]] = []

    def reset(
        self, *, seed: int, duration_limit: int | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del seed
        self.factory.resets += 1
        self.reset_durations.append(duration_limit)
        if len(self.reset_durations) == self.factory.fail_reset:
            raise RuntimeError("synthetic reused reset failure")
        self.native = 2200
        self.duration_limit = duration_limit
        self.journal_path = self.config.trace_dir / f"synthetic-{len(self.reset_durations)}.jsonl"
        if self.factory.resets == self.factory.stop_on_reset:
            assert self.factory.stop is not None
            self.factory.stop.touch()
        return state(2200), {}

    def session_reuse_status(self, **reserves: Any) -> dict[str, Any]:
        self.reuse_checks.append(reserves)
        return {
            "eligible": self.factory.rollover_reason is None,
            "reason": self.factory.rollover_reason,
            **reserves,
        }

    def step_native(self, action: int, *, native_iterations: int) -> dict[str, Any]:
        del action
        self.factory.steps += 1
        if self.factory.fail or self.factory.steps == self.factory.fail_step:
            raise RuntimeError("synthetic actor failure")
        assert self.duration_limit is not None
        n = min(native_iterations, (2200 + self.duration_limit - self.native) // 10)
        self.native += n * 10
        if self.factory.stop and self.factory.stop_on_reset is None:
            self.factory.stop.touch()
        return {
            "observation": state(self.native),
            "actualIterations": n,
            "truncated": self.native == 2200 + self.duration_limit,
        }

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True
        (self.config.trace_dir / "close-v7.json").write_text(
            json.dumps(
                {
                    "failures": [],
                    "supervisor": {"remainingOwned": [], "cleanupErrors": []},
                    "retainedLocalDirectory": "/tmp/tmre-immediate-v7-synthetic-no-runtime",
                }
            )
        )


def components(tmp: Path, monkeypatch: pytest.MonkeyPatch) -> trainer.TrainComponents:
    reference_raw = json.dumps({"points": [[0, 0, 0], [100, 0, 0]]}).encode()
    reference_sha = hashlib.sha256(reference_raw).hexdigest()
    monkeypatch.setattr(adapters, "KNOWN_REFERENCE_SHA256", reference_sha)
    monkeypatch.setattr(native_only, "KNOWN_REFERENCE_SHA256", reference_sha)
    reference = CommonPolylineReference(reference_raw, expected_sha256=reference_sha)
    proof = native_only.NativeOnlyClockProof(
        sha256="b" * 64,
        minimum_native=2200,
        maximum_native=42190,
        seconds_per_tick=0.001,
        scope={
            "native_profile": "native-immediate-kinematics6/v7",
            "reset_profile": "neutral-race-start/v7",
            "input_schedule": adapters.SCHEDULE,
            "map_sha256": adapters.KNOWN_MAP_SHA256,
            "reference_sha256": reference_sha,
            "server_sha256": native_only.KNOWN_SERVER_SHA256,
            "tape_sha256": native_only.KNOWN_TAPE_SHA256,
        },
        source_artifacts={"unit_proof": {}, "server": {}, "tape": {}},
    )
    asset_path = tmp / "assets.json"
    asset_path.write_text("{}")
    config = NativeRuntimeConfig(
        runtime_root=tmp / "runtime",
        trace_dir=tmp / "native",
        work_dir=tmp / "work",
        asset_manifest=asset_path,
        duration_limit=39950,
        protocol_profile="immediate-current/v7",
        reset_profile="neutral-race-start/v7",
        allow_native=True,
    )
    monkeypatch.setattr(trainer, "NativeOnlyAdapter", SyntheticAdapter)
    return trainer.TrainComponents(
        config=config,
        proof=proof,
        reference=reference,
        contract=native_only.native_only_contract(proof, reference),
    )


def options(tmp: Path, name: str = "run", **kwargs: Any) -> trainer.TrainOptions:
    output = tmp / name
    output.mkdir()
    return trainer.TrainOptions(
        output=output,
        stop_file=output / "STOP",
        transitions=40,
        episode_native_units=500,
        native_iterations=5,
        replay_capacity=16,
        batch_size=4,
        checkpoint_every=8,
        max_wall_seconds=30,
        **kwargs,
    )


def test_training_rolls_sessions_wraps_replay_and_resumes_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parts, opts = components(tmp_path, monkeypatch), options(tmp_path)
    factory = SyntheticFactory()
    first = trainer.run_training(parts, opts, factory)
    assert first["passed"], first
    assert first["transitions"] == first["cumulative_transitions"] == 40
    assert first["optimizer_steps"] == 10
    assert first["retained_rows"] == 16
    assert len(factory.instances) == 4
    assert all(env.closed for env in factory.instances)
    assert all(
        env.config.work_dir.is_relative_to(parts.config.work_dir) for env in factory.instances
    )
    assert all(case["truncated"] for case in first["episodes"])
    assert first["initial_model_sha256"] != first["final_model_sha256"]
    assert first["remaining_integral_update_credit"] == 0
    checkpoint = opts.output / "latest.pt"
    saved = torch.load(checkpoint, weights_only=False)
    second_opts = options(
        tmp_path, "resume", resume=checkpoint, resume_sha256=trainer._file_sha(checkpoint)
    )
    second = trainer.run_training(parts, second_opts, SyntheticFactory())
    assert second["passed"], second
    assert second["resume_loaded"]
    assert second["cumulative_transitions"] == 80
    assert second["optimizer_steps"] == 20
    assert second["initial_model_sha256"] == first["final_model_sha256"]
    assert second["episodes"][0]["episode"] == saved["next_episode"]
    assert second["learning_comparison_run"] is False
    assert second["paired_training_qualified"] is False
    baseline_options = replace(options(tmp_path, "uninterrupted"), transitions=80)
    baseline = trainer.run_training(parts, baseline_options, SyntheticFactory())
    assert baseline["passed"], baseline
    assert second["final_model_sha256"] == baseline["final_model_sha256"]

    def actions(output: Path) -> list[tuple[int, float, bool]]:
        return [
            (row["action"], row["reward"], row["truncated"])
            for row in map(json.loads, (output / "transitions.jsonl").read_text().splitlines())
        ]

    assert actions(opts.output) + actions(second_opts.output) == actions(baseline_options.output)


def test_stop_keeps_actual_endpoint_and_releases_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parts, opts = components(tmp_path, monkeypatch), options(tmp_path)
    factory = SyntheticFactory(stop=opts.stop_file)
    result = trainer.run_training(parts, opts, factory)
    assert result["passed"], result
    assert result["stopped"]
    assert result["transitions"] == 1
    assert factory.instances[0].closed
    assert not result["episodes"][0]["truncated"]
    saved = torch.load(opts.output / "latest.pt", weights_only=False)
    journal = json.loads((opts.output / "transitions.jsonl").read_text())
    assert journal["truncated"] is False
    assert journal["terminated"] is False
    assert saved["budget"]["transitions"] == 1


def test_opt_in_reuse_preserves_actions_weights_and_short_final_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parts = components(tmp_path, monkeypatch)
    old_options = replace(options(tmp_path, "fresh"), transitions=25)
    new_options = replace(options(tmp_path, "reused"), transitions=25, reuse_sessions=True)
    fresh, reused = SyntheticFactory(), SyntheticFactory()
    baseline = trainer.run_training(parts, old_options, fresh)
    result = trainer.run_training(parts, new_options, reused)
    assert baseline["passed"], baseline
    assert result["passed"], result
    assert len(fresh.instances) == 3
    assert len(reused.instances) == 1
    assert reused.instances[0].reset_durations == [500, 500, 250]
    assert reused.instances[0].config.duration_limit == parts.config.duration_limit == 39950
    assert reused.instances[0].close_calls == 1
    assert result["native_environment_instances"] == 1
    assert [case["reused_session"] for case in result["episodes"]] == [False, True, True]
    assert len({case["raw_journal"] for case in result["episodes"]}) == 3
    assert result["episodes"][-1]["last_nativeT"] == 2450
    assert result["final_model_sha256"] == baseline["final_model_sha256"]
    assert result["optimizer_steps"] == baseline["optimizer_steps"] == 6
    assert all(value >= 0 for value in result["phase_host_seconds"].values())

    def rows(output: Path) -> list[tuple[Any, ...]]:
        return [
            (row["action"], row["reward"], row["terminated"], row["truncated"])
            for row in map(json.loads, (output / "transitions.jsonl").read_text().splitlines())
        ]

    assert rows(old_options.output) == rows(new_options.output)
    checkpoint = new_options.output / "latest.pt"
    resume_options = replace(
        options(tmp_path, "resumed"),
        transitions=15,
        reuse_sessions=True,
        resume=checkpoint,
        resume_sha256=trainer._file_sha(checkpoint),
    )
    resumed_factory = SyntheticFactory()
    resumed = trainer.run_training(parts, resume_options, resumed_factory)
    # A shortened final episode is an explicit reset boundary on resume.
    assert resumed["passed"], resumed
    assert resumed["resume_loaded"]
    assert resumed["cumulative_transitions"] == 40
    assert resumed["optimizer_steps"] == 10
    assert resumed["initial_model_sha256"] == result["final_model_sha256"]
    assert len(resumed_factory.instances) == 1
    assert resumed_factory.instances[0].close_calls == 1
    old_checkpoint = old_options.output / "latest.pt"
    old_resume_options = replace(
        options(tmp_path, "fresh-resumed"),
        transitions=15,
        resume=old_checkpoint,
        resume_sha256=trainer._file_sha(old_checkpoint),
    )
    old_resumed = trainer.run_training(parts, old_resume_options, SyntheticFactory())
    assert old_resumed["passed"], old_resumed
    assert resumed["final_model_sha256"] == old_resumed["final_model_sha256"]
    assert rows(resume_options.output) == rows(old_resume_options.output)


@pytest.mark.parametrize("reason", ["session deadline reserve", "session evidence reserve"])
def test_reuse_rolls_driver_before_reset_when_reserve_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    parts = components(tmp_path, monkeypatch)
    opts = replace(options(tmp_path), transitions=12, reuse_sessions=True)
    factory = SyntheticFactory(rollover_reason=reason)
    result = trainer.run_training(parts, opts, factory)
    assert result["passed"], result
    assert len(factory.instances) == 2
    assert all(env.close_calls == 1 for env in factory.instances)
    assert [env.reset_durations for env in factory.instances] == [[500], [100]]
    assert result["session_rollovers"][0]["reason"] == reason
    assert not result["episodes"][1]["reused_session"]
    reserve = factory.instances[0].reuse_checks[0]
    assert reserve["minimum_remaining_seconds"] >= 30
    assert reserve["additional_trace_bytes"] > 2 * trainer.MIB


def test_stop_after_second_reused_reset_closes_once_without_extra_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parts = components(tmp_path, monkeypatch)
    opts = replace(options(tmp_path), reuse_sessions=True)
    factory = SyntheticFactory(stop=opts.stop_file, stop_on_reset=2)
    result = trainer.run_training(parts, opts, factory)
    assert result["passed"], result
    assert result["stopped"]
    assert result["transitions"] == factory.steps == 10
    assert factory.resets == 2
    assert len(factory.instances) == 1
    assert factory.instances[0].close_calls == 1
    assert result["episodes"][1]["transitions"] == 0
    saved = torch.load(opts.output / "latest.pt", weights_only=False)
    assert saved["budget"]["transitions"] == 10


def test_rollover_charges_retained_bytes_before_admitting_another_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parts = components(tmp_path, monkeypatch)
    opts = replace(options(tmp_path), reuse_sessions=True, max_run_bytes=256 * trainer.MIB)
    factory = SyntheticFactory(rollover_reason="session evidence reserve")
    original = trainer.directory_bytes

    def with_retained_evidence(paths: Any) -> int:
        measured = original(paths)
        if factory.instances and factory.instances[0].closed:
            measured += 64 * trainer.MIB
        return measured

    monkeypatch.setattr(trainer, "directory_bytes", with_retained_evidence)
    result = trainer.run_training(parts, opts, factory)
    assert result["passed"], result
    assert result["stop_reason"] == "run evidence budget"
    assert result["transitions"] == 10
    assert len(factory.instances) == 1
    assert factory.instances[0].close_calls == 1


@pytest.mark.parametrize("fault", ["reset", "step"])
def test_reused_episode_failure_closes_once_and_keeps_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    parts = components(tmp_path, monkeypatch)
    opts = replace(options(tmp_path), reuse_sessions=True)
    factory = SyntheticFactory(
        fail_reset=2 if fault == "reset" else None, fail_step=11 if fault == "step" else None
    )
    result = trainer.run_training(parts, opts, factory)
    assert not result["passed"]
    assert "synthetic" in result["error"]
    assert len(factory.instances) == 1
    assert factory.instances[0].close_calls == 1
    assert result["owned_cleanup_verified"]
    saved = torch.load(opts.output / "latest.pt", weights_only=False)
    assert saved["budget"]["transitions"] == 10


def test_actor_failure_preserves_last_valid_checkpoint_and_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parts, opts = components(tmp_path, monkeypatch), options(tmp_path)
    factory = SyntheticFactory(fail=True)
    result = trainer.run_training(parts, opts, factory)
    assert not result["passed"]
    assert "synthetic actor failure" in result["error"]
    assert result["owned_cleanup_verified"]
    assert factory.instances[0].closed
    saved = torch.load(opts.output / "latest.pt", weights_only=False)
    assert saved["budget"]["transitions"] == 0


def test_resume_hash_or_schedule_mismatch_cannot_start_native(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parts, opts = components(tmp_path, monkeypatch), options(tmp_path)
    trainer.run_training(parts, opts, SyntheticFactory())
    checkpoint = opts.output / "latest.pt"
    bad = replace(options(tmp_path, "bad"), resume=checkpoint, resume_sha256="0" * 64)
    factory = SyntheticFactory()
    result = trainer.run_training(parts, bad, factory)
    assert not result["passed"]
    assert "SHA-256 mismatch" in result["error"]
    assert not factory.instances
    assert not (bad.output / "latest.pt").exists()
    bad_schedule = replace(
        options(tmp_path, "different"),
        resume=checkpoint,
        resume_sha256=trainer._file_sha(checkpoint),
        native_iterations=1,
    )
    result = trainer.run_training(parts, bad_schedule, factory)
    assert not result["passed"]
    assert "schedule/runtime mismatch" in result["error"]
    assert not factory.instances


def test_existing_stop_does_not_claim_an_empty_cleanup_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parts, opts = components(tmp_path, monkeypatch), options(tmp_path)
    opts.stop_file.touch()
    factory = SyntheticFactory()
    result = trainer.run_training(parts, opts, factory)
    assert result["passed"]
    assert result["stopped"]
    assert result["transitions"] == 0
    assert not factory.instances
    assert result["owned_cleanup_verified"] is None


def test_storage_reservation_stops_before_a_new_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parts, opts = components(tmp_path, monkeypatch), options(tmp_path)
    opts = replace(opts, max_run_bytes=200 * trainer.MIB)
    (opts.output / "existing-evidence.bin").write_bytes(bytes(8 * trainer.MIB))
    factory = SyntheticFactory()
    result = trainer.run_training(parts, opts, factory)
    assert result["passed"]
    assert result["stop_reason"] == "run evidence budget"
    assert not factory.instances


def test_oversized_outer_journal_preserves_checkpoint_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parts, opts = components(tmp_path, monkeypatch), options(tmp_path)
    monkeypatch.setattr(trainer, "JOURNAL_ROW_BYTES", 32)
    factory = SyntheticFactory()
    result = trainer.run_training(parts, opts, factory)
    assert not result["passed"]
    assert "journal row exceeds reserved size" in result["error"]
    assert factory.instances[0].closed
    checkpoint = torch.load(opts.output / "latest.pt", weights_only=False)
    assert checkpoint["budget"]["transitions"] == 0
    assert (opts.output / "transitions.jsonl").stat().st_size == 0


def test_frequent_checkpoints_reserve_private_session_and_future_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parts, opts = components(tmp_path, monkeypatch), options(tmp_path)
    opts = replace(opts, max_run_bytes=400 * trainer.MIB, checkpoint_every=1)
    factory = SyntheticFactory()
    original = trainer.atomic_checkpoint
    allowances: list[int] = []

    def observed_checkpoint(path: Path, payload: Any, *, max_bytes: int) -> Any:
        allowances.append(max_bytes)
        return original(path, payload, max_bytes=max_bytes)

    def conservative_usage(_: Any) -> int:
        return 210 * trainer.MIB if factory.instances and factory.instances[0].native > 2200 else 0

    monkeypatch.setattr(trainer, "directory_bytes", conservative_usage)
    monkeypatch.setattr(trainer, "atomic_checkpoint", observed_checkpoint)
    result = trainer.run_training(parts, opts, factory)
    assert result["passed"], result
    assert result["stop_reason"] == "run evidence budget"
    assert 121 * trainer.MIB in allowances
    assert opts.max_checkpoint_bytes in allowances
