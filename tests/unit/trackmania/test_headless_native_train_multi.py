"""Actual learner/replay, explicitly synthetic actor transport; never an engine."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import Any, ClassVar

import pytest
import torch

from scripts import headless_native_train as single
from scripts import headless_native_train_multi as multi
from tests.unit.trackmania.test_common_training_performance import _assert_exact
from tests.unit.trackmania.test_headless_native_train import (
    SyntheticFactory,
    components,
    options,
    state,
)
from trackmaniarl.trackmania.common_training import native_only
from trackmaniarl.trackmania.common_training.contract import RewardMeasurement
from trackmaniarl.trackmania.common_training.features import CommonTransitionInput, make_transition

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux owned native training storage"
)


class SyntheticAdapter:
    validations: ClassVar[list[tuple[int, int]]] = []
    reject_actor: ClassVar[int | None] = None

    def __init__(self, contract: Any, **_: Any) -> None:
        self.contract = contract

    def bind_reset(self, before: Any, _: Any, *, episode_id: str) -> Any:
        self.before, self.episode_id, self.step = before, episode_id, 0
        self.actor = int(episode_id.split("/")[-2])
        return torch.zeros(6)

    def transition(self, action: int, result: Any) -> Any:
        type(self).validations.append((self.actor, self.step))
        if self.actor == type(self).reject_actor:
            raise ValueError("synthetic invalid peer row")
        after = result["observation"]
        terminal = result["terminated"]
        row = make_transition(
            CommonTransitionInput(
                observation={"backend": "native-headless", "raw": self.before},
                next_observation=None if terminal else {"backend": "native-headless", "raw": after},
                action=action,
                before_timestamp=self.before["nativeT"],
                after_timestamp=after["nativeT"],
                reward_measurement=RewardMeasurement(
                    progress_before_units=0,
                    progress_after_units=1,
                    deviation_after_units=0,
                    finished=terminal,
                ),
                terminated=terminal,
                truncated=result["truncated"],
                raw_transition_sha256="a" * 64,
                episode_id=self.episode_id,
                step=self.step,
            ),
            self.contract,
        )
        self.before, self.step = after, self.step + 1
        return row


class SyntheticRoundError(RuntimeError):
    def __init__(self, acquired: Any) -> None:
        super().__init__("synthetic peer process failure")
        self.evidence = {"acquired_unaccepted_results": acquired, "accepted_rows": 0}


class SyntheticPool:
    def __init__(self, config: Any, **fields: Any) -> None:
        self.config, self.fields = config, fields
        self.actor_count = fields["actor_count"]
        self.envs: dict[int, dict[str, Any]] = {}
        self.prepared = self.reset_requests = self.actions = 0
        self.raw_rows = 0
        self.resets: list[dict[int, Any]] = []
        self.barriers: list[int] = []
        self.in_flight = self.closed = False
        self.close_counts: dict[int, int] = {}
        self.fail_round: int | None = None
        self.stop_on_reset: Path | None = None
        self.stop_after_round: Path | None = None
        self.mixed = False
        self.rollover = False
        self.cleanup: list[dict[str, Any]] = []

    def open(self) -> SyntheticPool:
        self.fields["trace_root"].mkdir(parents=True)
        self.fields["work_root"].mkdir(parents=True)
        return self

    def checkpoint_barrier(self) -> dict[int, Any]:
        assert not self.in_flight
        self.barriers.append(self.actions)
        return {}

    def prepare_many(self, requests: Any) -> dict[int, Any]:
        result = {}
        for actor in sorted(requests):
            assert actor not in self.envs
            directory = self.fields["trace_root"] / f"fixture-{actor}-{self.prepared}"
            directory.mkdir()
            self.prepared += 1
            self.envs[actor] = {
                "trace_dir": str(directory),
                "work_dir": str(self.fields["work_root"]),
                "reset_count": 0,
                "seed": requests[actor]["seed"],
            }
            result[actor] = dict(self.envs[actor])
        return result

    def reset_many(self, requests: Any) -> dict[int, Any]:
        self.reset_requests += 1
        self.resets.append(copy.deepcopy(requests))
        result = {}
        for actor in sorted(requests):
            env = self.envs[actor]
            env.update(native=2200, limit=requests[actor]["duration_limit"], active=True)
            env["reset_count"] += 1
            env["journal_path"] = str(Path(env["trace_dir"]) / f"raw-{env['reset_count']}.jsonl")
            Path(env["journal_path"]).write_text("", encoding="utf-8")
            result[actor] = {"state": state(2200), "info": {}, "metadata": dict(env)}
        if self.stop_on_reset and self.reset_requests == 2:
            self.stop_on_reset.touch()
        return result

    def step_many(self, requests: Any) -> dict[int, Any]:
        self.in_flight = True
        acquired = {}
        try:
            self.actions += 1
            for actor in sorted(requests):
                env = self.envs[actor]
                assert env["active"]
                count = min(
                    requests[actor]["native_iterations"],
                    (2200 + env["limit"] - env["native"]) // 10,
                )
                terminal = self.mixed and actor == 1 and self.actions == 1
                if terminal:
                    count = 1
                env["native"] += count * 10
                truncated = not terminal and env["native"] == 2200 + env["limit"]
                env["active"] = not (terminal or truncated)
                acquired[actor] = {
                    "observation": state(env["native"]),
                    "actualIterations": count,
                    "terminated": terminal,
                    "truncated": truncated,
                }
                with Path(env["journal_path"]).open("a", encoding="utf-8") as journal:
                    journal.write(json.dumps(acquired[actor]) + "\n")
                self.raw_rows += 1
                if self.actions == self.fail_round and actor == 0:
                    raise SyntheticRoundError(acquired)
            if self.stop_after_round:
                self.stop_after_round.touch()
            return dict(reversed(list(acquired.items())))  # Deliberately reversed completion order.
        finally:
            self.in_flight = False

    def session_reuse_status_many(self, requests: Any) -> dict[int, Any]:
        return {
            actor: {
                "eligible": not self.rollover,
                "reason": "fixture reserve" if self.rollover else None,
            }
            for actor in requests
        }

    def close_session(self, actor: int) -> dict[str, Any]:
        env = self.envs.pop(actor)
        trace = env["trace_dir"]
        self.close_counts[actor] = self.close_counts.get(actor, 0) + 1
        value = {
            "verified": True,
            "owned_processes_gone": True,
            "close_errors": [],
            "native_failures": [],
            "retained_local_directory": None,
            "trace_dir": trace,
            "work_dir": env["work_dir"],
            "native_execution_started": True,
            "supervisor": {"remainingOwned": [], "cleanupErrors": []},
            "actor_id": actor,
        }
        self.cleanup.append(value)
        return value

    def close_all(self) -> dict[int, Any]:
        assert not self.in_flight
        result = {}
        for actor in range(self.actor_count):
            value = (
                self.close_session(actor)
                if actor in self.envs
                else {
                    "verified": True,
                    "owned_processes_gone": True,
                    "retained_local_directory": None,
                    "native_execution_started": False,
                }
            )
            result[actor] = {**value, "worker_reaped": True, "worker_exitcode": 0}
        self.closed = True
        return result


class PoolFactory:
    def __init__(self, **faults: Any) -> None:
        self.faults = faults
        self.instances: list[SyntheticPool] = []

    def __call__(self, config: Any, **fields: Any) -> SyntheticPool:
        pool = SyntheticPool(config, **fields)
        for name, value in self.faults.items():
            setattr(pool, name, value)
        self.instances.append(pool)
        return pool


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    parts = components(tmp_path, monkeypatch)
    monkeypatch.setattr(native_only, "NativeOnlyAdapter", SyntheticAdapter)
    SyntheticAdapter.validations, SyntheticAdapter.reject_actor = [], None
    return parts


def journal(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (path / "transitions.jsonl").read_text().splitlines()]


def test_single_actor_matches_r8_physics_model_optimizer_rng_and_resume(
    tmp_path: Path, fixture: Any
) -> None:
    first, second = options(tmp_path, "old"), options(tmp_path, "multi")
    expected = single.run_training(fixture, first, SyntheticFactory())
    factory = PoolFactory()
    actual = multi.run_training(fixture, second, pool_factory=factory)
    assert expected["passed"], expected
    assert actual["passed"], actual
    assert actual["final_model_sha256"] == expected["final_model_sha256"]
    old = torch.load(first.output / "latest.pt", weights_only=False)
    new = torch.load(second.output / "latest.pt", weights_only=False)
    _assert_exact(old["learner"], new["learner"])
    _assert_exact(old["sampler"], new["sampler"])
    _assert_exact(old["budget"], new["budget"])

    def semantic_rows(rows: Any) -> Any:
        return [
            (
                row["action"],
                row["reward"],
                row["terminated"],
                row["truncated"],
                row["info"]["_trackmaniarl_common_interval"]["observation_sha256"],
                row["info"]["_trackmaniarl_common_interval"]["next_observation_sha256"],
            )
            for row in rows
        ]

    assert semantic_rows(journal(first.output)) == semantic_rows(journal(second.output))
    resumed_options = replace(
        options(tmp_path, "resumed"),
        resume=second.output / "latest.pt",
        resume_sha256=single._file_sha(second.output / "latest.pt"),
    )
    resumed = multi.run_training(fixture, resumed_options, pool_factory=PoolFactory())
    baseline_options = replace(options(tmp_path, "uninterrupted"), transitions=80)
    baseline = single.run_training(fixture, baseline_options, SyntheticFactory())
    assert resumed["passed"], resumed
    assert baseline["passed"], baseline
    assert resumed["final_model_sha256"] == baseline["final_model_sha256"]
    assert resumed["episodes"][0]["episode"] == new["actor_next_episode"][0]
    assert factory.instances[0].closed


def test_reverse_completion_preserves_actor_order_policy_versions_and_partial_budget(
    tmp_path: Path, fixture: Any
) -> None:
    opts = replace(options(tmp_path), transitions=25, reuse_sessions=True)
    factory = PoolFactory()
    result = multi.run_training(fixture, opts, actors=2, pool_factory=factory)
    assert result["passed"], result
    rows = journal(opts.output)
    assert [row["actor_id"] for row in rows] == [0, 1] * 12 + [0]
    assert result["actor_new_transitions"] == {0: 13, 1: 12}
    assert result["optimizer_steps"] == 6
    assert result["native_intervals"] == 125
    assert result["native_environment_instances"] == 2
    assert len({case["raw_journal"] for case in result["episodes"]}) == 4
    for value in result["rounds"]:
        selected = [row for row in rows if row["round_id"] == value["round_id"]]
        assert {row["policy_update_version"] for row in selected} == {
            value["policy_update_version"]
        }
    assert [request[0]["duration_limit"] for request in factory.instances[0].resets] == [500, 150]
    assert [request[1]["duration_limit"] for request in factory.instances[0].resets] == [500, 100]
    assert all(count == 1 for count in factory.instances[0].close_counts.values())


@pytest.mark.parametrize("failure", ["peer", "invalid"])
def test_failed_round_preserves_raw_success_and_never_admits_peer_rows(
    tmp_path: Path, fixture: Any, failure: str
) -> None:
    opts = replace(options(tmp_path), transitions=12)
    factory = PoolFactory(fail_round=1 if failure == "peer" else None)
    if failure == "invalid":
        SyntheticAdapter.reject_actor = 1
    result = multi.run_training(fixture, opts, actors=2, pool_factory=factory)
    assert not result["passed"]
    assert result["transitions"] == result["optimizer_steps"] == 0
    assert (opts.output / "transitions.jsonl").read_bytes() == b""
    saved = torch.load(opts.output / "latest.pt", weights_only=False)
    assert saved["budget"]["transitions"] == 0
    assert saved["completed_rounds"] == 0
    assert next((opts.output / "native").rglob("raw-*.jsonl")).stat().st_size > 0
    assert factory.instances[0].closed
    if failure == "peer":
        assert result["pool_failure_evidence"]["accepted_rows"] == 0
    else:
        assert SyntheticAdapter.validations == [(0, 0), (1, 0)]


def test_mixed_actual_terminal_duration_and_stop_keep_literal_rows_and_credit(
    tmp_path: Path, fixture: Any
) -> None:
    opts = replace(options(tmp_path), transitions=12)
    factory = PoolFactory(mixed=True, stop_after_round=opts.stop_file)
    result = multi.run_training(fixture, opts, actors=2, pool_factory=factory)
    assert result["passed"], result
    assert result["stopped"]
    assert result["transitions"] == 2
    assert result["native_intervals"] == 6
    assert result["credited_simulated_seconds"] == 0.06
    assert result["remaining_update_credit"] == pytest.approx(0.3)
    assert result["optimizer_steps"] == 0
    rows = journal(opts.output)
    assert rows[0]["terminated"] is False
    assert rows[0]["truncated"] is False
    assert rows[1]["terminated"] is True
    assert rows[1]["truncated"] is False
    saved = torch.load(opts.output / "latest.pt", weights_only=False)
    assert saved["budget"]["transitions"] == 2


@pytest.mark.parametrize(
    "changed", ["actors", "scheduler", "schema", "rounds_zero", "rounds_too_many", "replay_count"]
)
def test_resume_requires_matching_multi_actor_scheduler_and_round_barrier(
    tmp_path: Path, fixture: Any, changed: str
) -> None:
    opts = replace(options(tmp_path), transitions=4)
    first = multi.run_training(fixture, opts, actors=2, pool_factory=PoolFactory())
    assert first["passed"], first
    path = opts.output / "latest.pt"
    saved = torch.load(path, weights_only=False)
    if changed == "scheduler":
        saved["schedule"]["scheduler"] = "different"
    elif changed == "schema":
        saved["schema"] = single.CHECKPOINT_SCHEMA
    elif changed == "rounds_zero":
        saved["completed_rounds"] = 0
    elif changed == "rounds_too_many":
        saved["completed_rounds"] = 5
    elif changed == "replay_count":
        saved["completed_rounds"] = 3
        saved["actor_cumulative_transitions"][0] += 1
        saved["budget"]["transitions"] += 1
    edited = tmp_path / "edited.pt"
    torch.save(saved, edited)
    resumed = replace(
        options(tmp_path, "bad"), resume=edited, resume_sha256=single._file_sha(edited)
    )
    factory = PoolFactory()
    result = multi.run_training(
        fixture, resumed, actors=1 if changed == "actors" else 2, pool_factory=factory
    )
    assert not result["passed"]
    assert not factory.instances
    assert not (resumed.output / "latest.pt").exists()


def test_spawn_import_and_help_do_not_import_torch() -> None:
    source = (
        "import sys; import scripts.headless_native_train_multi; assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", source], check=True, timeout=5)


def test_warmup_burst_reports_actual_non_power_of_two_policy_lag(
    tmp_path: Path, fixture: Any
) -> None:
    opts = replace(options(tmp_path), transitions=35, batch_size=32, replay_capacity=40)
    result = multi.run_training(fixture, opts, actors=3, pool_factory=PoolFactory())
    assert result["passed"], result
    assert result["actor_new_transitions"] == {0: 12, 1: 12, 2: 11}
    assert result["optimizer_steps"] == 8
    # Row 32 (actor 1) drains all eight accumulated updates; actor 2's
    # action was selected before that burst from the same round snapshot.
    admissions = result["rounds"][10]["admissions"]
    assert [item["update_lag"] for item in admissions] == [0, 0, 8]
    assert result["maximum_observed_policy_update_lag"] == 8
    assert all(item["issued_policy_update_version"] == 0 for item in admissions)
    assert all(
        item["learner_update_version_at_admission"] - item["issued_policy_update_version"]
        == item["update_lag"]
        for round_ in result["rounds"]
        for item in round_["admissions"]
    )


def test_failed_second_round_keeps_previously_durable_barrier_checkpoint(
    tmp_path: Path, fixture: Any
) -> None:
    opts = replace(options(tmp_path), transitions=12, checkpoint_every=1)
    factory = PoolFactory(fail_round=2)
    result = multi.run_training(fixture, opts, actors=2, pool_factory=factory)
    assert not result["passed"]
    assert result["transitions"] == 2
    assert len(journal(opts.output)) == 2
    saved = torch.load(opts.output / "latest.pt", weights_only=False)
    assert saved["budget"]["transitions"] == 2
    assert saved["completed_rounds"] == 1
    assert result["pool_failure_evidence"]["accepted_rows"] == 0
    assert factory.instances[0].barriers == [0, 1]
    assert factory.instances[0].closed


@pytest.mark.parametrize("rollover", [False, True])
def test_stop_after_second_reset_closes_reused_or_replaced_sessions_without_step(
    tmp_path: Path, fixture: Any, *, rollover: bool
) -> None:
    opts = replace(options(tmp_path), transitions=44, reuse_sessions=True)
    factory = PoolFactory(stop_on_reset=opts.stop_file, rollover=rollover)
    result = multi.run_training(fixture, opts, actors=2, pool_factory=factory)
    assert result["passed"], result
    assert result["stopped"]
    assert result["transitions"] == 20
    assert factory.instances[0].actions == 10
    assert result["native_environment_instances"] == (4 if rollover else 2)
    assert len(result["episodes"]) == 4
    assert [case["transitions"] for case in result["episodes"]] == [10, 10, 0, 0]
    assert factory.instances[0].close_counts == dict.fromkeys(range(2), 2 if rollover else 1)


def test_retained_close_roots_are_readmitted_before_any_replacement_reset(
    tmp_path: Path, fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trackmaniarl.trackmania.native_headless import training_storage

    opts = replace(options(tmp_path), transitions=44, reuse_sessions=True)
    factory = PoolFactory(rollover=True)
    original = training_storage.directory_bytes
    reserve = (
        2 * (fixture.config.max_trace_bytes or 64 * multi.MIB)
        + opts.max_checkpoint_bytes
        + multi.MIB
        + single.JOURNAL_ROW_BYTES * single.JOURNAL_GROUP_ROWS
    )

    def retained_usage(roots: Any) -> int:
        if factory.instances and factory.instances[0].close_counts:
            return opts.max_run_bytes - reserve + 1
        return original(roots)

    monkeypatch.setattr(training_storage, "directory_bytes", retained_usage)
    result = multi.run_training(fixture, opts, actors=2, pool_factory=factory)
    assert result["passed"], result
    assert result["stop_reason"] == "run evidence budget"
    assert result["transitions"] == 20
    assert factory.instances[0].prepared == 2
    assert factory.instances[0].reset_requests == 1


def test_multi_resume_resets_every_actor_and_preserves_independent_episode_ids(
    tmp_path: Path, fixture: Any
) -> None:
    opts = replace(options(tmp_path), transitions=8)
    first = multi.run_training(fixture, opts, actors=2, pool_factory=PoolFactory())
    assert first["passed"], first
    resumed = replace(
        options(tmp_path, "again"),
        transitions=8,
        resume=opts.output / "latest.pt",
        resume_sha256=single._file_sha(opts.output / "latest.pt"),
    )
    pool = PoolFactory()
    result = multi.run_training(fixture, resumed, actors=2, pool_factory=pool)
    assert result["passed"], result
    assert result["resume_loaded"]
    assert result["cumulative_transitions"] == 16
    assert pool.instances[0].resets[0] == {
        0: {"seed": resumed.seed + 2, "duration_limit": 200},
        1: {"seed": resumed.seed + 3, "duration_limit": 200},
    }
    assert [(case["actor_id"], case["episode"]) for case in result["episodes"]] == [(0, 1), (1, 1)]


def test_failed_cleanup_accounts_every_actor_retained_root_before_rejecting(
    tmp_path: Path, fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = SyntheticPool.close_session

    def failed_cleanup(self: SyntheticPool, actor: int) -> dict[str, Any]:
        value = original(self, actor)
        root = tmp_path / f"retained-{actor}"
        root.mkdir()
        value.update(verified=actor != 0, retained_local_directory=str(root))
        return value

    monkeypatch.setattr(SyntheticPool, "close_session", failed_cleanup)
    opts = replace(options(tmp_path), transitions=12)
    result = multi.run_training(
        fixture, opts, actors=2, pool_factory=PoolFactory(stop_after_round=opts.stop_file)
    )
    assert not result["passed"]
    assert result["owned_cleanup_verified"] is False
    assert len(result["cleanup"]) == 2
    assert result["retained_local_directories"] == [
        str(tmp_path / "retained-0"),
        str(tmp_path / "retained-1"),
    ]


def test_verified_native_cleanup_with_worker_crash_is_not_successful_training(
    tmp_path: Path, fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = SyntheticPool.close_all

    def crash_after_cleanup(self: SyntheticPool) -> dict[int, Any]:
        result = original(self)
        result[0]["worker_exitcode"] = -9
        return result

    monkeypatch.setattr(SyntheticPool, "close_all", crash_after_cleanup)
    opts = replace(options(tmp_path), transitions=2)
    result = multi.run_training(fixture, opts, pool_factory=PoolFactory())
    assert not result["passed"]
    assert "zero actor worker exit codes" in result["cleanup_error"]
    assert result["worker_cleanup"][0]["verified"] is True
    assert result["worker_cleanup"][0]["worker_reaped"] is True


def test_final_actor_close_failure_rejects_entire_pending_round_before_credit(
    tmp_path: Path, fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = SyntheticPool.close_session

    def failed_close(self: SyntheticPool, actor: int) -> dict[str, Any]:
        value = original(self, actor)
        value["verified"] = False
        return value

    monkeypatch.setattr(SyntheticPool, "close_session", failed_close)
    opts = replace(options(tmp_path), transitions=2)
    result = multi.run_training(fixture, opts, actors=2, pool_factory=PoolFactory())
    assert not result["passed"]
    assert result["transitions"] == 0
    assert result["optimizer_steps"] == 0
    assert journal(opts.output) == []
    saved = torch.load(opts.output / "latest.pt", weights_only=False)
    assert saved["budget"]["transitions"] == 0
    assert saved["completed_rounds"] == 0


@pytest.mark.parametrize("actors", [1, 3, 4])
def test_global_scan_groups_never_exceed_reserved_rows_and_avoid_round_scans(  # noqa: PLR0913
    tmp_path: Path, fixture: Any, monkeypatch: pytest.MonkeyPatch, actors: int
) -> None:
    from trackmaniarl.trackmania.native_headless import training_storage

    opts = replace(
        options(tmp_path), transitions=150, episode_native_units=10000, checkpoint_every=1000
    )
    factory = PoolFactory()
    original = training_storage.directory_bytes
    scans = []

    def observed_usage(roots: Any) -> int:
        scans.append(factory.instances[0].raw_rows if factory.instances else 0)
        return original(roots)

    monkeypatch.setattr(training_storage, "directory_bytes", observed_usage)
    result = multi.run_training(fixture, opts, actors=actors, pool_factory=factory)
    assert result["passed"], result
    assert result["maximum_unchecked_outer_rows"] <= single.JOURNAL_GROUP_ROWS
    assert max(after - before for before, after in pairwise(scans)) <= 64
    assert len(scans) < 20  # Includes fresh checkpoint/report scans, not just admission.
    assert result["storage_admission_checks"] == (6 if actors == 4 else 5)
    assert (63 if actors == 3 else 64) in scans
    assert (126 if actors == 3 else 128) in scans
    assert result["native_intervals"] == 750


def test_retired_peer_close_forces_fresh_scan_before_remaining_actor_step(
    tmp_path: Path, fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trackmaniarl.trackmania.native_headless import training_storage

    opts = replace(options(tmp_path), transitions=3)
    factory = PoolFactory()
    original = training_storage.directory_bytes
    reserve = (
        2 * (fixture.config.max_trace_bytes or 64 * multi.MIB)
        + opts.max_checkpoint_bytes
        + multi.MIB
        + single.JOURNAL_ROW_BYTES * single.JOURNAL_GROUP_ROWS
    )

    def retained_usage(roots: Any) -> int:
        if factory.instances and 1 in factory.instances[0].close_counts:
            return opts.max_run_bytes - reserve + 1
        return original(roots)

    monkeypatch.setattr(training_storage, "directory_bytes", retained_usage)
    result = multi.run_training(fixture, opts, actors=2, pool_factory=factory)
    assert result["passed"], result
    assert result["stop_reason"] == "run evidence budget"
    assert result["transitions"] == 2
    assert factory.instances[0].actions == 1
    assert result["storage_admission_checks"] == 4


def test_policy_refreshes_only_completed_rounds_with_actual_model_updates(
    tmp_path: Path, fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trackmaniarl.algorithms.value_based.policy import DiscreteValuePolicy

    original = DiscreteValuePolicy.load_state
    calls = []

    def observed_load(self: DiscreteValuePolicy, state: Any) -> None:
        calls.append(len(calls))
        original(self, state)

    monkeypatch.setattr(DiscreteValuePolicy, "load_state", observed_load)
    opts = replace(options(tmp_path), transitions=35, batch_size=32, replay_capacity=40)
    result = multi.run_training(fixture, opts, actors=3, pool_factory=PoolFactory())
    assert result["passed"], result
    assert result["optimizer_steps"] == 8
    assert len(calls) == 1  # The eight-update warmup burst shares one round refresh.
    assert result["maximum_observed_policy_update_lag"] == 8
