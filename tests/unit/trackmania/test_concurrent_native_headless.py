"""Spawn/IPC/failure proofs with tiny Python children, never the native engine."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import struct
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from trackmaniarl.trackmania.native_headless import concurrent
from trackmaniarl.trackmania.native_headless._runtime import ownership
from trackmaniarl.trackmania.native_headless._runtime.supervisor import spawn_supervised
from trackmaniarl.trackmania.native_headless.config import (
    IMMEDIATE_PROTOCOL,
    IMMEDIATE_RESET_PROFILE,
    NativeRuntimeConfig,
)

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux lifetime/IPC contract")


class _FakeEnvironment:
    def __init__(self, config: NativeRuntimeConfig, seed: int) -> None:
        self.config, self.seed = config, seed
        self._session: Any = None
        self.journal_path: Path | None = None
        self.count = 0
        self.closed = False

    def reset(self, *, seed: int | None = None, duration_limit: int | None = None) -> Any:
        if self._session is None:
            self.config.trace_dir.mkdir(parents=True)
            self.config.work_dir.mkdir(parents=True)
            self.private = Path(tempfile.mkdtemp(prefix="tmre-immediate-v7-"))
            self.stderr = (self.config.trace_dir / "dummy.stderr").open("xb")
            (self.config.trace_dir / "launch-v7.json").write_text(
                json.dumps(
                    {
                        "environment": {"TMRE_TREE_DIR": str(self.private)},
                    }
                )
            )
            process = spawn_supervised(
                [sys.executable, "-c", "import time; time.sleep(120)"],
                cwd=self.config.work_dir,
                env=dict(os.environ),
                stderr=self.stderr,
                evidence=self.config.trace_dir / "native-supervisor.json",
                timeout=5,
            )
            self._session = SimpleNamespace(proc=process, deadline=time.monotonic() + 30)
            self.journal_path = self.config.trace_dir / "raw.jsonl"
        if self.seed == 101:
            raise ValueError("explicit reset failure")
        if self.seed == 102:
            time.sleep(120)
        if self.seed == 104:
            os._exit(17)
        self.count = 0
        return {"position": 0}, {"seed": self.seed, "duration_limit": duration_limit}

    def step_native(self, action: int, *, native_iterations: int) -> dict[str, Any]:
        if self.seed == 103:
            time.sleep(0.15)
            raise ValueError("explicit step failure")
        if self.seed == 105:
            time.sleep(0.1)
        if self.seed == 107:
            time.sleep(120)
        self.count += 1
        result = {
            "observation": {"position": self.count},
            "action": action,
            "actualIterations": native_iterations,
            "terminated": False,
            "truncated": self.count >= 2,
        }
        assert self.journal_path is not None
        with self.journal_path.open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        return result

    def session_reuse_status(self, **fields: Any) -> dict[str, Any]:
        return {"eligible": self.count >= 2, **fields, "used_bytes": self.count}

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self._session is None:
            return
        report = self._session.proc.stop(timeout=5)
        self._session.proc.close_control()
        self.stderr.close()
        (self.config.trace_dir / "close-v7.json").write_text(
            json.dumps(
                {
                    "failures": [],
                    "supervisor": report,
                    "retainedLocalDirectory": str(self.private),
                }
            )
        )
        if self.seed == 106:
            timer = threading.Timer(0.05, lambda: os._exit(17))
            timer.daemon = False
            timer.start()


def _fake_environment(config: NativeRuntimeConfig, seed: int) -> _FakeEnvironment:
    lease = "env-" + uuid4().hex
    return _FakeEnvironment(
        replace(config, trace_dir=config.trace_dir / lease, work_dir=config.work_dir / lease), seed
    )


def _config(root: Path) -> NativeRuntimeConfig:
    return NativeRuntimeConfig(
        runtime_root=root / "runtime",
        trace_dir=root / "config-trace",
        work_dir=root / "runtime/build/server-probe/config-work",
        asset_manifest=root / "assets.json",
        protocol_profile=IMMEDIATE_PROTOCOL,
        reset_profile=IMMEDIATE_RESET_PROFILE,
        allow_native=True,
    )


def _pool(root: Path, count: int = 2, **fields: Any) -> concurrent.NativeActorPool:
    return concurrent.NativeActorPool(
        _config(root),
        actor_count=count,
        trace_root=root / "trace",
        work_root=root / "runtime/build/server-probe/pool",
        request_timeout_s=15,
        _environment_factory=_fake_environment,
        **fields,
    )


def _requests(count: int, seed: int = 0) -> dict[int, dict[str, int]]:
    return {key: {"seed": seed} for key in range(count)}


def _gone(row: dict[str, Any], timeout: float = 8) -> None:
    deadline = time.monotonic() + timeout
    while ownership.same_identity(row, ownership.process_identity(row["pid"])):
        assert time.monotonic() < deadline, row
        time.sleep(0.01)


def test_prepare_is_lazy_and_roots_are_disjoint_before_reset(tmp_path: Path) -> None:
    with _pool(tmp_path) as pool:
        before = pool.prepare_many(_requests(2))
        assert list(before) == [0, 1]
        assert all(not Path(row["trace_dir"]).exists() for row in before.values())
        assert len({row["trace_dir"] for row in before.values()}) == 2
        assert len({row["work_dir"] for row in before.values()}) == 2
        assert all(row["driver_identity"] is None for row in before.values())
        assert len({row["worker_identity"]["pid"] for row in before.values()}) == 2
        assert pool.checkpoint_barrier() == before
    assert all(row["verified"] and row["worker_reaped"] for row in pool.close_all().values())


def test_spawn_rounds_return_actor_order_and_explicit_session_rollover(tmp_path: Path) -> None:
    with _pool(tmp_path) as pool:
        pool.prepare_many({0: {"seed": 105}, 1: {"seed": 0}})
        reset = pool.reset_many({1: {"duration_limit": 100}, 0: {"duration_limit": 100}})
        assert list(reset) == [0, 1]
        assert (
            reset[0]["metadata"]["driver_identity"]["pid"]
            != reset[1]["metadata"]["driver_identity"]["pid"]
        )
        with pytest.raises(ValueError, match="ended episode"):
            pool.reset_many({0: {}})
        with pytest.raises(ValueError, match="new environment"):
            pool.prepare_many({0: {"seed": 9}})
        for _ in range(2):
            result = pool.step_many(
                {1: {"action": 3, "native_iterations": 1}, 0: {"action": 4, "native_iterations": 5}}
            )
            assert list(result) == [0, 1]
        assert not pool.in_flight
        assert pool.reuse_status(0, minimum_remaining_seconds=1, additional_trace_bytes=100)[
            "eligible"
        ]
        pool.reset_many({0: {}})
        cleanup = pool.close_session(0)
        assert cleanup["verified"]
        assert cleanup["retained_local_directory"]
        assert pool.metadata()[0]["prepared"] is False
        new = pool.prepare_many({0: {"seed": 2, "timeout_s": 1}})[0]
        assert new["trace_dir"] != cleanup["trace_dir"]
        assert not Path(new["trace_dir"]).exists()
    assert all(row["worker_reaped"] for row in pool.close_all().values())


def test_split_step_advances_workers_before_finish_and_blocks_other_operations(
    tmp_path: Path,
) -> None:
    with _pool(tmp_path) as pool:
        pool.prepare_many(_requests(2))
        pool.reset_many({0: {}, 1: {}})
        metadata = pool.metadata()
        requests = {
            1: {"action": 3, "native_iterations": 1},
            0: {"action": 4, "native_iterations": 5},
        }
        pool.begin_step_many(requests)
        assert pool.in_flight
        deadline = time.monotonic() + 3
        while not all(
            Path(row["journal_path"]).exists() and Path(row["journal_path"]).stat().st_size
            for row in metadata.values()
        ):
            assert time.monotonic() < deadline
            time.sleep(0.005)
        # Workers have already written their full evidence while the parent
        # remains free to do learner work. Nothing has been returned to admit.
        for operation in (
            lambda: pool.begin_step_many(requests),
            lambda: pool.prepare_many(_requests(2)),
            lambda: pool.reset_many({0: {}}),
            lambda: pool.close_actor(0),
            pool.metadata,
            pool.checkpoint_barrier,
        ):
            with pytest.raises(RuntimeError):
                operation()
        result = pool.finish_step_many()
        assert list(result) == [0, 1]
        assert result[0]["action"] == 4
        assert result[0]["actualIterations"] == 5
        assert result[1]["action"] == 3
        assert result[1]["actualIterations"] == 1
        assert not pool.in_flight
        assert all(row["episode_active"] for row in pool.checkpoint_barrier().values())
        synchronous = pool.step_many(requests)
        assert all(row["truncated"] for row in synchronous.values())
        with pytest.raises(RuntimeError, match="in-flight"):
            pool.finish_step_many()


def test_split_step_does_not_renew_deadline_after_parent_work(tmp_path: Path) -> None:
    pool = _pool(tmp_path, 1).open()
    pool.prepare_many({0: {"seed": 0}})
    pool.reset_many({0: {}})
    pool.request_timeout_s = 0.05
    pool.begin_step_many({0: {"action": 1, "native_iterations": 1}})
    # The reply can already be buffered. The parent's learner delay must still
    # count against the same deadline that covered dispatch and native work.
    time.sleep(0.08)
    with pytest.raises(concurrent.NativeActorPoolError) as caught:
        pool.finish_step_many()
    assert "deadline" in caught.value.evidence["failures"]["round"]
    assert caught.value.evidence["accepted_rows"] == 0
    assert not pool.in_flight
    cleanup = pool.close_all()[0]
    assert cleanup["worker_reaped"]
    assert cleanup["owned_processes_gone"]


def test_close_all_cancels_pending_step_with_zero_admission_and_scoped_cleanup(
    tmp_path: Path,
) -> None:
    pool = _pool(tmp_path, 1).open()
    pool.prepare_many({0: {"seed": 107}})
    pool.reset_many({0: {}})
    identity = pool.metadata()[0]["worker_identity"]
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        pool.begin_step_many({0: {"action": 1, "native_iterations": 5}})
        started = time.monotonic()
        with pytest.raises(concurrent.NativeActorPoolError, match="cancelled") as caught:
            pool.close_all()
        assert time.monotonic() - started < 13
        evidence = caught.value.evidence
        assert evidence["accepted_rows"] == 0
        assert evidence["acquired_unaccepted_results"] == {}
        assert evidence["cleanup"][0]["worker_reaped"]
        assert evidence["cleanup"][0]["owned_processes_gone"]
        assert unrelated.poll() is None
        assert not pool.in_flight
        _gone(identity)
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_parent_learner_error_cancels_pending_step_without_hiding_original(tmp_path: Path) -> None:
    pool = _pool(tmp_path, 1)

    def learner_failure() -> None:
        with pool:
            pool.prepare_many({0: {"seed": 107}})
            pool.reset_many({0: {}})
            pool.begin_step_many({0: {"action": 1, "native_iterations": 5}})
            raise ValueError("learner failed")

    with pytest.raises(ValueError, match="learner failed") as caught:
        learner_failure()
    assert any("cleanup also failed" in note for note in caught.value.__notes__)
    assert not pool.in_flight
    cleanup = pool.close_all()[0]
    assert cleanup["worker_reaped"]
    assert cleanup["owned_processes_gone"]


def test_split_empty_round_is_a_barrier_until_finished(tmp_path: Path) -> None:
    with _pool(tmp_path, 1) as pool:
        pool.begin_step_many({})
        assert pool.in_flight
        with pytest.raises(RuntimeError, match="boundary"):
            pool.checkpoint_barrier()
        assert pool.finish_step_many() == {}
        assert not pool.in_flight


def test_split_finish_remains_owned_by_creating_main_thread(tmp_path: Path) -> None:
    with _pool(tmp_path, 1) as pool:
        pool.prepare_many({0: {"seed": 0}})
        pool.reset_many({0: {}})
        pool.begin_step_many({0: {"action": 1, "native_iterations": 1}})
        errors = []

        def finish_elsewhere() -> None:
            try:
                pool.finish_step_many()
            except RuntimeError as error:
                errors.append(str(error))

        thread = threading.Thread(target=finish_elsewhere)
        thread.start()
        thread.join(timeout=2)
        assert errors
        assert "main thread" in errors[0]
        assert pool.in_flight
        assert pool.finish_step_many()[0]["action"] == 1


@pytest.mark.parametrize(
    "mutation",
    [
        {"schema": "foreign"},
        {"actor_id": 2},
        {"request_id": 8},
        {"op": "reset"},
        {"kind": "late"},
    ],
)
def test_foreign_or_late_reply_cannot_match_round(mutation: dict[str, Any]) -> None:
    value = {
        "schema": concurrent.SCHEMA,
        "actor_id": 0,
        "request_id": 7,
        "op": "step",
        "kind": "result",
        **mutation,
    }
    with pytest.raises(ValueError, match="identity/sequence"):
        concurrent.NativeActorPool._match(value, 0, 7, "step")


@pytest.mark.parametrize("partial", [b"\x00\x00", struct.pack("!I", 100) + b"{}"])
def test_partial_ipc_frame_cannot_outlive_round_deadline(partial: bytes) -> None:
    reader, writer = mp.get_context("spawn").Pipe(duplex=True)
    try:
        os.write(writer.fileno(), partial)
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="framed IPC"):
            concurrent._receive(reader, deadline=started + 0.05)
        assert time.monotonic() - started < 1
    finally:
        reader.close()
        writer.close()


def test_unread_large_ipc_reply_has_a_finite_write_deadline() -> None:
    reader, writer = mp.get_context("spawn").Pipe(duplex=True)
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="framed IPC"):
            concurrent._send(writer, {"data": "x" * (3 * 1024 * 1024)}, deadline=started + 0.1)
        assert time.monotonic() - started < 1
    finally:
        reader.close()
        writer.close()


def test_expired_ipc_deadline_rejects_even_immediately_ready_descriptor() -> None:
    reader, writer = mp.get_context("spawn").Pipe(duplex=True)
    try:
        concurrent._send(writer, {"buffered": True})
        with pytest.raises(TimeoutError, match="framed IPC"):
            concurrent._receive(reader, deadline=time.monotonic() - 0.01)
        assert concurrent._receive(reader) == {"buffered": True}
        with pytest.raises(TimeoutError, match="framed IPC"):
            concurrent._send(writer, {"late": True}, deadline=time.monotonic() - 0.01)
        assert not reader.poll()
    finally:
        reader.close()
        writer.close()


def test_worker_error_invalidates_round_and_preserves_unaccepted_peer_raw(tmp_path: Path) -> None:
    pool = _pool(tmp_path).open()
    pool.prepare_many({0: {"seed": 0}, 1: {"seed": 103}})
    pool.reset_many({0: {}, 1: {}})
    with pytest.raises(concurrent.NativeActorPoolError) as caught:
        pool.step_many(
            {0: {"action": 1, "native_iterations": 5}, 1: {"action": 2, "native_iterations": 5}}
        )
    evidence = caught.value.evidence
    assert evidence["accepted_rows"] == 0
    assert evidence["acquired_unaccepted_results"][0]["action"] == 1
    assert Path(evidence["evidence_path"]).is_file()
    assert all(
        row["worker_reaped"] and row["owned_processes_gone"] for row in evidence["cleanup"].values()
    )
    assert all(Path(row["trace_dir"]).is_dir() for row in evidence["cleanup"].values())
    assert not pool.in_flight
    with pytest.raises(RuntimeError, match="boundary"):
        pool.checkpoint_barrier()


def test_replacement_lease_cannot_inherit_prior_verified_cleanup(tmp_path: Path) -> None:
    pool = _pool(tmp_path, 1).open()
    pool.prepare_many({0: {"seed": 0}})
    pool.reset_many({0: {}})
    old = pool.close_session(0)
    assert old["verified"]
    new = pool.prepare_many({0: {"seed": 0}})[0]
    assert new["trace_dir"] != old["trace_dir"]
    pool.reset_many({0: {}})
    assert ownership.signal_identity(pool.metadata()[0]["worker_identity"], 9)
    with pytest.raises(concurrent.NativeActorPoolError) as caught:
        pool.step_many({0: {"action": 1, "native_iterations": 1}})
    cleanup = caught.value.evidence["cleanup"][0]
    assert cleanup["owned_processes_gone"]
    assert cleanup["verified"] is False
    assert cleanup["trace_dir"] == new["trace_dir"]
    assert cleanup["supervisor"]["supervisorIdentity"] != old["supervisor"]["supervisorIdentity"]


def test_native_close_ack_cannot_hide_abnormal_worker_exit(tmp_path: Path) -> None:
    pool = _pool(tmp_path, 1).open()
    pool.prepare_many({0: {"seed": 106}})
    pool.reset_many({0: {}})
    with pytest.raises(concurrent.NativeActorPoolError) as caught:
        pool.close_all()
    cleanup = caught.value.evidence["cleanup"][0]
    assert cleanup["worker_reaped"]
    assert cleanup["worker_exitcode"] == 17
    assert cleanup["owned_processes_gone"]


@pytest.mark.parametrize("seed", [102, 104])
def test_timeout_or_worker_death_recovers_only_its_supervisor(tmp_path: Path, seed: int) -> None:
    pool = _pool(tmp_path, 1).open()
    pool.prepare_many({0: {"seed": seed}})
    pool.request_timeout_s = 0.6
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        started = time.monotonic()
        with pytest.raises(concurrent.NativeActorPoolError) as caught:
            pool.reset_many({0: {}})
        assert time.monotonic() - started < 13
        cleanup = caught.value.evidence["cleanup"][0]
        assert cleanup["owned_processes_gone"]
        assert cleanup["worker_reaped"]
        assert unrelated.poll() is None
        rows = cleanup["supervisor"]["capturedOwned"]
        assert unrelated.pid not in {row["pid"] for row in rows}
        assert all(concurrent._identity_gone(row) for row in rows)
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_cancellation_reaps_spawn_workers_before_propagating_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _pool(tmp_path).open()
    pool.prepare_many(_requests(2))
    identities = [row["worker_identity"] for row in pool.metadata().values()]

    def interrupt(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(concurrent, "wait", interrupt)
    with pytest.raises(concurrent.NativeActorPoolError):
        pool.reset_many({0: {}, 1: {}})
    assert not pool.in_flight
    for row in identities:
        _gone(row)


@pytest.mark.parametrize("exit_mode", ["kill", "normal"])
def test_parent_death_releases_all_worker_native_leases(tmp_path: Path, exit_mode: str) -> None:
    script = tmp_path / "owner.py"
    script.write_text(
        "import json,sys,time\nfrom pathlib import Path\n"
        "from test_concurrent_native_headless import _pool\n"
        "if __name__ == '__main__':\n"
        " root=Path(sys.argv[1])\n pool=_pool(root).open()\n"
        " pool.prepare_many({0:{'seed':0},1:{'seed':0}})\n"
        " pool.reset_many({0:{},1:{}})\n"
        " (root/'ready.json').write_text(json.dumps(pool.metadata()))\n"
        " if sys.argv[2] == 'kill': time.sleep(120)\n"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(Path(__file__).resolve().parent),
            str(Path(__file__).resolve().parents[3]),
            environment.get("PYTHONPATH", ""),
        )
    )
    with (tmp_path / "parent.stderr").open("wb") as stderr:
        process = subprocess.Popen(
            [sys.executable, str(script), str(tmp_path), exit_mode], env=environment, stderr=stderr
        )
    try:
        deadline = time.monotonic() + 30
        while not (tmp_path / "ready.json").exists():
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.01)
        metadata = json.loads((tmp_path / "ready.json").read_text())
        if exit_mode == "kill":
            process.kill()
        process.wait(timeout=5)
        for row in metadata.values():
            _gone(row["worker_identity"])
            path = Path(row["trace_dir"]) / "native-supervisor.json"
            deadline = time.monotonic() + 8
            while not path.exists():
                assert time.monotonic() < deadline
                time.sleep(0.01)
            receipt = json.loads(path.read_text())
            assert receipt["remainingOwned"] == []
            assert receipt["reason"] == "actor-lease-eof"
            assert not (Path(row["trace_dir"]) / "close-v7.json").exists()
            for identity in receipt["capturedOwned"]:
                _gone(identity)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_non_main_thread_cannot_create_parent_lifetime_scope(tmp_path: Path) -> None:
    errors = []

    def attempt() -> None:
        try:
            _pool(tmp_path).open()
        except RuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=attempt)
    thread.start()
    thread.join(timeout=2)
    assert errors
    assert "main thread" in errors[0]
    assert not (tmp_path / "trace").exists()


@pytest.mark.parametrize("count", [0, 9, True])
def test_actor_count_is_bounded(tmp_path: Path, count: Any) -> None:
    with pytest.raises(ValueError, match=r"1\.\.8"):
        _pool(tmp_path, count)


def test_failed_root_admission_never_starts_or_modifies_foreign_files(tmp_path: Path) -> None:
    trace = tmp_path / "trace"
    trace.mkdir()
    retained = trace / "foreign.txt"
    retained.write_text("keep")
    pool = _pool(tmp_path)
    with pytest.raises(concurrent.NativeActorPoolError):
        pool.open()
    assert retained.read_text() == "keep"
    assert list(trace.iterdir()) == [retained]
    assert not pool._actors
