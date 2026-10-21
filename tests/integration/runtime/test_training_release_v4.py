"""Portable release contracts and mocked process supervision; no live game calls."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.tmrl_test_comparison.release_20261021_v4 import plan, runtime
from experiments.tmrl_test_comparison.release_20261021_v4.lease import controller_lease
from trackmaniarl.core.spec import RunSpec


def test_plan_has_fifteen_portable_equal_budget_jobs() -> None:
    protocol = plan.verify()
    assert len(protocol["jobs"]) == 15
    for job in protocol["jobs"]:
        path = plan.ROOT / job["config"]
        spec = RunSpec.from_yaml(path)
        assert spec.training.total_transitions == 7_168_000
        assert spec.training.checkpoint_milestones == plan.MILESTONES
        assert spec.components.environment.kwargs["config"]["max_projected_speed_mps"] == 278.0
        assert spec.evaluation.trials_per_map == 30
        assert spec.evaluation.max_step_race_time_ms is None
        assert (path.parent / spec.artifacts_dir).resolve() == plan.BASE
        for asset in (spec.evaluation.maps[0].map_path, spec.evaluation.maps[0].geometry_path):
            assert (path.parent / asset).is_file()
        if job["algorithm"] == "ppo":
            assert spec.components.feature_pipeline.class_path.endswith(
                ":PersistentPpoRecoveryPipeline"
            )


def test_source_change_blocks_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    pins = dict(plan.source_pins())
    pins["trackmaniarl/trackmania/linux_reset.py"] = "changed"
    monkeypatch.setattr(plan, "source_pins", lambda: pins)
    with pytest.raises(RuntimeError, match="Frozen source"):
        plan.verify()


def test_existing_plan_never_overwritten() -> None:
    before = plan.PLAN.read_bytes()
    with pytest.raises(RuntimeError, match="already prepared"):
        plan.prepare()
    assert before == plan.PLAN.read_bytes()


def test_reused_identity_is_not_terminated(monkeypatch: pytest.MonkeyPatch) -> None:
    stale = SimpleNamespace(
        pid=42, is_running=lambda: False, terminate=lambda: pytest.fail("Reused PID signalled")
    )
    monkeypatch.setattr(runtime.psutil, "wait_procs", lambda *args, **kwargs: ([], []))
    assert runtime.close_owned({(42, 123.0): stale}) == []


def test_reused_live_pid_with_different_creation_time_is_not_signalled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reused = SimpleNamespace(
        pid=42,
        is_running=lambda: True,
        status=lambda: runtime.psutil.STATUS_RUNNING,
        create_time=lambda: 124.0,
        children=lambda **_: pytest.fail("Reused PID traversed"),
        terminate=lambda: pytest.fail("Reused PID signalled"),
        kill=lambda: pytest.fail("Reused PID signalled"),
    )
    monkeypatch.setattr(runtime.psutil, "wait_procs", lambda *args, **kwargs: ([], []))
    assert runtime.close_owned({(42, 123.0): reused}) == []


def test_game_use_requires_the_released_commit_in_checkout_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(plan.subprocess, "check_output", lambda *args, **kwargs: "a" * 40)
    monkeypatch.setattr(
        plan.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1)
    )
    with pytest.raises(RuntimeError, match="released library commit"):
        plan.require_release_tag()


def test_stopped_stage_never_spawns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stop = tmp_path / "STOP"
    stop.touch()
    monkeypatch.setattr(runtime, "assert_idle", lambda: None)
    monkeypatch.setattr(plan, "stop_paths", lambda _: [stop])
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Spawned"))
    with pytest.raises(RuntimeError, match="STOP"):
        runtime.watch_stage(
            ["unused"], {"state": {}, "owned": {}, "save_at": float("inf"), "worker": "borys"}
        )


def test_stop_waits_then_closes_only_owned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stop = tmp_path / "STOP"
    clock = [0.0]
    terminated = []

    class Captured:
        pid = 456
        active = True

        def create_time(self) -> float:
            return 123.0

        def is_running(self) -> bool:
            return self.active

        def status(self) -> str:
            return "running"

        def children(self, *, recursive: bool) -> list:
            return []

        def terminate(self) -> None:
            assert stop.exists()
            terminated.append(self.pid)
            self.active = False

        def kill(self) -> None:
            self.terminate()

    proc = Captured()
    child = SimpleNamespace(
        pid=456, poll=lambda: None if proc.active else -15, wait=lambda **kwargs: -15
    )

    def pause(_: float) -> None:
        clock[0] += 2
        stop.touch()

    monkeypatch.setattr(runtime, "assert_idle", lambda: None)
    monkeypatch.setattr(plan, "stop_paths", lambda _: [stop])
    monkeypatch.setattr(runtime.psutil, "Process", lambda _: proc)
    monkeypatch.setattr(runtime.psutil, "wait_procs", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(runtime.time, "sleep", pause)
    monkeypatch.setattr(runtime.shutil, "disk_usage", lambda _: SimpleNamespace(free=200 * 1024**3))
    context = {
        "state": {"stage": "train"},
        "owned": {},
        "directory": tmp_path,
        "status": tmp_path / "status.json",
        "worker": "borys",
        "save_at": 100,
        "hard_at": 110,
        "started": 0,
        "stop_at": None,
        "grace": 5,
        "stop": stop,
    }
    with pytest.raises(RuntimeError, match="Bounded shutdown"):
        runtime.watch_stage(["unused"], context)
    assert terminated == [456]
    assert context["state"]["stop_reason"] == "user_stop"
    assert context["state"]["surviving_identities"] == []


def test_real_controller_lock_excludes_another_process() -> None:
    code = (
        "from experiments.tmrl_test_comparison.release_20261021_v4.lease import controller_lease\n"
        "try:\n with controller_lease(): pass\nexcept RuntimeError: raise SystemExit(23)"
    )
    with controller_lease():
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=plan.ROOT, capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 23, result.stderr
    with controller_lease():
        pass
