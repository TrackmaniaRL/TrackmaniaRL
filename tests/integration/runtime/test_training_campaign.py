"""Campaign membership, frozen plans and STOP handling without game input."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from experiments.tmrl_test_comparison import campaign, run_pilots
from experiments.tmrl_test_comparison.generate import ALGORITHMS, configuration
from experiments.tmrl_test_comparison.policy import (
    CAMPAIGN_ASSIGNMENTS,
    require_standard_algorithm,
    require_standard_config,
)


@pytest.fixture
def prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    here = tmp_path / "experiments/tmrl_test_comparison"
    here.mkdir(parents=True)
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    monkeypatch.setattr(campaign, "HERE", here)
    monkeypatch.setattr(campaign, "PLAN", here / "campaign.json")
    monkeypatch.setattr(campaign, "source_pins", lambda: {"source.py": "frozen"})
    return campaign.prepare(campaign.LONG_TRANSITIONS)


@pytest.mark.parametrize("alias", ["sd-sac", "dsac", "discrete-sac"])
def test_experimental_alias_fails_before_queue_directory_creation(
    alias: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "never-created"
    monkeypatch.setattr(run_pilots.sys, "argv", ["run_pilots", str(directory), alias])
    with pytest.raises(ValueError, match="EXPERIMENTAL"):
        run_pilots.main()
    assert not directory.exists()


def test_campaign_excludes_sd_sac_and_preserves_action_groups() -> None:
    assert "sd-sac" not in ALGORITHMS
    assert "sd-sac" not in {value for group in CAMPAIGN_ASSIGNMENTS.values() for value in group}
    assert configuration("sac", 17, "full")["metadata"]["group"] == "continuous-3"
    assert configuration("qr", 17, "full")["metadata"]["group"] == "discrete-78"
    assert configuration("sd-sac", 17, "full")["metadata"]["support_tier"] == "experimental"
    with pytest.raises(ValueError, match="Unknown"):
        require_standard_algorithm("bogus")


def test_metadata_cannot_smuggle_experimental_learner_into_queue(tmp_path: Path) -> None:
    data = configuration("sd-sac", 17, "full")
    data["metadata"]["algorithm"] = "qr"
    path = tmp_path / "spoofed.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="EXPERIMENTAL"):
        require_standard_config(path)


def test_prepared_campaign_has_fifteen_independent_equal_budget_jobs(prepared: dict) -> None:
    assert campaign.verify()["status"] == "PREPARED_NOT_LAUNCHED"
    assert len(prepared["jobs"]) == 15
    assert len({job["run_id"] for job in prepared["jobs"]}) == 15
    assert prepared["total_transitions_per_run"] == 8_192_000
    assert prepared["assignments"]["borys"] == ["sac"]
    for job in prepared["jobs"]:
        spec = require_standard_config(campaign.ROOT / job["config"])
        assert spec.training.total_transitions % 2048 == 0
        assert spec.training.offline_pretrain_updates == 0
        assert spec.training.checkpoint_keep_last == 3
        assert spec.training.save_final_checkpoint
    assert not (campaign.ROOT / "artifacts").exists()


def test_preparation_never_overwrites_an_existing_campaign(prepared: dict) -> None:
    original = campaign.PLAN.read_bytes()
    with pytest.raises(RuntimeError, match="already prepared"):
        campaign.prepare(campaign.LONG_TRANSITIONS)
    assert campaign.PLAN.read_bytes() == original


@pytest.mark.parametrize("budget", [0, 145408, 8192001])
def test_invalid_long_budget_fails_before_writing(budget: int) -> None:
    with pytest.raises(ValueError, match="budget"):
        campaign.prepare(budget)


def test_modified_configuration_blocks_the_whole_campaign(prepared: dict) -> None:
    path = campaign.ROOT / prepared["jobs"][0]["config"]
    path.write_text(path.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="configuration changed"):
        campaign.verify()


def test_source_change_blocks_launch(prepared: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(campaign, "source_pins", lambda: {"source.py": "changed"})
    with pytest.raises(RuntimeError, match="Frozen campaign"):
        campaign.verify()


def test_duplicate_jobs_are_not_a_valid_plan(prepared: dict) -> None:
    prepared["jobs"][1] = prepared["jobs"][0]
    campaign.PLAN.write_text(json.dumps(prepared), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        campaign.verify()


def test_manual_continuation_cannot_skip_incomplete_checkpoint(
    prepared: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.tmrl_test_comparison import launch_checks

    def incomplete(_: object) -> None:
        raise RuntimeError("Checkpoint does not contain the completed training budget")

    monkeypatch.setattr(launch_checks, "final_checkpoint", incomplete)
    worker_jobs = campaign.remaining_jobs(prepared, "borys", None)
    with pytest.raises(RuntimeError, match="completed training budget"):
        campaign.remaining_jobs(prepared, "borys", worker_jobs[1]["run_id"])
    with pytest.raises(ValueError, match="assigned"):
        campaign.remaining_jobs(prepared, "borys", prepared["jobs"][0]["run_id"])


def test_host_stop_is_checked_before_cuda_or_private_key(
    prepared: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    stop = campaign.ROOT / "artifacts/tmrl-test-comparison" / campaign.CAMPAIGN_ID / "STOP"
    stop.parent.mkdir(parents=True)
    stop.write_text("user stop", encoding="utf-8")
    monkeypatch.setattr(campaign.platform, "system", lambda: "Windows")
    monkeypatch.setattr(campaign.platform, "python_version_tuple", lambda: ("3", "12", "0"))
    with pytest.raises(RuntimeError, match="STOP active"):
        campaign.host_check(prepared, "borys")
    assert stop.read_text(encoding="utf-8") == "user stop"


def test_supervisor_never_marks_zero_exit_with_stop_as_complete(
    prepared: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = next(job for job in prepared["jobs"] if job["worker"] == "borys")
    stop = campaign.ROOT / "global-stop"
    monkeypatch.setattr(campaign, "host_check", lambda *_: None)
    monkeypatch.setattr(campaign, "stop_paths", lambda _: [stop])
    monkeypatch.setattr(campaign, "wait_for_owned_processes", lambda *_, **__: [])
    monkeypatch.setattr(
        campaign.psutil,
        "Process",
        lambda _: SimpleNamespace(pid=123, create_time=lambda: 456.0, is_running=lambda: False),
    )

    def finish_with_stop(*_: object, **__: object) -> SimpleNamespace:
        stop.write_text("stop", encoding="utf-8")
        return SimpleNamespace(pid=123, returncode=0, poll=lambda: 0)

    monkeypatch.setattr(campaign.subprocess, "Popen", finish_with_stop)
    with pytest.raises(RuntimeError, match="STOPPED_OR_FAILED"):
        campaign.run_job(prepared, job)
    status = campaign.ROOT / "artifacts/tmrl-test-comparison" / campaign.CAMPAIGN_ID
    state = json.loads((status / job["run_id"] / "status.json").read_text(encoding="utf-8"))
    assert state["status"] == "STOPPED_OR_FAILED"
    assert state["owned_identities"] == [{"pid": 123, "create_time": 456.0}]


def test_changed_safety_limit_blocks_campaign(prepared: dict) -> None:
    prepared["automatic_resume"] = True
    campaign.PLAN.write_text(json.dumps(prepared), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol/limits"):
        campaign.verify()


def test_restricted_script_policy_blocks_before_launch_or_key(
    prepared: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(campaign.platform, "system", lambda: "Windows")
    monkeypatch.setattr(campaign.platform, "python_version_tuple", lambda: ("3", "12", "0"))
    calls = []

    def restricted(args: list, **_: object) -> str:
        calls.append(args[-1])
        return "Restricted\n"

    monkeypatch.setattr(campaign.subprocess, "check_output", restricted)
    with pytest.raises(RuntimeError, match="Windows blocks PowerShell scripts"):
        campaign.host_check(prepared, "borys")
    assert calls == ["Get-ExecutionPolicy"]
    assert not (campaign.ROOT / "artifacts").exists()


def test_runtime_cap_requests_save_then_closes_only_captured_processes(
    prepared: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = next(job for job in prepared["jobs"] if job["worker"] == "borys")
    terminated = []

    class Captured:
        def __init__(self, pid: int) -> None:
            self.pid = pid
            self.active = True

        def create_time(self) -> float:
            return float(self.pid * 10)

        def is_running(self) -> bool:
            return self.active

        def children(self, *, recursive: bool) -> list:
            return [child] if self is parent else []

        def terminate(self) -> None:
            assert (campaign.ROOT / "artifacts/STOP-sac").exists()
            terminated.append(self.pid)
            self.active = False

    parent, child = Captured(123), Captured(124)
    limit = prepared["maximum_run_hours"] * 3600
    clock = iter([0, limit + 1, limit + 2, limit + 603])
    process = SimpleNamespace(pid=123, returncode=-15, poll=lambda: None if parent.active else -15)
    monkeypatch.setattr(campaign, "host_check", lambda *_: None)
    monkeypatch.setattr(campaign, "wait_for_owned_processes", lambda *_, **__: [])
    monkeypatch.setattr(campaign.psutil, "Process", lambda _: parent)
    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *_, **__: process)
    monkeypatch.setattr(campaign.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        campaign.shutil, "disk_usage", lambda _: SimpleNamespace(free=200 * 1024**3)
    )
    with pytest.raises(RuntimeError, match="SHUTDOWN_TIMEOUT"):
        campaign.run_job(prepared, job)
    assert terminated == [124, 123]
    directory = campaign.ROOT / "artifacts/tmrl-test-comparison" / campaign.CAMPAIGN_ID
    state = json.loads((directory / job["run_id"] / "status.json").read_text(encoding="utf-8"))
    assert state["stop_reason"] == "runtime_cap"
    assert state["surviving_identities"] == []
    assert state["owned_identities"] == [
        {"pid": 123, "create_time": 1230.0},
        {"pid": 124, "create_time": 1240.0},
    ]


def test_unreadable_script_policy_blocks_without_running_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*_: object, **__: object) -> str:
        raise campaign.subprocess.CalledProcessError(1, ["Get-ExecutionPolicy"])

    monkeypatch.setattr(campaign.subprocess, "check_output", unavailable)
    with pytest.raises(RuntimeError, match="Cannot inspect PowerShell policy/modules"):
        campaign.check_script_policy()
