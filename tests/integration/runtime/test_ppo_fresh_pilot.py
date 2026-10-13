"""Preparation and guard tests; no game control and no real training/checkpoint load."""

from __future__ import annotations

import copy
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pilot_tools import ppo_pilot as pilot
from pilot_tools.ppo_prefix import CampaignPrefixPPO
from trackmaniarl.algorithms.proximal_policy_optimization import ProximalPolicyOptimization
from trackmaniarl.trackmania.baseline import TelemetryPpoModel


def test_prefix_retains_all_71_campaign_learning_rates_and_optimizer_settings() -> None:
    torch.manual_seed(17)
    long = ProximalPolicyOptimization(TelemetryPpoModel(hidden_dim=8), execution={"device": "cpu"})
    torch.manual_seed(17)
    short = CampaignPrefixPPO(
        TelemetryPpoModel(hidden_dim=8),
        schedule_transitions=pilot.HORIZON,
        execution={"device": "cpu"},
    )
    long.setup({"seed": 17, "total_transitions": pilot.HORIZON})
    context = {"seed": 17, "total_transitions": pilot.BUDGET}
    short.setup(context)
    assert context["total_transitions"] == pilot.BUDGET
    assert short.total_transitions == long.total_transitions == pilot.HORIZON
    assert short.optimizer.defaults == long.optimizer.defaults
    assert all(
        torch.equal(value, short.model.state_dict()[key])
        for key, value in long.model.state_dict().items()
    )
    rates = []
    for update in range(71):
        short.processed_transitions = long.processed_transitions = update * 2048
        short._anneal_learning_rate()
        long._anneal_learning_rate()
        assert short._current_learning_rate() == long._current_learning_rate()
        rates.append(short._current_learning_rate())
    assert rates[0] == 0.0003
    assert rates[-1] == pytest.approx(0.00029475)
    assert not short.optimizer.state  # No optimizer step was taken.


@pytest.mark.parametrize("horizon", [0, -1, True, 1.5])
def test_invalid_schedule_horizon_fails_before_model_setup(horizon: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        CampaignPrefixPPO(schedule_transitions=horizon)


def test_prefix_cannot_exceed_campaign_budget() -> None:
    learner = CampaignPrefixPPO(schedule_transitions=2048)
    with pytest.raises(ValueError, match="within the campaign"):
        learner.setup({"total_transitions": 4096})


@pytest.fixture
def prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(pilot, "BASE", tmp_path / "artifacts")
    monkeypatch.setattr(pilot, "DIRECTORY", tmp_path / "artifacts/ppo-pilot")
    monkeypatch.setattr(pilot, "plugin_state", lambda: {"path": "bridge.op", "sha256": "pinned"})
    monkeypatch.setattr(
        pilot,
        "powershell_probe",
        lambda _: {"executable": "pwsh.exe", "version": "7.6.6", "policy": "RemoteSigned"},
    )
    return pilot.prepare(Path("pwsh.exe"))


def test_fresh_pilot_prepares_without_game_or_run_directory(prepared: dict) -> None:
    assert pilot.verify()["status"] == "PREPARED_NOT_LAUNCHED"
    assert prepared["expected_updates"] == 100
    assert prepared["evaluation_trials"] == 10
    assert prepared["learning_rate_schedule_transitions"] == 8192000
    assert prepared["controller_created"] is False
    assert not (pilot.BASE / pilot.RUN_ID).exists()
    assert not (pilot.DIRECTORY / "launch.json").exists()


def test_existing_pilot_is_never_overwritten(prepared: dict) -> None:
    original = (pilot.DIRECTORY / "plan.json").read_bytes()
    with pytest.raises(RuntimeError, match="never overwrite"):
        pilot.prepare(Path("pwsh.exe"))
    assert (pilot.DIRECTORY / "plan.json").read_bytes() == original


def test_verification_reports_actual_failed_launch(prepared: dict) -> None:
    (pilot.DIRECTORY / "status.json").write_text(
        json.dumps({"run_id": pilot.RUN_ID, "status": "STOPPED_OR_FAILED"})
    )
    assert pilot.observed_status(pilot.verify()) == "STOPPED_OR_FAILED"


def test_dead_supervisor_cannot_report_running(
    prepared: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    (pilot.DIRECTORY / "status.json").write_text(
        json.dumps({"run_id": pilot.RUN_ID, "status": "RUNNING",
                    "launcher": {"pid": 123, "create_time": 1.0}})
    )
    monkeypatch.setattr(pilot.psutil, "Process", lambda _: SimpleNamespace(create_time=lambda: 2.0))
    assert pilot.observed_status(pilot.verify()) == "SUPERVISOR_LOST"


def test_changed_plugin_blocks_verification(
    prepared: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pilot, "plugin_state", lambda: {"path": "bridge.op", "sha256": "new"})
    with pytest.raises(RuntimeError, match="telemetry plugin changed"):
        pilot.verify()


def test_duplicate_plugin_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pilot, "OPENPLANET_DIRECTORY", tmp_path)
    bridge = tmp_path / "Plugins/SAC_GetData.op"
    bridge.parent.mkdir()
    bridge.write_bytes(b"installed bridge")
    assert pilot.plugin_state()["sha256"] == pilot.sha(bridge)
    (tmp_path / "Plugins/TrackmaniaRL_Connect").mkdir()
    with pytest.raises(RuntimeError, match="Duplicate telemetry plugin"):
        pilot.plugin_state()


def test_status_retries_sharing_violation_without_losing_old_snapshot(
    prepared: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    status = pilot.DIRECTORY / "status.json"
    status.write_text('{"status":"old"}')
    replace = Path.replace
    calls = []

    def transient(path: Path, target: Path) -> Path:
        calls.append(path)
        if len(calls) < 3:
            assert json.loads(status.read_text())["status"] == "old"
            raise PermissionError("sharing violation")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", transient)
    monkeypatch.setattr(pilot.time, "sleep", lambda _: None)
    pilot._save_status({"status": "new"})
    assert len(calls) == 3
    assert json.loads(status.read_text())["status"] == "new"


def test_permanent_status_lock_has_bounded_retries(
    prepared: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def locked(_path: Path, _target: Path) -> Path:
        calls.append(1)
        raise PermissionError("permanent lock")

    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setattr(pilot.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError, match="permanent lock"):
        pilot._save_status({"status": "new"})
    assert len(calls) == 20


@pytest.mark.skipif(sys.platform != "win32", reason="Windows delete-sharing regression")
def test_status_handles_real_windows_reader_lock(prepared: dict) -> None:
    status = pilot.DIRECTORY / "status.json"
    status.write_text('{"status":"old"}')
    reader = status.open("r")
    release = threading.Timer(0.2, reader.close)
    release.start()
    try:
        pilot._save_status({"status": "new"})
    finally:
        release.join()
        reader.close()
    assert json.loads(status.read_text())["status"] == "new"


def test_supervisor_error_closes_cached_children_even_when_status_is_unwritable(
    prepared: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminated = []

    class Child:
        pid = 123

        def is_running(self) -> bool:
            return not terminated

        def children(self, *, recursive: bool) -> list:
            return []

        def create_time(self) -> float:
            return 1230.0

        def terminate(self) -> None:
            terminated.append(self.pid)

    child = Child()

    def failing_stage(_plan: dict, _state: dict, _command: tuple, owned: dict) -> None:
        owned[(child.pid, child.create_time())] = child
        raise PermissionError("status locked permanently")

    def wait(owned: object, *, timeout: float) -> list:
        assert (pilot.DIRECTORY / "STOP").exists()
        assert list(owned) == [child]
        return [] if terminated else [child]

    monkeypatch.setattr(pilot, "_run_stage_owned", failing_stage)
    monkeypatch.setattr(pilot, "wait_for_owned_processes", wait)
    with pytest.raises(PermissionError, match="locked permanently"):
        pilot._run_stage(prepared, {}, ("train-and-final-evaluation", [], 0))
    assert terminated == [123]
    failure = json.loads((pilot.DIRECTORY / "supervisor-failure.json").read_text())
    assert failure["status"] == "STOPPED_OR_FAILED"
    assert failure["surviving_identities"] == []


def test_changed_pilot_yaml_blocks_launch(prepared: dict) -> None:
    path = pilot.DIRECTORY / "ppo.yaml"
    path.write_text(path.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="configuration changed"):
        pilot.verify()


def test_changed_cap_cannot_extend_pilot(prepared: dict) -> None:
    prepared["maximum_seconds"] = 999999
    (pilot.DIRECTORY / "plan.json").write_text(json.dumps(prepared), encoding="utf-8")
    with pytest.raises(RuntimeError, match="decision contract changed"):
        pilot.verify()


def test_global_stop_blocks_before_shell_cuda_or_launch_file(
    prepared: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    stop = pilot.BASE / "STOP"
    stop.write_text("human stop", encoding="utf-8")
    monkeypatch.setattr(
        pilot, "powershell_probe", lambda _: pytest.fail("STOP precedes host actions")
    )
    with pytest.raises(RuntimeError, match="STOP active"):
        pilot.launch()
    assert stop.read_text(encoding="utf-8") == "human stop"
    assert not (pilot.DIRECTORY / "launch.json").exists()


def test_selected_shell_gets_own_module_path_without_overriding_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pilot.os,
        "environ",
        {"PSMODULEPATH": "incompatible modules", "PSExecutionPolicyPreference": "RemoteSigned"},
    )

    def query(args: list, **kwargs: object) -> SimpleNamespace:
        assert "-ExecutionPolicy" not in args
        assert not any(key.casefold() == "psmodulepath" for key in kwargs["env"])
        assert kwargs["env"]["PSExecutionPolicyPreference"] == "RemoteSigned"
        return SimpleNamespace(
            stdout=json.dumps({"version": "7.6.6", "policy": "RemoteSigned", "edition": "Core"})
        )

    monkeypatch.setattr(pilot.subprocess, "run", query)
    assert pilot.powershell_probe(Path("pwsh.exe"))["security_policy_changed"] is False
    assert pilot.os.environ["PSMODULEPATH"] == "incompatible modules"


@pytest.mark.parametrize(("version", "policy"), [("5.1", "RemoteSigned"), ("7.6.6", "Restricted")])
def test_incompatible_or_restricted_engine_is_rejected(
    version: str, policy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pilot.subprocess,
        "run",
        lambda *_, **__: SimpleNamespace(stdout=json.dumps({"version": version, "policy": policy})),
    )
    with pytest.raises(RuntimeError, match="PowerShell 7"):
        pilot.powershell_probe(Path("pwsh.exe"))


@pytest.fixture
def measurement(tmp_path: Path) -> tuple[Path, dict]:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test checkpoint identity, never loaded")
    return checkpoint, {
        "status": "complete",
        "expected_trials": 10,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": pilot.sha(checkpoint),
        "trials": [
            {
                "map_uid": pilot.MAP_UID,
                "trial_index": i,
                "finished": i < 8,
                "steps": 100,
                "step_race_time_ms_max": 50,
                "step_race_time_measurement_count": 100,
                "step_race_time_measurements_valid": True,
                "telemetry_error": None,
                "controller_error": None,
                "termination_reason": "finished" if i < 8 else "slow_progress",
            }
            for i in range(10)
        ],
    }


def test_gate_counts_finished_field_and_retains_a_valid_quality_failure(
    measurement: tuple[Path, dict],
) -> None:
    path, data = measurement
    assert pilot.pilot_measurement(path, data) == 8
    data["trials"][0]["finished"] = False
    assert pilot.pilot_measurement(path, data) == 7


def test_stage_requests_save_then_closes_its_process_at_hard_deadline(
    prepared: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"stop_reason": None}
    clock = iter([0, 13800, 14400])
    terminated = []

    class Parent:
        pid = 123
        active = True

        def create_time(self) -> float:
            return 1230.0

        def children(self, *, recursive: bool) -> list:
            return []

        def is_running(self) -> bool:
            return self.active

        def terminate(self) -> None:
            assert (pilot.DIRECTORY / "STOP").read_text(encoding="utf-8") == "save_deadline\n"
            terminated.append(self.pid)
            self.active = False

    parent = Parent()
    process = SimpleNamespace(returncode=None, pid=parent.pid)

    def poll() -> int | None:
        process.returncode = None if parent.active else -15
        return process.returncode

    process.poll = poll
    monkeypatch.setattr(pilot.subprocess, "Popen", lambda *_, **__: process)
    monkeypatch.setattr(pilot.psutil, "Process", lambda _: parent)
    monkeypatch.setattr(pilot.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(pilot.time, "sleep", lambda _: None)
    monkeypatch.setattr(pilot.shutil, "disk_usage", lambda _: SimpleNamespace(free=200 * 1024**3))
    monkeypatch.setattr(pilot, "wait_for_owned_processes", lambda *_, **__: [])
    with pytest.raises(RuntimeError, match="stage incomplete"):
        pilot._run_stage(prepared, state, ("train-and-final-evaluation", [], 0))
    assert terminated == [123]
    assert state["status"] == "STOPPED_OR_FAILED"
    assert state["owned_identities"] == [{"pid": 123, "create_time": 1230.0}]
    assert state["surviving_identities"] == []


@pytest.mark.parametrize(
    "fault", ["wrong_hash", "wrong_uid", "duplicate", "timing", "interruption", "partial"]
)
def test_pilot_rejects_invalid_measurement(measurement: tuple[Path, dict], fault: str) -> None:
    path, original = measurement
    data = copy.deepcopy(original)
    if fault == "wrong_hash":
        data["checkpoint_sha256"] = "wrong"
    elif fault == "wrong_uid":
        data["trials"][0]["map_uid"] = "other"
    elif fault == "duplicate":
        data["trials"][0]["trial_index"] = 1
    elif fault == "timing":
        data["trials"][0]["step_race_time_ms_max"] = 140
    elif fault == "interruption":
        data["trials"][0]["termination_reason"] = "operator_interruption"
    else:
        data["status"] = "cancelled"
    with pytest.raises(RuntimeError, match="incomplete or invalid"):
        pilot.pilot_measurement(path, data)
