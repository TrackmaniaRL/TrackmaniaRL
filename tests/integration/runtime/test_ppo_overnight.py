"""Prospective timing policy must not waive identity or runtime validation."""

import copy
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psutil
import pytest

from pilot_tools import ppo_overnight as overnight


@pytest.fixture(autouse=True)
def archived_campaign_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.tmrl_test_comparison import campaign

    # Exercise historical pilot contracts without repinning or launching an old run.
    # Production verify must keep rejecting this newer checkout.
    recorded = json.loads(campaign.PLAN.read_text(encoding="utf-8"))
    monkeypatch.setattr(campaign, "verify", lambda: copy.deepcopy(recorded))


def test_final_timing_failure_remains_strict_failure(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint identity; no model restore")
    evaluation = {
        "status": "complete",
        "expected_trials": 10,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": overnight.pilot.sha(checkpoint),
        "trials": [
            {
                "trial_index": index,
                "map_uid": "oqIJ5rQDRrNwLPTh9H2p_W4tLof",
                "finished": index < 7,
                "finish_time_s": 40.0 + index if index < 7 else None,
                "termination_reason": "finish" if index < 7 else "timeout",
                "telemetry_error": None,
                "controller_error": None,
                "steps": 100,
                "step_race_time_ms_max": 150.0,
                "step_race_time_measurement_count": 100,
                "step_race_time_measurements_valid": True,
            }
            for index in range(10)
        ],
    }
    result = overnight.measurement(checkpoint, evaluation)
    assert result["finishes"] == 7
    assert not result["timing_gate_passed"]
    assert not result["strict_gate_passed"]
    assert result["measurement_valid"]
    for mutation in ("identity", "runtime", "missing_trial", "wrong_uid"):
        invalid = copy.deepcopy(evaluation)
        if mutation == "identity":
            invalid["checkpoint_sha256"] = "wrong"
        elif mutation == "runtime":
            invalid["trials"][0]["telemetry_error"] = "failure"
        elif mutation == "missing_trial":
            invalid["trials"].pop()
        else:
            invalid["trials"][0]["map_uid"] = "wrong"
        with pytest.raises(RuntimeError):
            overnight.measurement(checkpoint, invalid)


@pytest.mark.parametrize("variant", ["kl", "recovery"])
def test_variant_changes_only_declared_components_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
) -> None:
    from pilot_tools import ppo_pilot as pilot

    baseline = overnight.ORIGINAL_CONFIGURATION()
    monkeypatch.setattr(pilot, "configuration", pilot.configuration)
    monkeypatch.setattr(pilot, "_launch_owned", pilot._launch_owned)
    monkeypatch.setattr(pilot, "DIRECTORY", pilot.DIRECTORY)
    monkeypatch.setattr(pilot, "RUN_ID", pilot.RUN_ID)
    overnight.configure(variant, 17)
    candidate = pilot.configuration()
    if variant == "recovery":
        assert candidate["components"]["learner"] == baseline["components"]["learner"]
    candidate["run_id"] = baseline["run_id"]
    candidate["metadata"] = baseline["metadata"]
    for name in ("learner",):
        candidate["components"][name]["class_path"] = baseline["components"][name]["class_path"]
    candidate["components"]["additional_loggers"] = baseline["components"]["additional_loggers"]
    candidate["components"]["feature_pipeline"] = baseline["components"]["feature_pipeline"]
    assert candidate == baseline


def test_independent_hard_deadline_preserves_mismatched_identity(tmp_path: Path) -> None:
    from pilot_tools.ppo_deadline_guard import main

    owned = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        state = {
            "started_at": (datetime.now(UTC) - timedelta(seconds=5)).isoformat(),
            "launcher": {"pid": owned.pid, "create_time": psutil.Process(owned.pid).create_time()},
            "owned_identities": [{"pid": unrelated.pid, "create_time": 0}],
        }
        (tmp_path / "launch.json").write_text(json.dumps(state))
        (tmp_path / "status.json").write_text(json.dumps(state))
        (tmp_path / "plan.json").write_text(
            json.dumps(
                {"maximum_seconds": 1, "save_margin_seconds": 0, "source_root": str(tmp_path)}
            )
        )
        main(tmp_path)
        owned.wait(timeout=5)
        assert unrelated.poll() is None
        assert (tmp_path / "STOP").read_text() == "hard_deadline\n"
    finally:
        for process in (owned, unrelated):
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)


def test_guard_does_not_treat_own_forwarding_parent_as_orphan(tmp_path: Path) -> None:
    from pilot_tools.ppo_deadline_guard import main

    parent = psutil.Process().parent()
    assert parent is not None
    state = {
        "started_at": datetime.now(UTC).isoformat(),
        "status": "COMPLETE",
        "launcher": {"pid": 2147483647, "create_time": 0},
        "owned_identities": [{"pid": parent.pid, "create_time": parent.create_time()}],
    }
    (tmp_path / "launch.json").write_text(json.dumps(state))
    (tmp_path / "status.json").write_text(json.dumps(state))
    (tmp_path / "plan.json").write_text(
        json.dumps(
            {"maximum_seconds": 100, "save_margin_seconds": 10, "source_root": str(tmp_path)}
        )
    )
    main(tmp_path)
    assert not (tmp_path / "STOP").exists()
    assert json.loads((tmp_path / "deadline-guard-exit.json").read_text())["reason"] is None


def test_predecessor_stop_appearance_and_touch_win(tmp_path: Path) -> None:
    from pilot_tools.ppo_deadline_guard import predecessor_stop_changed
    from pilot_tools.ppo_pilot import sha

    stop = tmp_path / "STOP"
    plan = {"predecessor_stop": str(stop), "predecessor_stop_present": False}
    assert not predecessor_stop_changed(plan)
    stop.write_text("operator stop")
    assert predecessor_stop_changed(plan)
    plan.update(
        predecessor_stop_present=True,
        predecessor_stop_sha256=sha(stop),
        predecessor_stop_mtime_ns=stop.stat().st_mtime_ns,
    )
    assert not predecessor_stop_changed(plan)
    stat = stop.stat()
    os.utime(stop, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert predecessor_stop_changed(plan)
    stop.unlink()
    assert predecessor_stop_changed(plan)
    absent = {"prior_stop_pins": [plan]}
    assert predecessor_stop_changed(absent)
    absent["prior_stop_pins"][0] = {
        "predecessor_stop": str(stop),
        "predecessor_stop_present": False,
    }
    assert not predecessor_stop_changed(absent)
    stop.write_text("new stop")
    assert predecessor_stop_changed(absent)


def test_real_windows_guard_subtree_closes_without_post_complete_stop(tmp_path: Path) -> None:
    (tmp_path / "plan.json").write_text(
        json.dumps({"maximum_seconds": 30, "save_margin_seconds": 5, "source_root": str(tmp_path)})
    )
    script = """
import json,subprocess,sys,time,psutil
from pathlib import Path
from datetime import datetime,UTC
q=Path(sys.argv[1]);p=psutil.Process()
s={'started_at':datetime.now(UTC).isoformat(),'status':'RUNNING',
   'launcher':{'pid':p.pid,'create_time':p.create_time()},'owned_identities':[]}
(q/'launch.json').write_text(json.dumps(s));(q/'status.json').write_text(json.dumps(s))
guard=subprocess.Popen([sys.executable,'-m','pilot_tools.ppo_deadline_guard',str(q)],
                      creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
limit=time.monotonic()+10
while not (q/'deadline-guard-ready.json').exists():
 if time.monotonic()>limit:raise RuntimeError('guard not ready')
 time.sleep(.1)
s['status']='COMPLETE';(q/'status.json').write_text(json.dumps(s))
"""
    launcher = subprocess.Popen([sys.executable, "-c", script, str(tmp_path)])
    launcher.wait(timeout=15)
    deadline = time.monotonic() + 10
    receipt = tmp_path / "deadline-guard-exit.json"
    while not receipt.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert receipt.exists()
    assert json.loads(receipt.read_text())["reason"] is None
    assert not (tmp_path / "STOP").exists()
