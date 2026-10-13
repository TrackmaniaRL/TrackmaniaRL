"""Prospective timing policy must not waive identity or runtime validation."""

import copy
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psutil
import pytest

from pilot_tools import ppo_overnight as overnight


def test_real_final_timing_failure_remains_strict_failure() -> None:
    path = overnight.PREDECESSOR.parent.parent / "evaluation.json"
    evaluation = json.loads(path.read_text())
    result = overnight.measurement(overnight.PREDECESSOR, evaluation)
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
            overnight.measurement(overnight.PREDECESSOR, invalid)


def test_kl_variant_changes_only_declared_components_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pilot_tools import ppo_pilot as pilot

    baseline = overnight.ORIGINAL_CONFIGURATION()
    monkeypatch.setattr(pilot, "configuration", pilot.configuration)
    monkeypatch.setattr(pilot, "_launch_owned", pilot._launch_owned)
    monkeypatch.setattr(pilot, "DIRECTORY", pilot.DIRECTORY)
    monkeypatch.setattr(pilot, "RUN_ID", pilot.RUN_ID)
    overnight.configure("kl", 17)
    candidate = pilot.configuration()
    candidate["run_id"] = baseline["run_id"]
    candidate["metadata"] = baseline["metadata"]
    for name in ("learner",):
        candidate["components"][name]["class_path"] = baseline["components"][name]["class_path"]
    candidate["components"]["additional_loggers"] = baseline["components"]["additional_loggers"]
    assert candidate == baseline


def test_independent_hard_deadline_preserves_mismatched_identity(tmp_path: Path) -> None:
    from pilot_tools.ppo_deadline_guard import main

    owned = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        state = {"started_at": (datetime.now(UTC)-timedelta(seconds=5)).isoformat(),
                 "launcher": {"pid": owned.pid,
                              "create_time": psutil.Process(owned.pid).create_time()},
                 "owned_identities": [{"pid": unrelated.pid, "create_time": 0}]}
        (tmp_path / "launch.json").write_text(json.dumps(state))
        (tmp_path / "status.json").write_text(json.dumps(state))
        (tmp_path / "plan.json").write_text(json.dumps({"maximum_seconds": 1,
            "save_margin_seconds": 0, "source_root": str(tmp_path)}))
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
    state = {"started_at": datetime.now(UTC).isoformat(), "status": "COMPLETE",
             "launcher": {"pid": 2147483647, "create_time": 0},
             "owned_identities": [{"pid": parent.pid, "create_time": parent.create_time()}]}
    (tmp_path / "launch.json").write_text(json.dumps(state))
    (tmp_path / "status.json").write_text(json.dumps(state))
    (tmp_path / "plan.json").write_text(json.dumps({"maximum_seconds": 100,
        "save_margin_seconds": 10, "source_root": str(tmp_path)}))
    main(tmp_path)
    assert not (tmp_path / "STOP").exists()
    assert json.loads((tmp_path / "deadline-guard-exit.json").read_text())["reason"] is None
