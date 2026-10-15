"""Offline checks for the new long-run guard; never acquire a game controller."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from experiments.jakub_1212 import deadline_guard, plan, runtime


def test_protocol_refuses_other_operator_and_detects_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plan, "PLAN", tmp_path / "plan.json")
    plan.prepare()
    assert plan.verify()["operator"] == "Jakub Szulc"
    with pytest.raises(ValueError, match="Jakub"):
        plan.stop_paths("borys")
    data = json.loads(plan.PLAN.read_text(encoding="utf-8"))
    data["maximum_run_hours"] = 999
    plan.PLAN.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RuntimeError, match="protocol settings"):
        plan.verify()


def test_guard_excludes_its_entire_forwarding_subtree() -> None:
    process = SimpleNamespace(
        pid=3, parents=lambda: [SimpleNamespace(pid=2), SimpleNamespace(pid=1)]
    )
    assert deadline_guard.guard_component(process, {2})
    assert not deadline_guard.guard_component(process, {9})


def test_guard_rejects_reused_pid_identity() -> None:
    identity = {"pid": os.getpid(), "create_time": psutil.Process().create_time() - 10}
    assert deadline_guard.exact(identity) is None


def test_existing_stop_is_preserved(tmp_path: Path) -> None:
    stop = tmp_path / "STOP"
    stop.write_text("human stop", encoding="utf-8")
    before = stop.stat().st_mtime_ns
    runtime.request_stop(stop, "another reason")
    assert stop.read_text(encoding="utf-8") == "human stop"
    assert stop.stat().st_mtime_ns == before
