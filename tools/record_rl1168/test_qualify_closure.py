"""A qualification cannot pass when cleanup or supervisor closure fails."""

import json
from types import SimpleNamespace

import pytest

from tools.record_rl1168.archive import archive_closed_trace
from tools.record_rl1168.closure import native_closure_clean
from tools.record_rl1168.qualify import close_qualification_episode, qualification_status


class StubEnvironment:
    def __init__(self, trace_dir, close_error=None):
        self.settings = SimpleNamespace(trace_dir=trace_dir)
        self.close_error = close_error

    def close(self):
        if self.close_error:
            raise self.close_error


def clean_closure():
    return {
        "returncode": 0,
        "remainingOwned": [],
        "cleanupErrors": [],
        "cleanupPidfdSignals": [],
        "supervisorError": None,
    }


def test_clean_closure_is_recorded(tmp_path):
    closure = clean_closure()
    (tmp_path / "native-supervisor.json").write_text(json.dumps(closure), encoding="utf-8")
    report, row = {"cleanupErrors": []}, {}
    close_qualification_episode(StubEnvironment(tmp_path), report, row, 0)
    assert report["cleanupErrors"] == []
    assert row["closure"] == closure


def test_forced_stop_rejects_qualification(tmp_path):
    closure = {**clean_closure(), "cleanupPidfdSignals": [{"pid": 123}]}
    (tmp_path / "native-supervisor.json").write_text(json.dumps(closure), encoding="utf-8")
    report, row = {"cleanupErrors": []}, {}
    with pytest.raises(RuntimeError, match="native closure failed"):
        close_qualification_episode(StubEnvironment(tmp_path), report, row, 1)
    assert row["closure"] == closure
    assert report["cleanupErrors"] == [{"trial": 1, "stage": "supervisor", "closure": closure}]


def test_close_and_supervisor_read_errors_are_both_preserved(tmp_path):
    report = {"cleanupErrors": []}
    with pytest.raises(RuntimeError, match="native closure failed"):
        close_qualification_episode(
            StubEnvironment(tmp_path, OSError("close failed")), report, {}, 2
        )
    assert [error["stage"] for error in report["cleanupErrors"]] == ["env.close", "supervisor"]
    assert "close failed" in report["cleanupErrors"][0]["error"]
    assert "FileNotFoundError" in report["cleanupErrors"][1]["error"]


def test_missing_forced_signal_field_is_an_error(tmp_path):
    closure = clean_closure()
    del closure["cleanupPidfdSignals"]
    (tmp_path / "native-supervisor.json").write_text(json.dumps(closure), encoding="utf-8")
    report = {"cleanupErrors": []}
    with pytest.raises(RuntimeError, match="native closure failed"):
        close_qualification_episode(StubEnvironment(tmp_path), report, {}, 0)
    assert report["cleanupErrors"][0]["closure"] == closure


@pytest.mark.parametrize(
    "changes",
    [
        {"cleanupPidfdSignals": None},
        {"cleanupPidfdSignals": [{"pid": 1}]},
        {"supervisorError": "native failure"},
        {"returncode": None},
    ],
)
def test_native_closure_rejects_missing_or_unclean_evidence(changes):
    assert not native_closure_clean({**clean_closure(), **changes})


def test_archive_rejects_forced_cleanup_without_touching_trace(tmp_path):
    private = tmp_path / "private"
    trace = private / "traces" / "one"
    trace.mkdir(parents=True)
    closure = {**clean_closure(), "cleanupPidfdSignals": [{"pid": 123}]}
    (trace / "native-supervisor.json").write_text(json.dumps(closure), encoding="utf-8")
    output = tmp_path / "archive.tar.gz"
    with pytest.raises(ValueError, match="clean closure"):
        archive_closed_trace(private, trace, output)
    assert trace.exists()
    assert not output.exists()


def test_one_trial_is_not_the_three_trial_qualification():
    one = {"cleanupErrors": [], "trials": [{"status": "NATIVE_GOAL", "actual_goal_ms": 17601}]}
    assert qualification_status(one) == "PASS_ONE_FULL_START_BC"
    three = {"cleanupErrors": [], "trials": one["trials"] * 3}
    assert qualification_status(three) == "PASS_FULL_START_BC"
    bad = {"cleanupErrors": [], "trials": [{"status": "NO_GOAL", "actual_goal_ms": 17601}] * 3}
    assert qualification_status(bad) == "BC_NOT_QUALIFIED"
