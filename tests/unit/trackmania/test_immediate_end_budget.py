"""End-publication quota failures preserve evidence and independent cleanup."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from trackmaniarl.trackmania.native_headless._runtime.capture import RxPersistenceError
from trackmaniarl.trackmania.native_headless._runtime.immediate import ImmediateSession

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux owned quota publication")


def ending_session(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ImmediateSession, list[str]]:
    session = ImmediateSession.__new__(ImmediateSession)
    session.out, session.work, session.tmp = (root / name for name in ("trace", "work", "local"))
    for directory in (session.out, session.work, session.tmp):
        directory.mkdir()
    session.ep = session.out / "episode0"
    session.ep.mkdir()
    (session.ep / "stdout-slice.bin").write_bytes(b"[")
    session.episode, session.pid = 0, 123
    session.capture = SimpleNamespace(persisted=0, finish=lambda: None)
    session.connection = SimpleNamespace(close=lambda: None)
    session.responses, session.requests = [b"response"], [{"hex": "request", "host_ns": 1}]
    monkeypatch.setattr(session, "poll_posts", lambda: None)
    calls: list[str] = []

    def command(op: str) -> dict[str, Any]:
        calls.append(op)
        return {
            "kind": "reaped",
            "pid": 123,
            "process_gone": True,
            "parent_unchanged": True,
            "exit_status": None,
            "truncated_by_driver": True,
        }

    monkeypatch.setattr(session, "command", command)
    return session, calls


@pytest.mark.parametrize(
    "blocked", ["responses-v7.bin", "requests-v7.json", "host-summary-v7.json"]
)
def test_each_end_publication_checks_budget_before_its_write(
    tmp_path: Path, blocked: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, calls = ending_session(tmp_path, monkeypatch)
    preceding = 1  # Existing literal stdout opening.
    if blocked != "responses-v7.bin":
        preceding += len(b"response")
    if blocked == "host-summary-v7.json":
        preceding += len(json.dumps(session.requests, indent=2, allow_nan=False).encode()) + 1
    session.max_trace_bytes = preceding
    with pytest.raises(RxPersistenceError, match="trace budget"):
        session.end(truncated=True)
    assert calls == ["truncate"]
    assert not (session.ep / blocked).exists()
    assert (session.ep / "stdout-slice.bin").read_bytes() == b"["


def test_quota_blocked_trace_copy_cannot_suppress_zero_owned_cleanup_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = ImmediateSession.__new__(ImmediateSession)
    session.closed = False
    session.out, session.tmp = tmp_path / "trace", tmp_path / "retained-local"
    session.out.mkdir()
    session.tmp.mkdir()
    session.local_posts = session.tmp / "posts"
    session.local_posts.mkdir()
    original = session.local_posts / "post-123.bin"
    original.write_bytes(b"retained literal native evidence")
    cleanup = {"remainingOwned": [], "cleanupErrors": [], "returncode": -15}
    session.proc = SimpleNamespace(stop=lambda **_: cleanup, close_control=lambda: None)
    session.captures, session.commands, session.messages = [], [], []
    session.connection = session.listener = session.stderr = None

    def exhausted(additional: int = 0) -> None:
        raise RxPersistenceError("injected exhausted trace quota")

    monkeypatch.setattr(session, "check_budget", exhausted)
    session.close(error=True)
    report = json.loads((session.out / "close-v7.json").read_bytes())
    assert report["supervisor"]["remainingOwned"] == []
    assert report["supervisor"]["cleanupErrors"] == []
    assert report["traceAccounting"] is None
    assert any("exhausted trace quota" in failure for failure in report["failures"])
    assert original.read_bytes() == b"retained literal native evidence"
    assert not (session.out / "posts" / original.name).exists()
