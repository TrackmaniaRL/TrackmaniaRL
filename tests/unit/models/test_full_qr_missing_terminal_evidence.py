"""Regress the missing-result path exposed by native diagnostic probe052."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.unit.trackmania.test_immediate_end_budget import ending_session
from trackmaniarl.trackmania.native_headless._runtime import codec_v7, env, outcome
from trackmaniarl.trackmania.native_headless._runtime.immediate import ImmediateSession


def _natural_end_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ImmediateSession, list[str]]:
    session, calls = ending_session(tmp_path, monkeypatch)
    session.max_trace_bytes = 1024**2
    session.local_posts = session.tmp / "posts"
    session.local_posts.mkdir()
    session.pins = {"explicit_test_identity": "not a game qualification"}
    # These portable tests isolate evidence assembly. The independent native
    # end-budget tests exercise the real Linux inode/quota walker separately.
    monkeypatch.setattr(session, "_reconcile_quota", lambda: None)
    monkeypatch.setattr(session, "check_budget", lambda additional=0: None)
    command = session.command

    def natural_reap(op: str) -> dict[str, Any]:
        return {**command(op), "truncated_by_driver": False}

    monkeypatch.setattr(session, "command", natural_reap)
    return session, calls


def test_missing_collector_is_preserved_and_never_becomes_a_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, calls = _natural_end_session(tmp_path, monkeypatch)
    session.end(truncated=False)
    assert calls == ["reap"]
    assert session.episode is None
    assert session.result_evidence["collectorHex"] is None
    assert session.result_evidence["validatorArrayHex"] == b"[".hex()
    assert not (session.local_posts / "result-123.bin").exists()
    assert (session.ep / "responses-v7.bin").read_bytes() == b"response"
    checks = SimpleNamespace(
        RUNTIME_PROFILE=codec_v7.PROFILE_ID, PINS=session.pins, require=env.require
    )
    packet = {"profile": codec_v7.PROFILE_ID, "pid": 123, "resultEvidence": session.result_evidence}
    classified = outcome.classify(packet, checks, tmp_path)
    assert classified["status"] == "inconclusive"
    assert classified["terminated"] is None
    assert classified["truncated"] is None
    assert classified["computedResult"] is None
    assert classified["reason"] == "result collector unavailable; socket EOF is not a terminal"


def test_present_collector_bytes_are_retained_without_claiming_their_validity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, _ = _natural_end_session(tmp_path, monkeypatch)
    raw = b"deliberately invalid collector computation fixture"
    (session.local_posts / "result-123.bin").write_bytes(raw)
    session.end(truncated=False)
    assert session.result_evidence["collectorHex"] == raw.hex()
    assert session.episode is None


def test_collector_io_error_is_not_hidden_as_a_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, _ = _natural_end_session(tmp_path, monkeypatch)
    collector = session.local_posts / "result-123.bin"
    original = Path.read_bytes

    def unreadable(path: Path) -> bytes:
        if path == collector:
            raise PermissionError("injected collector access failure")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", unreadable)
    with pytest.raises(PermissionError, match="collector access failure"):
        session.end(truncated=False)
