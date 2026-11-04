import json

import pytest

from tools.headless_campaign_queue import legacy_handoff_watch as watcher


def _capture(tmp_path, monkeypatch):
    source = tmp_path / "state.json"
    source.write_text(json.dumps({"job": "sac"}), encoding="utf8")
    lock = tmp_path / "lock"
    lock.mkdir()
    observation = {
        "schema": "legacy-owner-observation-v1",
        "job": {"pid": 35},
        "supervisor": {"pid": 30},
    }
    monkeypatch.setattr(watcher, "capture_running_legacy", lambda state, lock: observation)
    job = {"id": "sac", "transitions": 100}
    snapshot = watcher.capture_handoff(source, tmp_path / "audit", lock, job, "a" * 64)
    return snapshot, lock, job


def test_handoff_verifies_receipt_before_releasing_claim(tmp_path, monkeypatch):
    snapshot, lock, job = _capture(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(watcher, "claim_released_lock", lambda observation, path: (10, 20))
    monkeypatch.setattr(
        watcher, "release_claimed_lock", lambda path, identity: calls.append(("release", identity))
    )

    def verify(actual, digest, pid):
        calls.append(("verify", pid))
        return {"accepted_transitions": 100}

    result = watcher.attempt_handoff(snapshot, lock, job, "a" * 64, verify=verify)
    assert result["accepted_transitions"] == 100
    assert calls == [("verify", 35), ("release", (10, 20))]
    assert json.loads(snapshot.read_text())["status"] == "verified"


def test_failed_verification_keeps_lock_and_preserves_audit(tmp_path, monkeypatch):
    snapshot, lock, job = _capture(tmp_path, monkeypatch)
    monkeypatch.setattr(watcher, "claim_released_lock", lambda observation, path: (10, 20))
    monkeypatch.setattr(
        watcher, "release_claimed_lock", lambda *args: pytest.fail("must keep lock")
    )

    def verify(*args):
        raise ValueError("receipt budget mismatch")

    with pytest.raises(ValueError, match="budget"):
        watcher.attempt_handoff(snapshot, lock, job, "a" * 64, verify=verify)
    assert lock.is_dir()
    assert json.loads(snapshot.read_text())["status"] == "blocked_requires_audit"


def test_cannot_handoff_with_changed_plan_or_restart_completed_audit(tmp_path, monkeypatch):
    snapshot, lock, job = _capture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        watcher, "claim_released_lock", lambda *args: pytest.fail("claim forbidden")
    )
    with pytest.raises(ValueError, match="qualified plan"):
        watcher.attempt_handoff(snapshot, lock, dict(job, transitions=200), "a" * 64)
    with pytest.raises(ValueError, match="qualified plan"):
        watcher.attempt_handoff(snapshot, lock, job, "b" * 64)


def test_handoff_reads_real_checkpoint_receipt_and_all_trials(tmp_path, monkeypatch):
    from tools.headless_campaign_queue.test_strict_supervisor import make_fixture, sha
    from trackmaniarl.core.builtins import TorchCheckpointCodec

    job, receipt, state, evaluation = make_fixture(tmp_path)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.unlink()
    TorchCheckpointCodec().save(state, checkpoint)
    evaluation_data = json.loads(evaluation.read_text())
    evaluation_data["checkpoint_sha256"] = sha(checkpoint)
    evaluation.write_text(json.dumps(evaluation_data))
    receipt.update(
        process_pid=35,
        manifest_sha256="a" * 64,
        checkpoint_sha256=sha(checkpoint),
        evaluation_sha256=sha(evaluation),
    )
    (tmp_path / "receipt.json").write_text(json.dumps(receipt))
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"job": "sac"}))
    lock = tmp_path / "lock"
    lock.mkdir()
    observation = {
        "schema": "legacy-owner-observation-v1",
        "job": {"pid": 35},
        "supervisor": {"pid": 30},
    }
    monkeypatch.setattr(watcher, "capture_running_legacy", lambda *args: observation)
    snapshot = watcher.capture_handoff(source, tmp_path / "audit", lock, job, "a" * 64)
    # Only process/lock ownership is simulated; receipt, checkpoint codec, hashes,
    # SQLite journal and all thirty outcomes follow the real verifier path.
    monkeypatch.setattr(watcher, "claim_released_lock", lambda *args: (10, 20))
    monkeypatch.setattr(watcher, "release_claimed_lock", lambda *args: None)
    assert watcher.attempt_handoff(snapshot, lock, job, "a" * 64)["accepted_transitions"] == 101
