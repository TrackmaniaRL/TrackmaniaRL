import json

import pytest

from tools.headless_campaign_queue import legacy_handoff_watch as watcher


def _capture(tmp_path, monkeypatch):
    source = tmp_path / 'state.json'
    source.write_text(json.dumps({'job': 'sac'}), encoding='utf8')
    lock = tmp_path / 'lock'
    lock.mkdir()
    observation = {'schema': 'legacy-owner-observation-v1', 'job': {'pid': 35},
                   'supervisor': {'pid': 30}}
    monkeypatch.setattr(watcher, 'capture_running_legacy', lambda state, lock: observation)
    job = {'id': 'sac', 'transitions': 100}
    snapshot = watcher.capture_handoff(source, tmp_path / 'audit', lock, job, 'a' * 64)
    return snapshot, lock, job


def test_handoff_verifies_receipt_before_releasing_claim(tmp_path, monkeypatch):
    snapshot, lock, job = _capture(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(watcher, 'claim_released_lock', lambda observation, path: (10, 20))
    monkeypatch.setattr(watcher, 'release_claimed_lock', lambda path, identity: calls.append(('release', identity)))
    def verify(actual, digest, pid):
        calls.append(('verify', pid))
        return {'accepted_transitions': 100}
    result = watcher.attempt_handoff(snapshot, lock, job, 'a' * 64, verify=verify)
    assert result['accepted_transitions'] == 100
    assert calls == [('verify', 35), ('release', (10, 20))]
    assert json.loads(snapshot.read_text())['status'] == 'verified'


def test_failed_verification_keeps_lock_and_preserves_audit(tmp_path, monkeypatch):
    snapshot, lock, job = _capture(tmp_path, monkeypatch)
    monkeypatch.setattr(watcher, 'claim_released_lock', lambda observation, path: (10, 20))
    monkeypatch.setattr(watcher, 'release_claimed_lock', lambda *args: pytest.fail('must keep lock'))
    def verify(*args):
        raise ValueError('receipt budget mismatch')
    with pytest.raises(ValueError, match='budget'):
        watcher.attempt_handoff(snapshot, lock, job, 'a' * 64, verify=verify)
    assert lock.is_dir()
    assert json.loads(snapshot.read_text())['status'] == 'blocked_requires_audit'


def test_cannot_handoff_with_changed_plan_or_restart_completed_audit(tmp_path, monkeypatch):
    snapshot, lock, job = _capture(tmp_path, monkeypatch)
    monkeypatch.setattr(watcher, 'claim_released_lock', lambda *args: pytest.fail('claim forbidden'))
    with pytest.raises(ValueError, match='qualified plan'):
        watcher.attempt_handoff(snapshot, lock, dict(job, transitions=200), 'a' * 64)
    with pytest.raises(ValueError, match='qualified plan'):
        watcher.attempt_handoff(snapshot, lock, job, 'b' * 64)
