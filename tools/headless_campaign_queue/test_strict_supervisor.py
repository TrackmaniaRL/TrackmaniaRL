import hashlib
import json
from pathlib import Path

import pytest

from tools.headless_campaign_queue import strict_supervisor as supervisor


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeCodec:
    def __init__(self, state):
        self.state = state

    def load(self, path):
        return self.state


def make_fixture(tmp_path):
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'checkpoint')
    evaluation = tmp_path / 'evaluation.json'
    evaluation.write_text(json.dumps({'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint), 'status': 'complete', 'expected_trials': 30,
        'trials': [{'map_id': 'map', 'map_uid': 'uid', 'trial_index': i,
                    'telemetry_error': None, 'controller_error': None, 'finished': True,
                    'finish_time_s': 34.0, 'termination_reason': 'finished', 'steps': 680,
                    'progress_pct': 100.0, 'reward': 1.0} for i in range(30)]}))
    receipt_path = tmp_path / 'receipt.json'
    receipt = {'job_id': 'sac', 'status': 'completed', 'qualification_only': False,
        'manifest_sha256': 'manifest', 'process_pid': 42, 'requested_transitions': 100,
        'transitions': 101, 'max_overshoot': 2, 'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha(checkpoint), 'evaluation': str(evaluation),
        'evaluation_sha256': sha(evaluation), 'run_fingerprint': 'fingerprint'}
    receipt_path.write_text(json.dumps(receipt))
    job = {'id': 'sac', 'receipt': str(receipt_path), 'transitions': 100, 'max_overshoot': 2,
           'run_fingerprint': 'fingerprint', 'map_id': 'map', 'map_uid': 'uid', 'on_policy': False}
    state = {'run_fingerprint': 'fingerprint', 'learner': {'weights': 1},
             'distributed': {'transitions': 101, 'updates': 1, 'update_credit': 0},
             'replay_store': {}, 'sampler': {}}
    return job, receipt, state, evaluation


def test_valid_receipt_and_artifacts(tmp_path):
    job, _, state, _ = make_fixture(tmp_path)
    assert supervisor.verify_completion(job, 'manifest', 42, codec=FakeCodec(state))['accepted_transitions'] == 101


@pytest.mark.parametrize('change', ['budget', 'fingerprint', 'trials', 'checkpoint_hash', 'pid', 'no_outcome', 'bad_time'])
def test_bad_evidence_is_rejected(tmp_path, change):
    job, receipt, state, evaluation = make_fixture(tmp_path)
    if change == 'budget': state['distributed']['transitions'] = 99
    elif change == 'fingerprint': state['run_fingerprint'] = 'wrong'
    elif change == 'trials':
        doc = json.loads(evaluation.read_text()); doc['trials'].pop(); evaluation.write_text(json.dumps(doc))
        receipt['evaluation_sha256'] = sha(evaluation); Path(job['receipt']).write_text(json.dumps(receipt))
    elif change == 'checkpoint_hash': Path(receipt['checkpoint']).write_bytes(b'tampered')
    elif change in ('no_outcome', 'bad_time'):
        doc = json.loads(evaluation.read_text())
        if change == 'no_outcome': del doc['trials'][0]['finished']
        else: doc['trials'][0]['finish_time_s'] = 0
        evaluation.write_text(json.dumps(doc))
        receipt['evaluation_sha256'] = sha(evaluation); Path(job['receipt']).write_text(json.dumps(receipt))
    pid = 43 if change == 'pid' else 42
    with pytest.raises(ValueError): supervisor.verify_completion(job, 'manifest', pid, codec=FakeCodec(state))


def test_valid_unsuccessful_trial_is_not_discarded(tmp_path):
    job, receipt, state, evaluation = make_fixture(tmp_path)
    doc = json.loads(evaluation.read_text())
    doc['trials'][0].update(finished=False, finish_time_s=None, termination_reason='no_progress', progress_pct=12.5)
    evaluation.write_text(json.dumps(doc))
    receipt['evaluation_sha256'] = sha(evaluation); Path(job['receipt']).write_text(json.dumps(receipt))
    assert supervisor.verify_completion(job, 'manifest', 42, codec=FakeCodec(state))['accepted_transitions'] == 101


def test_group_audit_retries_then_passes(monkeypatch):
    attempts = []
    def check(group):
        attempts.append(group)
        if len(attempts) < 3: raise ValueError('still alive')
    monkeypatch.setattr(supervisor, 'reject_surviving_group', check)
    monkeypatch.setattr(supervisor.time, 'sleep', lambda _: None)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    supervisor.wait_for_group_exit(42, boot)
    assert attempts == [42, 42, 42]


def test_group_audit_rejects_boot_change():
    with pytest.raises(ValueError, match='boot changed'):
        supervisor.wait_for_group_exit(42, 'not-this-boot')


def test_invalid_zero_exit_keeps_lock_and_never_starts_next(tmp_path, monkeypatch):
    job, _, _, _ = make_fixture(tmp_path)
    job.update(argv=['fake'], cwd=str(tmp_path))
    second = {**job, 'id': 'tqc'}
    started = []
    class Process:
        pid = 42
        def wait(self): return 0
    def spawn(*args, **kwargs):
        started.append(args)
        return Process()
    monkeypatch.setattr(supervisor.subprocess, 'Popen', spawn)
    monkeypatch.setattr(supervisor, 'process_identity', lambda _: {'pid': 42, 'process_group': 42, 'boot_id': 'boot', 'start_ticks': 1})
    monkeypatch.setattr(supervisor, 'wait_for_group_exit', lambda *args: None)
    monkeypatch.setattr(supervisor, 'reject_live_competitors', lambda: None)
    monkeypatch.setattr(supervisor, 'verify_completion', lambda *args: (_ for _ in ()).throw(ValueError('bad receipt')))
    output = tmp_path / 'output'
    lock = tmp_path / 'global-lock'
    with pytest.raises(ValueError, match='bad receipt'):
        supervisor.run({'schema_version': 2, 'jobs': [job, second]}, 'manifest', output, lock)
    assert len(started) == 1
    assert lock.is_dir()
    assert json.loads((output / 'strict-state.json').read_text())['status'] == 'blocked_requires_audit'


def test_verified_job_allows_next_and_releases_lock(tmp_path, monkeypatch):
    job, _, _, _ = make_fixture(tmp_path)
    job.update(argv=['fake'], cwd=str(tmp_path))
    jobs = [job, {**job, 'id': 'tqc'}]
    started = []
    class Process:
        pid = 42
        def wait(self): return 0
    monkeypatch.setattr(supervisor.subprocess, 'Popen', lambda *a, **kw: (started.append(a) or Process()))
    monkeypatch.setattr(supervisor, 'process_identity', lambda _: {'pid': 42, 'process_group': 42, 'boot_id': 'boot', 'start_ticks': 1})
    monkeypatch.setattr(supervisor, 'wait_for_group_exit', lambda *a: None)
    monkeypatch.setattr(supervisor, 'reject_live_competitors', lambda: None)
    monkeypatch.setattr(supervisor, 'verify_completion', lambda job, *a: {'verified': job['id']})
    output, lock = tmp_path / 'output', tmp_path / 'global-lock'
    assert supervisor.run({'schema_version': 2, 'jobs': jobs}, 'manifest', output, lock) == 0
    assert len(started) == 2 and not lock.exists()
    assert json.loads((output / 'strict-state.json').read_text())['status'] == 'completed'


def test_recovery_shutdown_waits_for_group(monkeypatch):
    from tools.headless_campaign_queue import recovery
    calls = []
    def check(group):
        calls.append(group)
        if len(calls) == 1: raise ValueError('still exiting')
    monkeypatch.setattr(recovery, 'reject_surviving_group', check)
    monkeypatch.setattr(supervisor.time, 'sleep', lambda _: None)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    recovery.wait_for_group_shutdown(123, boot)
    assert calls == [123, 123]
