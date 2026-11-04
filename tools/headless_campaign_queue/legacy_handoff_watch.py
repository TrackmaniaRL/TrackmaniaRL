"""One-shot, fail-closed handoff audit for a legacy running queue job.

Capture must occur while the original supervisor and job are both alive. A later
attempt never deletes a pre-existing lock and verifies the receipt only after a
fresh lock is acquired. This module does not start a next training job.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.headless_campaign_queue.__main__ import atomic_json
from tools.headless_campaign_queue.legacy_handoff import (
    capture_running_legacy, claim_released_lock, release_claimed_lock,
)
from tools.headless_campaign_queue.strict_supervisor import verify_completion


def _job_sha(job: dict) -> str:
    return hashlib.sha256(json.dumps(job, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def capture_handoff(state_path: Path, output: Path, lock: Path, job: dict, digest: str) -> Path:
    """Capture live process identities to a fresh durable audit directory."""
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('expected SHA-256 manifest digest')
    if not state_path.is_absolute() or state_path.is_symlink() or not state_path.is_file():
        raise ValueError('unsafe legacy state path')
    state_raw = state_path.read_bytes()
    legacy_state = json.loads(state_raw)
    if legacy_state.get('job') != job.get('id'):
        raise ValueError('legacy state job differs from qualified plan')
    observation = capture_running_legacy(legacy_state, lock)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / 'handoff.json'
    atomic_json(snapshot, {'schema': 'legacy-handoff-v1', 'status': 'watching',
                           'observation': observation, 'manifest_sha256': digest,
                           'job_sha256': _job_sha(job),
                           'legacy_state_sha256': hashlib.sha256(state_raw).hexdigest()})
    return snapshot


def attempt_handoff(snapshot: Path, lock: Path, job: dict, digest: str, *, verify=verify_completion) -> dict:
    """Acquire a new lock only after job and original supervisor are gone.

    On receipt failure the newly acquired lock remains and the failure is saved.
    Successful audit releases the lock: starting the next job requires its own
    qualified plan and acquisition, and is intentionally not performed here.
    """
    if not snapshot.is_absolute() or snapshot.is_symlink() or not snapshot.is_file():
        raise ValueError('unsafe handoff state')
    state = json.loads(snapshot.read_text(encoding='utf-8'))
    if (state.get('schema') != 'legacy-handoff-v1' or state.get('status') != 'watching'
        or state.get('manifest_sha256') != digest or state.get('job_sha256') != _job_sha(job)):
        raise ValueError('handoff state does not match qualified plan')
    observation = state['observation']
    lock_identity = claim_released_lock(observation, lock)
    state.update(status='auditing', lock_device=lock_identity[0], lock_inode=lock_identity[1])
    atomic_json(snapshot, state)
    try:
        evidence = verify(job, digest, observation['job']['pid'])
        state.update(status='verified', evidence=evidence)
        atomic_json(snapshot, state)
        release_claimed_lock(lock, lock_identity)
    except BaseException as exc:
        state.update(status='blocked_requires_audit', error=f'{type(exc).__name__}: {exc}')
        atomic_json(snapshot, state)
        raise
    return evidence
