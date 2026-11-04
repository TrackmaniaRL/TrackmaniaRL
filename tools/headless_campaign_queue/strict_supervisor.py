"""Fail-closed sequential supervisor for qualified Linux campaign jobs.

Run only against trusted artifacts and a newly compiled plan. A stopped or failed
job is preserved for explicit audited recovery; no automatic replay is claimed.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from tools.headless_campaign_queue.__main__ import atomic_json, now, verify_receipt
from tools.headless_campaign_queue.recovery import process_identity, reject_live_competitors, reject_surviving_group
from tools.headless_campaign_queue.verify_artifacts import validate_checkpoint, validate_evaluation


def wait_for_group_exit(group: int, boot_id: str, *, timeout: float = 20.0, interval: float = 0.25) -> None:
    """Wait briefly for descendants; never kill or assume PID alone is identity."""
    deadline = time.monotonic() + timeout
    while True:
        if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != boot_id:
            raise ValueError('boot changed during process audit')
        try:
            reject_surviving_group(group)
            return
        except ValueError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(interval)


def verify_completion(job: dict, digest: str, pid: int, *, codec=None) -> dict:
    """Recheck producer receipt, trusted checkpoint contents and all 30 trials."""
    for key in ('run_fingerprint', 'map_id', 'map_uid'):
        if not isinstance(job.get(key), str) or not job[key]:
            raise ValueError(f'missing trusted plan field {key}')
    if type(job.get('on_policy')) is not bool:
        raise ValueError('missing trusted plan field on_policy')
    evidence = verify_receipt(job, digest, pid)
    receipt = json.loads(Path(job['receipt']).read_text(encoding='utf-8'))
    if receipt.get('run_fingerprint') != job['run_fingerprint']:
        raise ValueError('receipt fingerprint mismatch')
    if codec is None:
        from trackmaniarl.core.builtins import TorchCheckpointCodec
        codec = TorchCheckpointCodec()
    state = codec.load(Path(receipt['checkpoint']))
    actual = validate_checkpoint(state, job['run_fingerprint'], job['transitions'],
                                 on_policy=job['on_policy'], max_overshoot=job['max_overshoot'])
    if actual != receipt['transitions']:
        raise ValueError('receipt and checkpoint transition counters differ')
    evaluation = json.loads(Path(receipt['evaluation']).read_text(encoding='utf-8'))
    validate_evaluation(evaluation, Path(receipt['checkpoint']), job['map_id'], job['map_uid'], 30)
    if evaluation.get('status') != 'complete' or evaluation.get('expected_trials') != 30:
        raise ValueError('evaluation suite is incomplete')
    for trial in evaluation['trials']:
        finished = trial.get('finished')
        reason = trial.get('termination_reason')
        steps = trial.get('steps')
        progress = trial.get('progress_pct')
        reward = trial.get('reward')
        finish_time = trial.get('finish_time_s')
        if (type(finished) is not bool or not isinstance(reason, str) or not reason.strip()
            or reason == 'unknown' or type(steps) is not int or steps <= 0
            or isinstance(progress, bool) or not isinstance(progress, (int, float))
            or not math.isfinite(progress) or not 0 <= progress <= 100
            or isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward)):
            raise ValueError('evaluation trial outcome missing or invalid')
        if finished and (reason != 'finished' or isinstance(finish_time, bool)
            or not isinstance(finish_time, (int, float)) or not math.isfinite(finish_time) or finish_time <= 0):
            raise ValueError('finished evaluation trial has invalid finish time or reason')
        if not finished and reason == 'finished':
            raise ValueError('unfinished evaluation trial has finished reason')
    if evaluation.get('checkpoint_sha256') != receipt['checkpoint_sha256']:
        raise ValueError('evaluation checkpoint hash mismatch')
    if any('telemetry_error' not in trial or 'controller_error' not in trial or
           trial['telemetry_error'] is not None or trial['controller_error'] is not None
           for trial in evaluation['trials']):
        raise ValueError('evaluation runtime errors or missing error fields')
    return evidence


def run(plan: dict, digest: str, output: Path, global_lock: Path) -> int:
    """Start fresh jobs in order, failing closed and preserving all evidence."""
    if os.name == 'nt' or not Path('/proc').is_dir():
        raise ValueError('Linux /proc is required')
    jobs = plan.get('jobs')
    if plan.get('schema_version') != 2 or not isinstance(jobs, list) or not jobs:
        raise ValueError('qualified schema 2 plan required')
    seen = set()
    for job in jobs:
        name = job.get('id')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', name) or name in seen:
            raise ValueError('invalid or duplicate job identity')
        seen.add(name)
        for field in ('run_fingerprint', 'map_id', 'map_uid'):
            if not isinstance(job.get(field), str) or not job[field]:
                raise ValueError(f'missing trusted plan field {field}')
        if type(job.get('on_policy')) is not bool or type(job.get('max_overshoot')) is not int:
            raise ValueError('invalid trusted plan fields')
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / 'strict-state.json'
    if state_path.exists():
        raise ValueError('existing state requires explicit recovery audit')
    global_lock.mkdir()  # never remove an existing lock
    release = False
    state = {'status': 'running', 'manifest_sha256': digest, 'started_at': now(),
             'jobs': [{'id': j['id'], 'status': 'pending'} for j in jobs]}
    try:
        atomic_json(state_path, state)
        for job, entry in zip(jobs, state['jobs'], strict=True):
            reject_live_competitors()
            env = os.environ.copy()
            env.update(WANDB_MODE='offline', TRACKMANIARL_QUEUE_JOB_ID=job['id'],
                       TRACKMANIARL_QUEUE_MANIFEST_SHA256=digest)
            with (output / (job['id'] + '.stdout.log')).open('xb') as out, \
                 (output / (job['id'] + '.stderr.log')).open('xb') as err:
                process = subprocess.Popen(job['argv'], cwd=job['cwd'], env=env, stdout=out,
                                           stderr=err, start_new_session=True, shell=False)
                identity = process_identity(process.pid)
                entry.update(status='running', identity=identity, started_at=now())
                atomic_json(state_path, state)
                code = process.wait()
            wait_for_group_exit(identity['process_group'], identity['boot_id'])
            reject_live_competitors()
            if code != 0:
                raise ValueError(f"{job['id']}: process exit {code}; recovery audit required")
            entry['evidence'] = verify_completion(job, digest, process.pid)
            entry.update(status='verified', finished_at=now(), returncode=code)
            atomic_json(state_path, state)
        state.update(status='completed', finished_at=now())
        atomic_json(state_path, state)
        release = True
        return 0
    except BaseException as exc:
        state.update(status='blocked_requires_audit', error=f'{type(exc).__name__}: {exc}', finished_at=now())
        atomic_json(state_path, state)
        raise
    finally:
        if release:
            global_lock.rmdir()


def main() -> int:
    import argparse
    from tools.headless_campaign_queue.campaign import prepare

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    try:
        plan, digest = prepare(args.manifest)
        if not args.execute:
            print(f'Preflight OK: {len(plan["jobs"])} jobs, manifest {digest}; nothing started')
            return 0
        return run(plan, digest, args.output, Path.home() / '.trackmaniarl-headless-campaign.lock')
    except (OSError, ValueError) as exc:
        print(f'Supervisor blocked: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
