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
    verify_trial_outcomes(evaluation)
    if evaluation.get('checkpoint_sha256') != receipt['checkpoint_sha256']:
        raise ValueError('evaluation checkpoint hash mismatch')
    return evidence


def verify_trial_outcomes(evaluation: dict) -> None:
    """Validate completed observations, allowing valid unsuccessful attempts."""
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
    if any('telemetry_error' not in trial or 'controller_error' not in trial or
           trial['telemetry_error'] is not None or trial['controller_error'] is not None
           for trial in evaluation['trials']):
        raise ValueError('evaluation runtime errors or missing error fields')


def verify_recovered_completion(job: dict, report_path: Path, *, codec=None) -> dict:
    """Audit an isolated recovery result without editing source or admitting a job.

    Call from the deployment matching job's fingerprint. This alone does not
    acquire a lock, resume a queue, or establish trust in the producing process.
    """
    from tools.headless_campaign_queue.recovery import sha, validate_state
    from tools.headless_campaign_queue.verify_artifacts import validate_drained_journal

    report = json.loads(report_path.read_text(encoding='utf-8'))
    if (report.get('kind') != 'isolated-full-recovery-v1'
        or report.get('status') != 'evaluation_verified_queue_pending'
        or report.get('stage') not in ('training', 'evaluation')
        or report.get('fingerprint') != job['run_fingerprint']
        or report.get('config_sha256') != job['config_sha256']
        or report.get('offline_restore_proof', {}).get('status') != 'PASS'
        or report.get('on_policy') is not job['on_policy']):
        raise ValueError('recovery report does not match qualified job')
    source_config = Path(report['config'])
    clone_config = Path(report['recovery_config'])
    checkpoint = Path(report['checkpoint'])
    source_checkpoint = Path(report['source_checkpoint'])
    evaluation_path = Path(report['evaluation'])
    for path in (report_path, source_config, clone_config, checkpoint, source_checkpoint, evaluation_path):
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ValueError('unsafe recovery evidence path')
    checkpoint_sha = sha(checkpoint)
    if (sha(source_config) != job['config_sha256']
        or sha(clone_config) != report.get('recovery_config_sha256')
        or sha(source_checkpoint) != report.get('source_checkpoint_sha256')
        or checkpoint_sha != report.get('checkpoint_sha256')
        or sha(evaluation_path) != report.get('evaluation_sha256')):
        raise ValueError('recovery evidence hash mismatch')
    if report.get('stage') == 'evaluation' and checkpoint_sha != report['source_checkpoint_sha256']:
        raise ValueError('evaluation-only recovery changed checkpoint')
    from trackmaniarl.core.fingerprint import run_fingerprint
    from trackmaniarl.core.spec import RunSpec
    if (run_fingerprint(RunSpec.from_yaml(source_config), source_config.parent) != job['run_fingerprint']
        or run_fingerprint(RunSpec.from_yaml(clone_config), clone_config.parent) != job['run_fingerprint']):
        raise ValueError('recovery source or runtime fingerprint mismatch')
    if codec is None:
        from trackmaniarl.core.builtins import TorchCheckpointCodec
        codec = TorchCheckpointCodec()
    state = codec.load(checkpoint)
    counters = validate_state(state, job['run_fingerprint'], job['transitions'], job['on_policy'],
                              stage='evaluation', max_overshoot=job['max_overshoot'])
    if not job['on_policy']:
        validate_drained_journal(checkpoint.parent.parent, state)
    evaluation = json.loads(evaluation_path.read_text(encoding='utf-8'))
    validate_evaluation(evaluation, checkpoint, job['map_id'], job['map_uid'], 30)
    verify_trial_outcomes(evaluation)
    if evaluation.get('checkpoint_sha256') != checkpoint_sha:
        raise ValueError('evaluation checkpoint hash mismatch')
    return {'report_sha256': sha(report_path), 'checkpoint_sha256': checkpoint_sha,
            'evaluation_sha256': sha(evaluation_path), 'accepted_transitions': counters['transitions']}


def recover_interrupted(job: dict, output: Path, entry: dict, state: dict, state_path: Path) -> dict:
    """Attempt one isolated FULL recovery while the supervisor owns the lock.

    A stopped job, an ambiguous checkpoint, pending spool or changed source fails
    closed. The source is never edited. Return only after final evaluation audit.
    """
    from tools.headless_campaign_queue import recovery
    from tools.headless_campaign_queue.disk_guard import check_disk_space
    from tools.headless_campaign_queue.job import monitor_disk
    from tools.headless_campaign_queue.verify_artifacts import final_checkpoint, validate_drained_journal
    from trackmaniarl.core.builtins import TorchCheckpointCodec
    from trackmaniarl.core.spec import RunSpec

    argv = job['argv']
    def arg(flag: str) -> str:
        if argv.count(flag) != 1:
            raise ValueError(f'missing or duplicate {flag}')
        index = argv.index(flag)
        if index + 1 >= len(argv):
            raise ValueError(f'missing value for {flag}')
        return argv[index + 1]

    config = Path(arg('--config'))
    stop = Path(arg('--stop-file'))
    config_sha = arg('--config-sha256')
    if not config.is_absolute() or not stop.is_absolute() or stop.exists():
        raise ValueError('recovery blocked by stop request or unsafe paths')
    if config_sha != recovery.sha(config) or arg('--run-fingerprint') != job['run_fingerprint']:
        raise ValueError('qualified config changed')
    spec = RunSpec.from_yaml(config)
    run_dir = (config.parent / spec.artifacts_dir / spec.run_id).resolve()
    checkpoint_dir = (config.parent / (getattr(spec, 'checkpoint_root', None) or spec.artifacts_dir)
                      / spec.run_id / 'checkpoints').resolve()
    checkpoint, _ = final_checkpoint(run_dir, checkpoint_dir, on_policy=job['on_policy'])
    destination = output / (job['id'] + '-recovery-0001')
    entry['recovery'] = {'status': 'preparing', 'destination': str(destination),
                         'source_checkpoint': str(checkpoint)}
    atomic_json(state_path, state)
    recovery.reject_live_competitors()
    report = recovery.prepare(config, checkpoint, destination, config_sha)
    if report['fingerprint'] != job['run_fingerprint']:
        raise ValueError('recovery fingerprint differs from qualified job')
    clone_config = Path(report['recovery_config'])
    clone_spec = RunSpec.from_yaml(clone_config)
    check_disk_space(clone_spec, destination)
    recovery_stop = destination / 'STOP'
    command = [sys.executable, '-m', 'trackmaniarl', 'resume', str(clone_config),
               report['checkpoint'], '--stop-file', str(recovery_stop)]
    with (destination / 'stdout.log').open('xb') as out, (destination / 'stderr.log').open('xb') as err:
        with monitor_disk(clone_spec, destination, recovery_stop):
            env = os.environ.copy(); env['WANDB_MODE'] = 'offline'
            process = subprocess.Popen(command, cwd=job['cwd'], stdout=out, stderr=err,
                                       start_new_session=True, env=env)
            identity = process_identity(process.pid)
            entry['recovery'].update(status='running', identity=identity)
            atomic_json(state_path, state)
            code = process.wait()
    wait_for_group_exit(identity['process_group'], identity['boot_id'])
    recovery.reject_live_competitors()
    if code != 0 or recovery_stop.exists() or stop.exists():
        raise ValueError('isolated recovery interrupted; evidence retained')
    cloned_run = destination / 'artifacts' / spec.run_id
    final, logged_updates = final_checkpoint(cloned_run, cloned_run / 'checkpoints', on_policy=job['on_policy'])
    final_state = TorchCheckpointCodec().load(final)
    counters = recovery.validate_state(final_state, job['run_fingerprint'], job['transitions'],
                                       job['on_policy'], stage='evaluation', max_overshoot=job['max_overshoot'])
    if counters['updates'] != logged_updates:
        raise ValueError('recovery final checkpoint update mismatch')
    if not job['on_policy']:
        validate_drained_journal(cloned_run, final_state)
    report.update(status='training_verified_evaluation_pending', checkpoint=str(final),
                  recovery_training_process=identity, recovery_training_exit_code=code)
    atomic_json(destination / 'recovery.json', report)
    recovery.evaluate_copy(clone_spec, destination, final, recovery_stop, report)
    recovery.reject_live_competitors()
    if recovery_stop.exists() or stop.exists():
        raise ValueError('stop requested during evaluation')
    qualified = dict(job, config_sha256=config_sha)
    evidence = verify_recovered_completion(qualified, destination / 'recovery.json')
    entry['recovery'].update(status='verified', evidence=evidence)
    atomic_json(state_path, state)
    return evidence


def run(plan: dict, digest: str, output: Path, global_lock: Path, *, auto_recover: bool = False) -> int:
    """Start fresh jobs in order, failing closed and preserving all evidence."""
    if os.name == 'nt' or not Path('/proc').is_dir():
        raise ValueError('Linux /proc is required')
    output = output.resolve()
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
            try:
                if code != 0:
                    raise ValueError(f"{job['id']}: process exit {code}; recovery audit required")
                entry['evidence'] = verify_completion(job, digest, process.pid)
            except (OSError, ValueError) as exc:
                entry.update(status='failed_before_recovery', returncode=code, error=str(exc))
                atomic_json(state_path, state)
                if not auto_recover:
                    raise
                entry['evidence'] = recover_interrupted(job, output, entry, state, state_path)
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
    parser.add_argument('--auto-recover', action='store_true', help='attempt one isolated FULL recovery; requires live qualification')
    args = parser.parse_args()
    try:
        plan, digest = prepare(args.manifest)
        if not args.execute:
            print(f'Preflight OK: {len(plan["jobs"])} jobs, manifest {digest}; nothing started')
            return 0
        return run(plan, digest, args.output, Path.home() / '.trackmaniarl-headless-campaign.lock',
                   auto_recover=args.auto_recover)
    except (OSError, ValueError) as exc:
        print(f'Supervisor blocked: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
