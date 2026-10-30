"""Fresh CPU sequence search after the independently audited failed run1096.

Preserve its failure. Never resume a learner. Qualify five unchanged baselines
before edits, then confirm any improvement five times. Not a public WR or agent.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from tools.local_search1105.plans import confirmed_improvement
from tools.local_search1105.prepare import JIMMY, sha

BASE = Path('/home/kuba/tmrl-full-qr-20261022-011')
SOURCE = BASE / 'shortcut-multi-source1094'


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path: Path, value: object) -> None:
    temporary = path.with_suffix('.stage')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def require(condition: bool, message: str) -> None:  # noqa: FBT001
    if not condition:
        raise RuntimeError(message)


def clock() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def check_prior_documents(done: Any, audit: Any, windows: Any) -> None:
    require(done['phase'] == 'failed'
            and done['status'] == 'FAILED_OR_INCOMPLETE', 'preserve original1096 FAIL')
    require(done['exits'] == audit['exits'] == {
        'learner': 0, 'actor-000': 1, 'actor-001': 0, 'actor-002': 0},
        'explicit known failed run, not a generic failure exemption')
    require(audit['status'] == 'PASS_PRESERVATION_AUDIT_OF_FAILED_RUN_NOT_TRAINING_SUCCESS',
            'independent preservation audit required')
    for document in (done, audit):
        require(all(not document['closure'][k]
                    for k in ('remainingOwned', 'cleanupErrors', 'forced')), 'old cohort closure')
    require(not audit['remaining_exact_identities'], 'old identities must be closed')
    require(audit['finite_final_tensors'] == {'online': 90, 'target': 90, 'optimizers': 264},
            'preserved final checkpoint finite audit required')
    require(audit['final_checkpoint']['sha256'] ==
            done['result']['checkpoint']['sha256'] ==
            'e06c33110eacb3d5eaad00592210adaa422c94926dfca8fc200a3312da747a3c',
            'this exact failed campaign checkpoint only')
    require(audit['power_request']['phase'] == 'released'
            and windows['old_owner_identity_gone'] is True
            and windows['old_power_guard_identity_gone'] is True,
            'actual Windows owner and power guard closure required')


def verify_prior_release(study: Path, root: Path, *, full: bool = False) -> None:
    from trackmaniarl.trackmania.native_headless._runtime.ownership import (
        process_identity,
        same_identity,
    )

    done = read(study / 'record-multi1096/completion.json')
    audit = read(study / 'failure-audit1108/independent-audit.json')
    windows = read(root / 'admission.json')
    check_prior_documents(done, audit, windows)
    require(read(study / 'record-multi1096/power-guard-state.json')['phase'] == 'released',
            'original request remains released')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    require(all(boot != done['boot'] or not same_identity(p, process_identity(p['pid']))
                for p in audit['identities_checked']), 'old exact Linux identity alive')
    if full:
        require(sha(Path(audit['final_checkpoint']['path'])) == audit['final_checkpoint']['sha256'],
                'preserved source checkpoint hash changed')


def reserves(root: Path) -> None:
    for path, minimum in ((Path('/mnt/c'), 10), (BASE, 4), (root, 4)):
        require(shutil.disk_usage(path).free >= minimum * 2**30, 'storage reserve')
    total = 0
    for directory in (root, BASE / root.name):
        for path in directory.rglob('*'):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except FileNotFoundError:
                continue  # Atomic heartbeat replacements may disappear during a scan.
    require(total < 8 * 2**30, 'search 8 GiB quota')


def search_config(study: Path, root: Path, trial: Path) -> dict[str, Any]:
    config: dict[str, Any] = read(study / 'native' / JIMMY / 'factory-config.json')
    config.pop('training_demonstration', None)
    catalog_path = root / 'macro-catalog.npz'
    config['macro_catalog'] = {'path': str(catalog_path), 'sha256': sha(catalog_path)}
    config['runtime'].update(
        trace_dir=str(BASE / root.name / trial.name / 'traces'),
        work_dir=str(BASE / 'waypoint-runtime002/build/server-probe'
                     / (root.name + '-' + trial.name)), timeout_s=150,
    )
    config.update(trace_archive_root=str(trial / 'closed-traces'), driver_episode_limit=1,
                  keep_trace_archives=1)
    return config


def worker(study: Path, trial: Path, row: int) -> None:
    import numpy as np
    import torch

    from trackmaniarl.experiments.shortcut_macro import (
        AutonomousMacroEnvironment,
        AutonomousMacroProvider,
        MacroCatalog,
        actual_goal_ms,
    )
    from trackmaniarl.trackmania.native_headless._runtime.ownership import (
        process_identity,
        same_identity,
    )

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    root = trial.parent.parent
    config = search_config(study, root, trial)
    with np.load(root / 'candidate-plans.npz', allow_pickle=False) as data:
        actions = [int(a) for a in data['macro_actions'][row]]
    catalog = MacroCatalog(config['macro_catalog'])
    neutral = catalog.programs.index((36,) * 5)
    from .environment import SequenceSearchFactory

    factory = SequenceSearchFactory(config)
    result: dict[str, Any] = {'row': row, 'status': 'FAILED', 'actual_goal_ms': None,
                              'native': [], 'learner_updates': 0}
    env = None
    started = clock()
    try:
        save(trial / 'factory-config.json', config)
        save(trial / 'progress.json', {'phase': 'resetting', 'clock': clock(), 'steps': 0})
        env = factory.create(seed=0)
        if not isinstance(env, AutonomousMacroEnvironment):
            raise TypeError('qualified macro environment required')
        env.reset(seed=0)
        provider = env.provider
        if not isinstance(provider, AutonomousMacroProvider) or provider._session is None:
            raise TypeError('qualified live macro provider required')
        result['native'].append(process_identity(provider._session.pid))
        for step in range(3000):
            require(clock() - started < 180 and not (trial / 'STOP').exists(),
                    'own trial deadline/STOP')
            _, reward, terminated, truncated, info = env.step(
                actions[step] if step < len(actions) else neutral)
            require(math.isfinite(reward), 'nonfinite reward')
            if step % 10 == 0 or terminated or truncated:
                save(trial / 'progress.json', {
                    'phase': 'terminal' if terminated or truncated else 'driving',
                    'clock': clock(), 'steps': step + 1,
                    'observed_physics_time_ms': info['observed_physics_time_ms'],
                    'progress_pct': info['progress_pct'], 'learner_updates': 0,
                })
            if terminated or truncated:
                result.update(status='FINISHED' if info['termination_reason'] == 'finished'
                              else 'DNF', result=info, steps=step + 1)
                if result['status'] == 'FINISHED':
                    result['actual_goal_ms'] = actual_goal_ms(env)
                    result['physical_race_time_ms'] = info['observed_physics_time_ms']
                break
        else:
            result['status'] = 'DNF_BOUNDED'
    finally:
        if env is not None:
            env.close()
            if isinstance(env.provider, AutonomousMacroProvider):
                result['archives'] = env.provider.archive_receipts
        result['remaining_native'] = [p for p in result['native']
                                      if same_identity(p, process_identity(p['pid']))]
        save(trial / 'receipt.json', result)
    require(not result['remaining_native'], 'native closure')


def supervise(study: Path, root: Path) -> None:
    from trackmaniarl.experiments.full_qr.campaign_processes import CampaignProcessCohort
    from trackmaniarl.trackmania.native_headless._runtime.ownership import (
        process_identity,
    )

    verify_prior_release(study, root, full=True)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    plan = read(root / 'plan.json')
    # Prepared Windows input paths are mapped to /mnt drives for the Linux runner.
    for original, digest in plan['input_sha256'].items():
        normalized = original.replace('\\', '/')
        path = Path('/mnt/' + normalized[0].lower() + normalized[2:])
        require(sha(path) == digest, 'pinned source artifact changed')
    for name, digest in plan['output_sha256'].items():
        require(sha(root / name) == digest, 'candidate artifact changed')
    execution = root / 'execution'
    execution.mkdir(exist_ok=False)  # Never duplicate or resume a search.
    started = clock()
    deadline = started + 3600
    history: list[dict[str, Any]] = []
    parent = process_identity(os.getpid())

    def heartbeat(phase: str, **details: Any) -> None:
        save(execution / 'state.json', {
            'phase': phase, 'boot': boot, 'parent': parent, 'clock': 'monotonic_raw',
            'started': started, 'deadline': deadline, 'heartbeat': clock(),
            'trials': len(history), 'learner_updates': 0, **details,
        })

    heartbeat('qualifying_baseline')

    def trial(row: int) -> int | None:
        require(clock() + 240 <= deadline, 'total search budget')
        reserves(root)
        folder = execution / f'trial-{len(history):04d}'
        folder.mkdir()
        cohort = CampaignProcessCohort(folder / 'cohort',
                                      lambda: (folder / 'STOP').touch(exist_ok=True))
        report: dict[str, Any] = {'row': row}
        began = clock()
        try:
            report['identities'] = cohort.start(
                'search', [sys.executable, '-B', '-m', 'tools.wr_search1112.run',
                           '--study', str(study), '--execute-after-1096',
                           '--attempt', root.name,
                           '--worker', str(folder), '--row', str(row)],
                cwd=Path(__file__).resolve().parents[2])
            save(folder / 'launch.json', report)
            while not cohort.exited():
                require(clock() - began < 240, 'trial supervisor deadline')
                require(not (execution / 'STOP').exists(), 'own search STOP')
                reserves(root)
                heartbeat('trial_running', row=row, identities=report['identities'])
                time.sleep(0.5)
            report['exits'] = cohort.exited()
            require(report['exits'] == {'search': 0}, 'worker failure')
        finally:
            report['closure'] = cohort.close(grace_seconds=30)
            save(folder / 'supervision.json', report)
        require(all(not report['closure'][k]
                    for k in ('remainingOwned', 'cleanupErrors', 'forced')), 'trial closure')
        receipt = read(folder / 'receipt.json')
        require(receipt['status'] != 'FAILED' and not receipt['remaining_native'], 'native failure')
        history.append({'row': row, 'receipt': str(folder / 'receipt.json'),
                        'actual_goal_ms': receipt['actual_goal_ms'], 'status': receipt['status'],
                        'reason': receipt.get('result', {}).get('termination_reason'),
                        'progress_pct': receipt.get('result', {}).get('progress_pct')})
        save(execution / 'history.json', history)
        goal = receipt['actual_goal_ms']
        require(goal is None or (type(goal) is int and goal > 0), 'native Goal domain')
        return None if goal is None else int(goal)

    best = 17715
    best_row = 0
    status = 'FAILED'
    try:
        require([trial(0) for _ in range(5)] == [17715] * 5, 'baseline qualification failed')
        for candidate in plan['candidates']:
            # Reserve five complete bounded trials before testing a potential winner.
            if clock() + 1200 > deadline:
                status = 'BUDGET_STOP'
                break
            row = int(candidate['row'])
            first = trial(row)
            if first is not None and first < best:
                goals = [first, *(trial(row) for _ in range(4))]
                if confirmed_improvement(goals, best):
                    best, best_row = first, row
                    save(execution / 'best.json', {'row': row, 'native_goals_ms': goals,
                         'status': 'REPEATABLE_CONTROL_SEQUENCE_NOT_AUTONOMOUS_AGENT_OR_PUBLIC_WR',
                         'plan_sha256': sha(root / 'candidate-plans.npz')})
                    status = 'IMPROVEMENT_CONFIRMED'
                    break
        else:
            status = 'QUEUE_COMPLETED'
    finally:
        save(execution / 'completion.json', {'status': status, 'best_goal_ms': best,
             'best_row': best_row, 'trials': len(history),
             'elapsed_seconds': clock() - started, 'learner_updates': 0})
        heartbeat('failed' if status == 'FAILED' else 'completed', status=status,
                  best_goal_ms=best, best_row=best_row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, type=Path)
    parser.add_argument('--execute-after-1096', action='store_true')
    parser.add_argument('--attempt', default='local-search1113')
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--row', type=int, default=0)
    args = parser.parse_args()
    require(args.execute_after_1096 and sys.platform == 'linux',
            'explicit future execution flag and Linux required')
    require(re.fullmatch(r'local-search[0-9]{4}', args.attempt) is not None,
            'a fresh named controller within the study is required')
    root = args.study / args.attempt
    if args.worker:
        require(args.worker.resolve().parent ==
                (root / 'execution').resolve(), 'private worker directory')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SOURCE))
    verify_prior_release(args.study, root)
    if args.worker:
        worker(args.study, args.worker, args.row)
    else:
        supervise(args.study, root)


if __name__ == '__main__':
    main()
