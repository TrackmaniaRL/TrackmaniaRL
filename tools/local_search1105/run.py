"""Explicitly gated future CPU/native search. Never imported by offline preparation.

Each trial runs in an owned pidfd cohort with a fresh native reset. This module
is prepared, not native-qualified. The first five trials qualify the baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from .plans import confirmed_improvement
from .prepare import JIMMY, sha

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


def reserves(root: Path) -> None:
    for path, minimum in ((Path('/mnt/c'), 10), (BASE, 4), (root, 4)):
        require(shutil.disk_usage(path).free >= minimum * 2**30, 'storage reserve')
    require(sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) < 8 * 2**30,
            'search 8 GiB quota')


def worker(study: Path, trial: Path, row: int) -> None:
    import numpy as np
    import torch

    from trackmaniarl.experiments.shortcut_macro import (
        AutonomousMacroEnvironment,
        AutonomousMacroProvider,
        AutonomousMacroShortcutFactory,
        MacroCatalog,
        actual_goal_ms,
    )
    from trackmaniarl.trackmania.native_headless._runtime.ownership import (
        process_identity,
        same_identity,
    )

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    root = study / 'local-search1105'
    config = read(study / 'native' / JIMMY / 'factory-config.json')
    config.pop('training_demonstration', None)
    catalog_path = root / 'macro-catalog.npz'
    config['macro_catalog'] = {'path': str(catalog_path), 'sha256': sha(catalog_path)}
    config['runtime'].update(
        trace_dir=str(trial / 'traces'),
        work_dir=str(BASE / 'waypoint-runtime002/build/server-probe'
                     / ('search1105-' + trial.name)),
        timeout_s=150,
    )
    config.update(trace_archive_root=str(trial / 'closed-traces'), driver_episode_limit=1,
                  keep_trace_archives=1)
    with np.load(root / 'candidate-plans.npz', allow_pickle=False) as data:
        actions = [int(a) for a in data['macro_actions'][row]]
    catalog = MacroCatalog(config['macro_catalog'])
    neutral = catalog.programs.index((36,) * 5)
    factory = AutonomousMacroShortcutFactory(config)
    result: dict[str, Any] = {'row': row, 'status': 'FAILED', 'actual_goal_ms': None,
                              'native': [], 'learner_updates': 0}
    env = None
    started = time.monotonic()
    try:
        save(trial / 'factory-config.json', config)
        env = factory.create(seed=0)
        if not isinstance(env, AutonomousMacroEnvironment):
            raise TypeError('qualified macro environment required')
        env.reset(seed=0)
        provider = env.provider
        if not isinstance(provider, AutonomousMacroProvider) or provider._session is None:
            raise TypeError('qualified live macro provider required')
        result['native'].append(process_identity(provider._session.pid))
        for step in range(3000):
            require(time.monotonic() - started < 180 and not (trial / 'STOP').exists(),
                    'own trial deadline/STOP')
            _, reward, terminated, truncated, info = env.step(
                actions[step] if step < len(actions) else neutral)
            require(math.isfinite(reward), 'nonfinite reward')
            if terminated or truncated:
                result.update(status='FINISHED' if info['termination_reason'] == 'finished'
                              else 'DNF', result=info, steps=step + 1)
                if result['status'] == 'FINISHED':
                    result['actual_goal_ms'] = actual_goal_ms(env)
                    result['physical_race_time_ms'] = info['race_time_ms']
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


def supervise(study: Path) -> None:
    from trackmaniarl.experiments.full_qr.campaign_processes import CampaignProcessCohort
    from trackmaniarl.trackmania.native_headless._runtime.ownership import (
        process_identity,
        same_identity,
    )

    old = study / 'record-multi1096'
    audit = read(old / 'final-export/checkpoint-readback1100.json')
    require(audit['status'] == 'PASS_CPU_CHECKPOINT_READBACK_AND_POLICY_ONLY_EXPORT'
            and not audit['remaining_exact_identities'], '1096 final audit required')
    done = read(old / 'completion.json')
    require(done['phase'] == 'completed' and len(done['exits']) == 4
            and set(done['exits'].values()) == {0}, '1096 must be fully closed')
    require(all(not done['closure'][k] for k in ('remainingOwned', 'cleanupErrors', 'forced')),
            '1096 clean closure required')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    require(all(boot != done['boot'] or not same_identity(p, process_identity(p['pid']))
                for p in audit['identities_checked']), '1096 exact process identity still alive')
    require(read(old / 'power-guard-state.json')['phase'] == 'released',
            '1096 Windows power request must be released')
    root = study / 'local-search1105'
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
    started = time.monotonic()
    deadline = started + 3600
    history: list[dict[str, Any]] = []

    def trial(row: int) -> int | None:
        require(time.monotonic() + 240 <= deadline, 'total search budget')
        reserves(root)
        folder = execution / f'trial-{len(history):04d}'
        folder.mkdir()
        cohort = CampaignProcessCohort(folder / 'cohort',
                                      lambda: (folder / 'STOP').touch(exist_ok=True))
        report: dict[str, Any] = {'row': row}
        began = time.monotonic()
        try:
            report['identities'] = cohort.start(
                'search', [sys.executable, '-B', '-m', 'tools.local_search1105.run',
                           '--study', str(study), '--execute-after-1096',
                           '--worker', str(folder), '--row', str(row)],
                cwd=Path(__file__).resolve().parents[2])
            save(folder / 'launch.json', report)
            while not cohort.exited():
                require(time.monotonic() - began < 240, 'trial supervisor deadline')
                require(not (execution / 'STOP').exists(), 'own search STOP')
                reserves(root)
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
                        'actual_goal_ms': receipt['actual_goal_ms'], 'status': receipt['status']})
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
            if time.monotonic() + 1200 > deadline:
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
        else:
            status = 'QUEUE_COMPLETED'
    finally:
        save(execution / 'completion.json', {'status': status, 'best_goal_ms': best,
             'best_row': best_row, 'trials': len(history),
             'elapsed_seconds': time.monotonic() - started, 'learner_updates': 0})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, type=Path)
    parser.add_argument('--execute-after-1096', action='store_true')
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--row', type=int, default=0)
    args = parser.parse_args()
    require(args.execute_after_1096 and sys.platform == 'linux',
            'explicit future execution flag and Linux required')
    done = read(args.study / 'record-multi1096/completion.json')
    require(done['phase'] == 'completed' and set(done['exits'].values()) == {0}
            and len(done['exits']) == 4, 'run1096 must be closed even for worker invocation')
    require(all(not done['closure'][k] for k in ('remainingOwned', 'cleanupErrors', 'forced')),
            'run1096 closure required')
    if args.worker:
        require(args.worker.resolve().parent ==
                (args.study / 'local-search1105/execution').resolve(), 'private worker directory')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SOURCE))
    if args.worker:
        worker(args.study, args.worker, args.row)
    else:
        supervise(args.study)


if __name__ == '__main__':
    main()
