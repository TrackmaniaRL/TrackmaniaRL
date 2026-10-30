"""One fresh, bounded CPU line-search campaign. No online RL and no Windows game."""
from __future__ import annotations
import argparse
import datetime as dt
import gzip
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from backend import BASE, clock, descendants, identity, reserves, run_batch, save, sha
from core import digest, mutate, neighbourhood, validate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--qualification', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root / 'plan.json').read_text())
    private = BASE / root.name
    if (root / 'state.json').exists() or private.exists():
        raise RuntimeError('Never duplicate or resume a controller')
    private.mkdir(mode=0o700)
    (root / 'parent-launch-marker').write_text(str(os.getpid()))
    for name, expected in plan['source_sha256'].items():
        if sha(root / 'source' / name) != expected:
            raise RuntimeError('Frozen controller fingerprint')
    shutil.copyfile(Path(plan['writer']), private / 'native-line1134')
    (private / 'native-line1134').chmod(0o700)
    baseline = tuple(json.loads((root / 'baseline-motors.json').read_text()))
    validate(baseline)
    if digest(baseline) != 'e5eabd9626604f09d56b7a3b139b5ca03ff50d8ed04ac47081e94f7c1e35bf35':
        raise RuntimeError('Source17.712 motor fingerprint')
    state: dict[str, Any] = dict(phase='qualifying', parent=identity(os.getpid()),
                                boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                                baseline_goal_ms=17712, best_goal_ms=17712,
                                best_autonomous_goal_ms=plan.get('initial_best_autonomous_goal_ms',17712), learner_updates=0,
                                batches=0, candidate_evaluations=0, native_records=0,
                                confirmed_improvements=0, source_shim_loaded=False,
                                windows_game_used=False, public_submission=False,
                                remainingOwned=[], cleanupErrors=[], forced=[])
    begun = clock()
    ready: float | None = None
    deadline = begun + 240
    batch = 0
    seen = {digest(baseline)}
    if 'qualified_seed' in plan:
        seen.update(json.loads((root/'previously-evaluated-input-sha256.json').read_text()))
    best = baseline
    best_time = 17712
    elite: list[tuple[int, tuple[int, ...]]] = [(best_time, best)]
    rng = random.Random(plan['seed'])
    last_improvement_batch = 0

    def stopped() -> bool:
        return (root / 'STOP').exists() or clock() >= deadline

    def heartbeat(extra: dict[str, Any] | None = None) -> None:
        now = clock()
        state.update(clock_monotonic_raw=now, checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                     elapsed_seconds=now-(ready if ready is not None else begun))
        if extra:
            state['active_native_batch'] = extra
        save(root / 'state.json', state)

    def execute(jobs: list[tuple[str, tuple[int, ...]]]
                ) -> tuple[dict[str, int | None], dict[str, Any]]:
        nonlocal batch
        batch += 1
        results, receipt = run_batch(root, private, batch, baseline, jobs, stopped, heartbeat)
        values = dict(results)
        state['batches'] = batch
        state['native_records'] += len(results)
        state['active_native_batch'] = None
        heartbeat()
        return values, receipt

    def retain(ticks: tuple[int, ...], goal: int, receipt: dict[str, Any]) -> None:
        nonlocal best, best_time, last_improvement_batch
        best, best_time = ticks, goal
        state['best_goal_ms'] = goal
        state['confirmed_improvements'] += 1
        last_improvement_batch = batch
        winner = root / 'winners' / f'winner-{state["confirmed_improvements"]:04d}-{goal}ms'
        winner.mkdir(parents=True)
        save(winner / 'motors.json', list(ticks))
        # Actual fitness comes from the native engine; this file is explicitly input-only.
        source = Path(receipt['batch_best']['input_only_ghost'])
        shutil.copyfile(source, winner / 'inputs-NOT-VISUAL.Ghost.Gbx')
        binding = dict(actual_native_goal_ms=goal, motor_sha256=digest(ticks),
                       confirmation_batch=batch, confirmation_goals_ms=[goal] * 5,
                       input_only_ghost=str(winner / 'inputs-NOT-VISUAL.Ghost.Gbx'),
                       input_only_ghost_sha256=sha(winner / 'inputs-NOT-VISUAL.Ghost.Gbx'),
                       motors=str(winner / 'motors.json'), autonomous_model_pending=True,
                       visual_ghost_pending=True)
        save(winner / 'receipt.json', binding)
        save(root / 'best.json', binding)
        with (root / 'improvements.jsonl').open('a') as log:
            log.write(json.dumps(binding) + '\n')

    try:
        # Five fresh engine resets of the incumbent plus a known different-time control.
        jimmy = list(baseline)
        jimmy[1718:1720] = [39, 39]
        jobs = [(f'baseline{i}', baseline) for i in range(5)]
        jobs += [('known17715', tuple(jimmy)), ('neutralDNF', (36,) * 1775)]
        seeded=None
        if 'qualified_seed' in plan:
            seeded=tuple(json.loads((root/'qualified-motors.json').read_text()))
            validate(seeded)
            if digest(seeded)!=plan['qualified_seed']['motor_sha256']:raise RuntimeError('Qualified seed hash')
            jobs += [(f'seed{i}',seeded) for i in range(5)]
        values, initial_receipt = execute(jobs)
        if [values[f'baseline{i}'] for i in range(5)] != [17712] * 5 or values['known17715'] != 17715 or values['neutralDNF'] is not None:
            raise RuntimeError('Native independently-computed Goal qualification failed')
        if seeded is not None:
            goal=plan['qualified_seed']['actual_native_goal_ms']
            if [values[f'seed{i}'] for i in range(5)]!=[goal]*5:raise RuntimeError('Seed fresh confirmation failed')
            retain(seeded,goal,initial_receipt)
            elite=[(goal,seeded),(17712,baseline)]
        ready = clock()
        utc = dt.datetime.now(dt.timezone.utc)
        budget = 600 if args.qualification else int(plan['total_budget_seconds'])
        reserve = 0 if args.qualification else int(plan['finalization_reserve_seconds'])
        search_seconds = budget - reserve
        deadline = ready + search_seconds
        state.update(phase='searching', ready_utc=utc.isoformat(), ready_clock=ready,
                     search_deadline_clock=deadline, total_deadline_clock=ready+budget,
                     search_deadline_utc=(utc+dt.timedelta(seconds=search_seconds)).isoformat(),
                     total_deadline_utc=(utc+dt.timedelta(seconds=budget)).isoformat(),
                     total_budget_seconds=budget, finalization_reserve_seconds=reserve)
        heartbeat()
        save(root / 'ready.json', {k: state[k] for k in (
            'parent','boot','ready_utc','ready_clock','search_deadline_utc','total_deadline_utc',
            'search_deadline_clock','total_deadline_clock','total_budget_seconds','finalization_reserve_seconds')})
        edge_source = digest(best)
        edge_iter = iter(neighbourhood(best, rng))
        while clock() < deadline - 245 and not (root / 'STOP').exists():
            jobs = [('incumbent', best)]
            by_name: dict[str, tuple[int, ...]] = {}
            scale = 2 if batch-last_improvement_batch > 12 else 1
            for attempt in range(10000):
                if len(jobs) >= int(plan['batch_size']) + 1:
                    break
                if digest(best) != edge_source:
                    edge_source = digest(best)
                    edge_iter = iter(neighbourhood(best, rng, scale))
                draw = rng.random()
                if draw < .50:
                    try:
                        _, candidate = next(edge_iter)
                    except StopIteration:
                        edge_iter = iter(neighbourhood(best, rng, scale))
                        _, candidate = next(edge_iter)
                elif draw < .88:
                    parent = best if rng.random() < .7 else rng.choice(elite)[1]
                    candidate = mutate(parent, rng, rng.choice([1, 1, 2, 2, 3]), scale)
                else:
                    other = rng.choice(elite)[1]
                    first = rng.randrange(0, 1720)
                    last = min(1771, first+rng.choice([5,10,20,40,80,160]))
                    candidate = best[:first] + other[first:last] + best[last:]
                key = digest(candidate)
                if key in seen:
                    continue
                validate(candidate)
                seen.add(key)
                name = f'candidate{len(by_name):04d}'
                jobs.append((name, candidate))
                by_name[name] = candidate
            if not by_name:
                raise RuntimeError('No new candidate; do not idle on repeated plans')
            values, _ = execute(jobs)
            if values['incumbent'] != best_time:
                raise RuntimeError('Qualified incumbent changed under identical native input')
            state['candidate_evaluations'] += len(by_name)
            candidates = [(goal, by_name[name]) for name, goal in values.items()
                          if name in by_name and goal is not None]
            elite.extend((goal,ticks) for goal,ticks in candidates if goal <= best_time+500)
            unique = {digest(ticks):(goal,ticks) for goal,ticks in elite}
            elite = sorted(unique.values(), key=lambda item:item[0])[:16]
            if candidates:
                goal, candidate = min(candidates, key=lambda item:item[0])
                if goal < best_time:
                    confirm, proof = execute([(f'confirm{i}', candidate) for i in range(5)])
                    if list(confirm.values()) != [goal]*5:
                        raise RuntimeError('Candidate repeatability failure; preserve all results')
                    retain(candidate, goal, proof)
            state['distinct_inputs_considered'] = len(seen)
            state['elite_size'] = len(elite)
            heartbeat()
            if args.qualification:
                break
        if (root / 'STOP').exists():
            state['stop_reason'] = 'user_stop'
        else:
            state['stop_reason'] = 'qualification_complete' if args.qualification else 'search_budget_and_finalization_reserve'
        if not args.qualification and best_time < 17712 and not (root/'STOP').exists():
            state['phase'] = 'finalizing_model'
            heartbeat()
            # The helper has its own native/BC/evaluation bounds and exact cleanup.
            command = [sys.executable, '-B', str(root/'source/finalize.py'), str(root)]
            with (root/'finalizer.stdout').open('xb') as out, (root/'finalizer.stderr').open('xb') as err:
                child = subprocess.Popen(command, cwd=root/'source', stdout=out, stderr=err,
                                         start_new_session=True)
                state['finalizer_identity'] = identity(child.pid)
                heartbeat()
                pidfd=os.pidfd_open(child.pid)
                try:
                    while child.poll() is None:
                        if clock() >= state['total_deadline_clock'] or (root/'STOP').exists():
                            raise RuntimeError('Finalization total deadline/STOP')
                        state['finalization_seconds_remaining'] = state['total_deadline_clock']-clock()
                        heartbeat()
                        time.sleep(1)
                finally:
                    if child.poll() is None:
                        (root/'model-finalization/STOP').touch(exist_ok=True)
                        try:child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            signal.pidfd_send_signal(pidfd,signal.SIGTERM)
                            try:child.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                state['forced'].append(state['finalizer_identity'])
                                signal.pidfd_send_signal(pidfd,signal.SIGKILL)
                                child.wait(timeout=5)
                    os.close(pidfd)
                    state['finalizer_exit_code']=child.returncode
                    state['remainingOwned']=descendants(os.getpid())
                    heartbeat()
            if child.returncode != 0:
                raise RuntimeError('Finalizer failed; preserve best line/model independently')
            result = json.loads((root/'finalization.json').read_text())
            state['best_autonomous_goal_ms'] = result['best_autonomous_goal_ms']
            state['visual_ghost_pending']=result['visual_ghost_pending']
        state['phase'] = 'qualification_complete' if args.qualification else 'completed'
        heartbeat()
        save(root/'completion.json',state)
    except BaseException as error:
        state.update(phase='failed', error=repr(error), best_goal_ms=best_time)
        heartbeat()
        save(root/'completion.json',state)
        raise


if __name__ == '__main__':
    main()
