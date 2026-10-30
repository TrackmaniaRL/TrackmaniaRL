"""Fresh, bounded RL experiment. Never resumes or modifies an older campaign.

The second map is admitted only with an independently qualified step environment;
a replay validator or a partially reconstructed mesh is not such an environment.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import subprocess

BASE = Path('/home/kuba/tmrl-full-qr-20261022-011')
OUT = Path(sys.argv[1]).resolve()
PRIVATE = BASE / OUT.name
SOURCE = PRIVATE / 'source'
sys.path.insert(0, str(SOURCE))
sys.dont_write_bytecode = True
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('WANDB_MODE', 'offline')
from contracts import clean_dependency, meets_goal


def stopped():
    role = sys.argv[2] if len(sys.argv) > 2 else 'parent'
    return (OUT / 'STOP').exists() or (OUT / ('stop-' + role)).exists()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.stage')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')
    os.replace(tmp, path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reserves():
    for path, gib in [('/mnt/c', 10), ('/mnt/f', 6), ('/mnt/j', 10), (BASE, 6)]:
        if shutil.disk_usage(path).free < gib * 2**30:
            raise RuntimeError(f'Storage reserve: {path}, {gib} GiB')
    size = 0
    for root in (OUT, PRIVATE):
        for p in root.rglob('*'):
            try:
                if p.is_file():
                    size += p.stat().st_size
            except FileNotFoundError:
                pass
    if size >= 12 * 2**30:
        raise RuntimeError('Own experiment quota 12 GiB')


def verify_sources():
    manifest = read(OUT / 'source-manifest.json')
    for item in manifest:
        if sha(PRIVATE / item['path']) != item['sha256']:
            raise RuntimeError(f'Frozen source changed: {item["path"]}')
    for item in read(OUT / 'input-manifest.json'):
        if sha(item['path']) != item['sha256']:
            raise RuntimeError(f'Pinned input changed: {item["path"]}')


def env_config(role):
    cfg = read(OUT / 'environment.json')
    cfg.pop('training_demonstration', None)
    cfg['runtime'].update(trace_dir=str(PRIVATE / role / 'traces'),
        work_dir=str(BASE / 'waypoint-runtime002/build/server-probe' / OUT.name / role),
        timeout_s=150)
    cfg.update(trace_archive_root=str(OUT / role / 'closed-traces'),
               keep_trace_archives=2, driver_episode_limit=1)
    return cfg


def wait_for_dependency():
    from trackmaniarl.trackmania.native_headless._runtime.ownership import process_identity, same_identity
    dependency = read(OUT / 'plan.json')['dependency']
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if boot != dependency['boot']:
        raise RuntimeError('Boot changed since queue admission; require a fresh inspection')
    started = time.monotonic()
    while time.monotonic() - started < 6 * 3600:
        if (OUT / 'STOP').exists():
            raise RuntimeError('Own pipeline STOP')
        reserves()
        alive = [p for p in dependency['identities'] if same_identity(p, process_identity(p['pid']))]
        result = Path(dependency['directory']) / 'parent-result.json'
        save(OUT / 'state.json', dict(phase='WAITING_FOR_EXISTING_QUALIFICATION',
            parent=process_identity(os.getpid()), boot=boot, dependency_alive=alive,
            monotonic=time.monotonic(), training_started=False))
        if not alive:
            if not result.is_file():
                raise RuntimeError('Dependency disappeared without its closure receipt')
            row = read(result)
            if not clean_dependency(row):
                raise RuntimeError('Existing qualification did not complete cleanly; no automatic competing launch')
            windows = read(OUT / 'windows-dependency.json')
            checks = []
            for identity in windows:
                pid, ticks = int(identity['pid']), int(identity['birthUtcTicks'])
                command = f'$p=Get-Process -Id {pid} -ErrorAction SilentlyContinue; if ($p -and $p.StartTime.ToUniversalTime().Ticks -eq {ticks}) {{exit 9}}; exit 0'
                result_check = subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe', '-NoProfile', '-Command', command], capture_output=True, timeout=20)
                checks.append(dict(**identity, returncode=result_check.returncode))
                if result_check.returncode != 0:
                    raise RuntimeError('Exact Windows dependency owner still alive or inspection failed')
            save(OUT / 'windows-dependency-closure.json', checks)
            # Also refuse an unrelated active learner which appeared while waiting.
            foreign = []
            for proc in Path('/proc').iterdir():
                if not proc.name.isdigit() or int(proc.name) == os.getpid():
                    continue
                try:
                    cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue
                if 'campaign_entry' in cmd and '--role learner' in cmd:
                    foreign.append(int(proc.name))
            if foreign:
                raise RuntimeError(f'Another learner is active: {foreign}')
            save(OUT / 'dependency-closure.json', row)
            return
        time.sleep(15)
    raise RuntimeError('Bounded six-hour dependency wait expired')


def collect():
    import numpy as np
    import torch
    from dataclasses import asdict
    from trackmaniarl.experiments.shortcut_macro import AutonomousMacroShortcutFactory, actual_goal_ms
    from trackmaniarl.trackmania.native_headless._runtime.ownership import process_identity, same_identity
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    cfg = env_config('collection')
    factory = AutonomousMacroShortcutFactory(cfg)
    plan = read(OUT / 'plan.json')
    with np.load(OUT / 'macro-catalog.npz', allow_pickle=False) as z:
        programs = [tuple(map(int, p)) for p in z['programs']]
    lookup = {p: i for i, p in enumerate(programs)}
    episodes = []
    for index, tape in enumerate(plan['training_tapes']):
        ticks = read(tape['path'])
        if hashlib.sha256(bytes(ticks)).hexdigest() != tape['motor_sha256']:
            raise RuntimeError('Training input tape hash mismatch')
        actions = [lookup[tuple(ticks[i:i+5])] for i in range(0, len(ticks), 5)]
        env = factory.create(seed=index)
        identity = None
        rows, rewards, infos, chosen, times = [], [], [], [], [0]
        started = time.monotonic()
        report = dict(status='FAILED', target_ms=tape['goal_ms'])
        try:
            observation, _ = env.reset(seed=index)
            identity = process_identity(env.provider._session.pid)
            rows.append({k: v.detach().cpu().numpy().copy() for k, v in observation.items()})
            for action in actions:
                if time.monotonic() - started > 180 or stopped():
                    raise RuntimeError('Collection deadline or STOP')
                observation, reward, terminated, truncated, info = env.step(int(action))
                rows.append({k: v.detach().cpu().numpy().copy() for k, v in observation.items()})
                chosen.append(action)
                rewards.append(reward)
                infos.append(info)
                times.append(int(info['observed_physics_time_ms']))
                save(OUT / 'collection/progress.json', dict(lap=index, decisions=len(chosen), monotonic=time.monotonic()))
                if terminated or truncated:
                    goal = actual_goal_ms(env) if info['termination_reason'] == 'finished' else None
                    if goal != tape['goal_ms']:
                        raise RuntimeError(f'Collection actual Goal {goal}, expected {tape["goal_ms"]}')
                    break
            else:
                raise RuntimeError('Training tape did not reach an actual Goal')
            dataset = OUT / 'collection' / f'native-demonstration-{index}.npz'
            arrays = dict(schema=np.asarray('tmrl.native-macro-demonstration/v1'),
                map_uid=np.asarray(factory._map_uid),
                map_sha256=np.asarray(plan['map_sha256']),
                reward_sha256=np.asarray(hashlib.sha256(json.dumps(asdict(factory._reward), sort_keys=True, allow_nan=False).encode()).hexdigest()),
                reward_clock=np.asarray('native_goal_event/v1'), catalog_sha256=np.asarray(sha(OUT / 'macro-catalog.npz')),
                actions=np.asarray(chosen, dtype=np.int64), rewards=np.asarray(rewards, dtype=np.float32),
                terminated=np.asarray([False] * (len(chosen)-1) + [True]),
                observed_physics_ms=np.asarray(times, dtype=np.int64),
                observation_keys=np.asarray(json.dumps(sorted(rows[0]))), infos_json=np.asarray(json.dumps(infos)))
            arrays.update({'observation/' + k: np.stack([r[k] for r in rows]) for k in rows[0]})
            np.savez_compressed(dataset, **arrays)
            episodes.append(dict(path=str(dataset), sha256=sha(dataset), native_goal_ms=goal))
            report.update(status='PASS_REAL_NATIVE_DEMONSTRATION', decisions=len(chosen), actual_goal_ms=goal, dataset=episodes[-1])
        finally:
            env.close()
            report['remaining_exact_native'] = [identity] if identity and same_identity(identity, process_identity(identity['pid'])) else []
            report['archives'] = env.provider.archive_receipts
            save(OUT / 'collection' / f'result-{index}.json', report)
        if report['remaining_exact_native']:
            raise RuntimeError('Collection native did not close')
    save(OUT / 'collection/native-demonstration-manifest.json', dict(schema='tmrl.native-multi-demonstration/v1', episodes=episodes))


def build_model(policy_path):
    import torch
    import yaml
    from trackmaniarl.core.builtins import TorchCheckpointCodec
    from trackmaniarl.models.factory import CompositeValueModelFactory
    raw = yaml.safe_load((OUT / 'model-spec.yaml').read_text())
    model = CompositeValueModelFactory(**raw['components']['model_factory']['kwargs']).build()
    policy = TorchCheckpointCodec().load(policy_path)
    online = policy.get('online') or policy['learner']['online']
    for name, values in online.items():
        getattr(model, name).load_state_dict(values, strict=True)
    return model, policy


def bc():
    import numpy as np
    import torch
    from trackmaniarl.core.builtins import TorchCheckpointCodec
    from trackmaniarl.experiments.shortcut_pretraining import BehaviorCloningPretrainer
    torch.set_num_threads(2)
    torch.manual_seed(1160)
    model, original = build_model(OUT / 'baseline-policy.pt')
    model.to('cuda')
    fastest = min(read(OUT / 'collection/native-demonstration-manifest.json')['episodes'], key=lambda e: e['native_goal_ms'])
    with np.load(fastest['path'], allow_pickle=False) as z:
        names = json.loads(str(z['observation_keys'].item()))
        obs = {k: torch.from_numpy(z['observation/'+k][:-1].copy()).to('cuda') for k in names}
        actions = torch.from_numpy(z['actions'].copy()).long().to('cuda')
    trainer = BehaviorCloningPretrainer(model, actions, learning_rate=3e-4)
    started = time.monotonic()
    history, best, consecutive = [], None, 0
    for update in range(1, 10001):
        if stopped():
            raise RuntimeError('Own BC STOP')
        if time.monotonic()-started > 1200:
            break
        metrics = trainer.update(obs, actions)
        if not all(math.isfinite(x) for x in metrics.values()):
            raise RuntimeError('Nonfinite BC metrics')
        if update % 25:
            continue
        model.eval()
        with torch.no_grad():
            logits = trainer.logits(obs)
            margin = float((logits.gather(1, actions[:, None]).squeeze(1) - logits.scatter(1, actions[:, None], -torch.inf).max(1).values).min())
            mistakes = int((logits.argmax(1) != actions).sum())
        row = dict(updates=update, mistakes=mistakes, margin=margin, **metrics)
        history.append(row)
        save(OUT / 'bc/fit-history.json', history)
        if mistakes == 0 and (best is None or margin > best[0]):
            online = {n: {k: v.detach().cpu().clone() for k, v in getattr(model, n).state_dict().items()} for n in original['online']}
            best = (margin, update, online)
        consecutive = consecutive+1 if mistakes == 0 and margin >= .5 else 0
        if consecutive >= 3:
            break
    result = dict(status='FIT_GATE_NOT_PASSED_USE_QUALIFIED_BASELINE', updates=update,
        online_RL_updates=0, wall_seconds=time.monotonic()-started)
    if best is not None and best[0] >= .5:
        policy = dict(original, online=best[2], training_updates=best[1])
        TorchCheckpointCodec().save(policy, OUT / 'bc/policy-only.pt')
        result.update(status='FIT_GATE_PASSED_NATIVE_EVALUATION_PENDING', minimum_margin=best[0], selected_update=best[1], policy_sha256=sha(OUT / 'bc/policy-only.pt'))
    save(OUT / 'bc/result.json', result)


def evaluate(policy_path, destination, trials, maximum_ms, *, require_all=False):
    import torch
    from trackmaniarl.algorithms.value_based.policy import DiscreteValuePolicy, DiscreteValuePolicyConfig
    from trackmaniarl.core.contracts import PolicyMode
    from trackmaniarl.models.contracts import RiskSpec
    from trackmaniarl.experiments.shortcut_macro import AutonomousMacroShortcutFactory, actual_goal_ms
    from trackmaniarl.trackmania.native_headless._runtime.ownership import process_identity, same_identity
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    model, _ = build_model(policy_path)
    model.eval()
    policy = DiscreteValuePolicy(model, torch.device('cpu'), DiscreteValuePolicyConfig(0, None, RiskSpec(), RiskSpec()))
    initial = {k: v.clone() for k, v in policy.export_state().items()}
    checksum = sha(policy_path)
    blocked = []
    def deny(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)) and any(w in str(args[0]).lower() for w in ('training-only', 'native-demonstration', '/winners/', 'motors.json')):
            blocked.append(str(args[0]))
            raise RuntimeError('Policy evaluation attempted teacher access')
    sys.addaudithook(deny)
    made, identities, rows = [], [], []
    report = dict(status='FAILED', policy_sha256=checksum, trials=rows, learner_updates_during_evaluation=0)
    try:
        for trial in range(trials):
            if stopped():
                raise RuntimeError('Own evaluation STOP')
            cfg = env_config(destination)
            # Preserve every attempted episode in this bounded 100-trial suite.
            cfg['keep_trace_archives'] = trials
            env = AutonomousMacroShortcutFactory(cfg).create(seed=trial)
            made.append(env)
            policy.reset_episode()
            started = time.monotonic()
            commands = []
            try:
                observation, _ = env.reset(seed=trial)
                identities.append(process_identity(env.provider._session.pid))
                for step in range(3000):
                    if time.monotonic()-started > 180 or stopped():
                        raise RuntimeError('Evaluation episode deadline/STOP')
                    action = policy.act(observation, PolicyMode.EVALUATION)
                    observation, _, terminated, truncated, info = env.step(action)
                    commands.append(action)
                    if terminated or truncated:
                        goal = actual_goal_ms(env) if info['termination_reason'] == 'finished' else None
                        rows.append(dict(trial=trial, actual_goal_ms=goal, reason=info['termination_reason'], decisions=len(commands)))
                        break
                else:
                    raise RuntimeError('Evaluation exceeded physical step budget')
            finally:
                env.close()
            save(OUT / destination / f'actions-{trial:03d}.json', commands)
            save(OUT / destination / 'progress.json', dict(trials=rows, monotonic=time.monotonic()))
        successful = [r for r in rows if meets_goal([r], maximum_ms)]
        remaining = [p for p in identities if same_identity(p, process_identity(p['pid']))]
        if blocked or remaining or sha(policy_path) != checksum or any(not torch.equal(initial[k], v) for k, v in policy.export_state().items()):
            raise RuntimeError('Evaluation independence, weight immutability or closure failed')
        passed = len(successful) == trials if require_all else bool(successful)
        report.update(status='PASS_FROZEN_POLICY_GOAL_TARGET' if passed else 'TARGET_NOT_REACHED',
            passed=passed, successful_trials=len(successful), total_trials=trials,
            acceptance_maximum_ms=maximum_ms, remaining_exact_native=remaining, teacher_access_attempts=blocked,
            weights_unchanged=True, greedy=True)
    finally:
        for env in made:
            env.close()
        save(OUT / destination / 'result.json', report)


def rl_worker(role):
    from trackmaniarl.core.spec import RunSpec
    from trackmaniarl.core.runtime import resolve_run
    from trackmaniarl.core.builtins import TorchCheckpointCodec
    from trackmaniarl.distributed.coordinator import Coordinator
    from trackmaniarl.distributed.coordinator_types import CoordinatorConfig
    from trackmaniarl.distributed.protocol import run_fingerprint
    from trackmaniarl.distributed.actor_requests import ActorProcessRequest
    from trackmaniarl.experiments.shortcut_rl import NativeGoalActor
    from trackmaniarl.experiments.full_qr.campaign_budget import CampaignBudget, BudgetSettings
    from trackmaniarl.experiments.full_qr.campaign_policy import StorageReserve
    from trackmaniarl.experiments.full_qr.campaign_signal import MappedStopSignal
    from trackmaniarl.experiments.full_qr.campaign_worker import WorkerProgress
    from trackmaniarl.experiments.full_qr.campaign_coordinator import verify_final_checkpoint
    import torch
    spec = RunSpec.from_yaml(PRIVATE / 'run.yaml')
    if sha(PRIVATE / 'run.yaml') != read(OUT / 'rl-prepared.json')['config_sha256']:
        raise RuntimeError('Frozen training configuration changed')
    signal = MappedStopSignal(PRIVATE / 'rl-stop')
    run = progress = budget = None
    report = dict(status='FAILED', role=role)
    try:
        if role != 'learner':
            def deny(event, args):
                if event == 'open' and isinstance(args[0], (str, bytes)) and any(w in str(args[0]).lower() for w in ('native-demonstration', 'training-only', 'motors.json', '/winners/')):
                    raise RuntimeError('Actor attempted teacher access')
            sys.addaudithook(deny)
            ready = read(PRIVATE / 'learner-ready.json')
            actor = NativeGoalActor(ActorProcessRequest(str(PRIVATE / 'run.yaml'), f'127.0.0.1:{ready["port"]}', role, (PRIVATE / 'token').read_text(), signal))
            actor.run_forever()
            report.update(status='PASS_ACTOR_EXIT', stop_reason=actor.stop_reason)
            return
        run = resolve_run(spec, base_dir=PRIVATE)
        budget = CampaignBudget(BudgetSettings(PRIVATE / 'budget', tuple(StorageReserve(Path(p), n * 2**30) for p, n in [('/mnt/c',10),('/mnt/f',6),('/mnt/j',10),(str(BASE),6)]), duration_seconds=14400, clock_source='monotonic_raw'), stop_signal=signal)
        class RecordCoordinator(Coordinator):
            def _start_server(self):
                super()._start_server()
                expected = TorchCheckpointCodec().load(OUT / 'rl-initialization.pt')['learner']['online']
                if self.run.learner.update_count != 0 or self.run.learner.optimizer.state:
                    raise RuntimeError('Fresh optimizer and zero RL updates required')
                for name, state in expected.items():
                    for key, value in state.items():
                        if not torch.equal(getattr(self.run.learner.model, name).state_dict()[key].cpu(), value) or not torch.equal(getattr(self.run.learner.target_model, name).state_dict()[key].cpu(), value):
                            raise RuntimeError('Warm initialization online/target mismatch')
                save(OUT / 'initialization-proof.json', dict(status='PASS', new_optimizer=True, initial_RL_updates=0, warm_sha256=sha(OUT / 'rl-initialization.pt')))
                armed = budget.start()
                save(PRIVATE / 'learner-ready.json', dict(pid=os.getpid(), port=self.bound_port, **armed))
            def _external_stop_requested(self):
                if signal.is_set() and budget.state.get('stop_reason') is None:
                    budget.request_stop('pipeline_stop')
                return budget.stop_requested() or super()._external_stop_requested()
            def _finish_evaluation_batch(self, summaries):
                super()._finish_evaluation_batch(summaries)
                history = read(OUT / 'quality-history.json') if (OUT / 'quality-history.json').exists() else []
                row = dict(updates=self.counters.updates, policy_version=summaries[0]['policy_version'], trials=summaries,
                    finished=sum(bool(s['finished']) for s in summaries))
                history.append(row)
                save(OUT / 'quality-history.json', history)
                if self.counters.updates >= 500 and len(history) >= 3 and all(r['finished'] == 0 for r in history[-3:]):
                    budget.request_stop('quality_regression')
                if any(s['finished'] and type(s.get('actual_goal_race_time_ms')) is int and s['actual_goal_race_time_ms'] <= 17600 for s in summaries):
                    self._checkpoint_writer.wait()
                    files = list(self.run.checkpoint_dir.glob(f'*-eval-policy-{row["policy_version"]:08d}-at-update-*.pt'))
                    # Actual snapshot name is additionally matched by coordinator's evidence below.
                    save(OUT / 'record-candidate.json', dict(**row, matching_files=[str(p) for p in files], requires_independent_100_trial_suite=True))
                    budget.request_stop('record_candidate')
        binding = spec.components.environment.kwargs['config']['training_demonstration']
        coordinator = RecordCoordinator(run, CoordinatorConfig('127.0.0.1:0', (PRIVATE / 'token').read_text(), run_fingerprint(spec, PRIVATE), external_stop=signal, demo_paths=(Path(binding['path']),)))
        progress = WorkerProgress(coordinator, PRIVATE / 'progress.json', signal)
        progress.start()
        result = coordinator.run_forever()
        proof = verify_final_checkpoint(run, result, coordinator.fingerprint)
        if result.updates <= 0 or result.transitions <= 0:
            raise RuntimeError('No actual online RL training occurred')
        report.update(status='PASS_RL_WORKER_CLOSED', result=dataclasses.asdict(result), checkpoint=proof)
    finally:
        if progress is not None:
            progress.close()
        if budget is not None and budget.state:
            report['budget'] = budget.close()
        signal.set()
        signal.close()
        if run is not None:
            run.logger.close()
        save(OUT / 'rl' / (role + '-result.json'), report)


def prepare_rl(policy_path):
    import secrets
    import torch
    import yaml
    from trackmaniarl.core.builtins import TorchCheckpointCodec
    from trackmaniarl.core.spec import RunSpec
    from trackmaniarl.experiments.shortcut_pretraining import balanced_action_weights
    from trackmaniarl.experiments.shortcut_multi import MultiDemonstrationShortcutFactory
    from trackmaniarl.experiments.full_qr.campaign_signal import MappedStopSignal
    baseline = TorchCheckpointCodec().load(policy_path)
    TorchCheckpointCodec().save(dict(schema_version='2.0', learner=dict(schema_version='2.0', architecture_fingerprint=baseline['architecture_fingerprint'], online=baseline['online'])), OUT / 'rl-initialization.pt')
    raw = yaml.safe_load((OUT / 'model-spec.yaml').read_text())
    raw.update(run_id=OUT.name, artifacts_dir=str(OUT / 'artifacts'), checkpoint_root=None)
    cfg = env_config('rl')
    cfg['training_demonstration'] = dict(path=str(OUT / 'collection/native-demonstration-manifest.json'), sha256=sha(OUT / 'collection/native-demonstration-manifest.json'))
    raw['components']['environment'] = dict(class_path='trackmaniarl.experiments.shortcut_multi:MultiDemonstrationShortcutFactory', kwargs=dict(config=cfg))
    factory = MultiDemonstrationShortcutFactory(cfg)
    demos = factory.load_demonstration(Path(cfg['training_demonstration']['path']), None)
    weights = balanced_action_weights(torch.tensor([t.action for t in demos], dtype=torch.long), len(factory.catalog.programs)).tolist()
    kwargs = raw['components']['learner']['kwargs']
    kwargs.update(model_initialization_checkpoint=str(OUT / 'rl-initialization.pt'), learning_rate=1e-5,
        warm_start_required_tensors=[f'{name}.{key}' for name, state in baseline['online'].items() for key in state],
        objectives=[dict(class_path='trackmaniarl.experiments.shortcut_multi:ReleasingExpertObjective', kwargs=dict(class_weights=weights, config=dict(initial_weight=128, final_weight=8, hold_transitions=10000, release_transitions=150000)))])
    raw['components']['sampler'] = dict(class_path='trackmaniarl.experiments.shortcut_multi:PaceAwareMixedSampler', kwargs=dict(seed=1160, config=dict(expert_lap_time_s=20, best_demo_time_s=17.601, online_demo_fraction=.5, online_demo_fraction_final=.25, expert_fraction_anneal_transitions=150000)))
    raw['components']['replay_store']['kwargs']['capacity'] = 100000
    raw['training'].update(warmup_transitions=2000, offline_pretrain_updates=0, total_transitions=2**60,
        checkpoint_interval_updates=2000, checkpoint_keep_last=2, max_episode_artifacts=20,
        evaluation_stop_min_finish_rate=None, evaluation_stop_median_s=None, evaluation_stop_consecutive_batches=None)
    raw['distributed'].update(epsilon_profiles=[.5, 1.0], epsilon_start=.02, epsilon_final=.002, epsilon_decay_transitions=150000)
    raw['evaluation']['trials_per_map'] = 3
    raw['metadata'].update(initialization_only=True, resume_checkpoint=None, requested_wall_time_seconds=14400,
        strict_success_goal_ms_at_most=17600, record_to_beat_ms=17601, initial_policy_sha256=sha(policy_path),
        authorization='Jakub 2026-10-29: prepare and launch new actual RL record pipeline; one success in 100 acceptable')
    spec = RunSpec.model_validate(raw)
    (PRIVATE / 'run.yaml').write_text(spec.to_yaml())
    (OUT / 'run.yaml').write_text(spec.to_yaml())
    (PRIVATE / 'token').write_text(secrets.token_hex(32))
    os.chmod(PRIVATE / 'token', 0o600)
    signal = MappedStopSignal(PRIVATE / 'rl-stop', create=True)
    signal.close()
    save(OUT / 'rl-prepared.json', dict(config_sha256=sha(PRIVATE / 'run.yaml'), demonstrations=len(demos), new_optimizer=True, policy_sha256=sha(policy_path), duration_seconds=14400, actors=2))


def phase(role, limit, *args):
    from trackmaniarl.experiments.full_qr.campaign_processes import CampaignProcessCohort
    cohort = CampaignProcessCohort(PRIVATE / ('phase-' + role), lambda: (OUT / ('stop-' + role)).touch(exist_ok=True))
    report = dict(role=role, status='RUNNING')
    started, last_storage = time.monotonic(), 0.0
    try:
        report['identities'] = cohort.start(role, [sys.executable, '-B', str(PRIVATE / 'pipeline.py'), str(OUT), role, *map(str, args)], cwd=SOURCE)
        save(OUT / 'state.json', dict(phase=role, **report, monotonic=time.monotonic()))
        while not cohort.exited():
            now = time.monotonic()
            if now-started > limit or (OUT / 'STOP').exists():
                raise RuntimeError(f'Phase {role} deadline or STOP')
            if now-last_storage > 20:
                reserves()
                last_storage = now
            time.sleep(.5)
        report['exits'] = cohort.exited()
        if report['exits'] != {role: 0}:
            raise RuntimeError(f'Phase failed: {report["exits"]}')
        report['status'] = 'PASS_ROLE_CLOSED'
    except BaseException as error:
        report.update(status='FAILED', error=repr(error))
        raise
    finally:
        report['closure'] = cohort.close(grace_seconds=30)
        save(OUT / (role + '-supervision.json'), report)
    if any(report['closure'][k] for k in ('remainingOwned', 'cleanupErrors', 'forced')):
        raise RuntimeError(f'Phase {role} did not close cleanly')


def rl():
    from trackmaniarl.experiments.full_qr.campaign_processes import CampaignProcessCohort
    from trackmaniarl.experiments.full_qr.campaign_signal import MappedStopSignal
    signal = MappedStopSignal(PRIVATE / 'rl-stop')
    cohort = CampaignProcessCohort(PRIVATE / 'rl-processes', signal.set)
    report = dict(status='FAILED', children=[])
    started, changed, last_updates = time.monotonic(), time.monotonic(), None
    try:
        report['children'].append(cohort.start('learner', [sys.executable, '-B', str(PRIVATE / 'pipeline.py'), str(OUT), 'learner'], cwd=SOURCE))
        while not (PRIVATE / 'learner-ready.json').exists():
            if cohort.exited() or time.monotonic()-started > 240:
                raise RuntimeError('RL learner readiness failed')
            time.sleep(.5)
        for index in range(2):
            role = f'actor-{index:03d}'
            report['children'].append(cohort.start(role, [sys.executable, '-B', str(PRIVATE / 'pipeline.py'), str(OUT), role], cwd=SOURCE))
        ready = read(PRIVATE / 'learner-ready.json')
        save(OUT / 'rl/launch.json', dict(**report, ready=ready))
        while 'learner' not in cohort.exited():
            now = time.monotonic()
            if now-started > 14940 or (OUT / 'STOP').exists():
                signal.set()
                raise RuntimeError('RL outer deadline or STOP')
            if cohort.exited() and not signal.is_set():
                raise RuntimeError('Actor exited before stop')
            reserves()
            if (PRIVATE / 'progress.json').exists():
                progress = read(PRIVATE / 'progress.json')
                if progress['updates'] != last_updates:
                    changed, last_updates = now, progress['updates']
                if not signal.is_set() and (now-progress['monotonic'] > 30 or now-changed > 600):
                    raise RuntimeError('Real RL heartbeat or progress stalled')
                save(OUT / 'state.json', dict(phase='ACTUAL_ONLINE_RL', ready=ready, progress=progress, children=report['children'], monotonic=now))
            time.sleep(10)
        worker = read(OUT / 'rl/learner-result.json')
        if worker['status'] != 'PASS_RL_WORKER_CLOSED':
            raise RuntimeError('RL learner failed')
        report.update(status='PASS_RL_CLOSED', learner=worker)
    finally:
        report['closure'] = cohort.close(grace_seconds=120)
        signal.close()
        save(OUT / 'rl/completion.json', report)
    if any(report['closure'][k] for k in ('remainingOwned', 'cleanupErrors', 'forced')) or any(c['returncode'] != 0 for c in report['closure']['children'].values()):
        raise RuntimeError('RL process closure failed')


def main():
    from trackmaniarl.trackmania.native_headless._runtime.ownership import process_identity
    with (OUT / 'launch-marker').open('x') as stream:
        stream.write(str(os.getpid()))
    report = dict(status='FAILED', parent=process_identity(os.getpid()), started_unix=time.time())
    try:
        verify_sources()
        if read(OUT / 'offline-audit.json')['status'] != 'PASS_OFFLINE_REAL_DATA_AUDIT':
            raise RuntimeError('Actual saved-observation CPU audit required before queue launch')
        wait_for_dependency()
        phase('collect', 450)
        phase('baseline-evaluate', 300, OUT / 'baseline-policy.pt', 'baseline-evaluation', 3, 17705, 'all')
        baseline = read(OUT / 'baseline-evaluation/result.json')
        if not baseline['passed']:
            raise RuntimeError('Qualified baseline did not reproduce after catalog remap')
        phase('bc', 1260)
        selected = OUT / 'baseline-policy.pt'
        if (OUT / 'bc/policy-only.pt').exists():
            phase('bc-evaluate', 300, OUT / 'bc/policy-only.pt', 'bc-evaluation', 3, 17601, 'all')
            if read(OUT / 'bc-evaluation/result.json')['passed']:
                selected = OUT / 'bc/policy-only.pt'
        prepare_rl(selected)
        verify_sources()
        rl()
        import torch
        from trackmaniarl.core.builtins import TorchCheckpointCodec
        codec = TorchCheckpointCodec()
        result = read(OUT / 'rl/learner-result.json')
        path = Path(result['checkpoint']['path'])
        checkpoint = codec.load(path)
        learner = checkpoint['learner']
        tensors = [v for state in learner['online'].values() for v in state.values()]
        if len(tensors) != 90 or not all(torch.isfinite(t).all() for t in tensors):
            raise RuntimeError('Final checkpoint tensor audit failed')
        # Prefer the exact evaluated record snapshot, when one was discovered.
        best_version = result['result']['updates']
        history = read(OUT / 'quality-history.json') if (OUT / 'quality-history.json').exists() else []
        finished = [(s['actual_goal_race_time_ms'], r['policy_version']) for r in history for s in r['trials'] if s['finished'] and type(s.get('actual_goal_race_time_ms')) is int and r['policy_version'] > 0]
        if finished:
            _, version = min(finished)
            matches = list((OUT / 'artifacts' / OUT.name / 'checkpoints').glob(f'fastest-eval-policy-{version:08d}-at-update-*.pt'))
            if not matches:
                matches = list((OUT / 'artifacts' / OUT.name / 'checkpoints').glob(f'best-eval-policy-{version:08d}-at-update-*.pt'))
            if not matches:
                raise RuntimeError('Cannot bind best result to its exact evaluated checkpoint')
            path = sorted(matches)[-1]
            learner = codec.load(path)['learner']
            best_version = version
        final = dict(schema='tmrl.autonomous-shortcut-policy-only/v1', architecture_fingerprint=learner['architecture_fingerprint'], online=learner['online'],
            online_RL_updates=best_version, total_run_RL_updates=result['result']['updates'], source_checkpoint_sha256=sha(path), macro_catalog_sha256=sha(OUT / 'macro-catalog.npz'))
        codec.save(final, OUT / 'retained-RL-policy.pt')
        phase('final-evaluate', 3600, OUT / 'retained-RL-policy.pt', 'final-100-evaluation', 100, 17600, 'any')
        final_eval = read(OUT / 'final-100-evaluation/result.json')
        report.update(status='RL_TARGET_REACHED_EXPORT_PENDING' if final_eval['passed'] else 'RL_COMPLETED_TARGET_NOT_REACHED',
            evaluation=final_eval, policy=str(OUT / 'retained-RL-policy.pt'), policy_sha256=sha(OUT / 'retained-RL-policy.pt'),
            fall05_status=read(OUT / 'fall05-admission.json')['status'])
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['ended_unix'] = time.time()
        save(OUT / 'completion.json', report)
        save(OUT / 'state.json', report)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        main()
    elif sys.argv[2] == 'collect':
        collect()
    elif sys.argv[2] == 'bc':
        bc()
    elif sys.argv[2].endswith('-evaluate'):
        evaluate(Path(sys.argv[3]), sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), require_all=sys.argv[7] == 'all')
    else:
        rl_worker(sys.argv[2])
