"""Full-control QR training; a NEW explicit UTC budget is mandatory."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time


def write(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('preparation',type=Path);parser.add_argument('qualification',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--until-utc',required=True)
    parser.add_argument('--seed',type=int,default=1168)
    parser.add_argument('--functional-test',action='store_true',help='hard cap180s,2training episodes,32updates,2evals')
    args=parser.parse_args()
    end=datetime.fromisoformat(args.until_utc.replace('Z','+00:00'))
    if end.tzinfo is None:raise ValueError('UTC offset required')
    budget=(end-datetime.now(timezone.utc)).total_seconds()
    if args.functional_test:
        if not 60<=budget<=180:raise ValueError('functional test requires60..180s')
    elif not 1800<=budget<=36000:raise ValueError('new explicit budget must be30min–10h; old budget is never extended')
    args.output.mkdir(exist_ok=False)
    started=time.monotonic();deadline=started+budget;training_end=deadline-(60 if args.functional_test else 1200)
    report=dict(status='preparing',public_WR_claim=False,deadline_utc=end.isoformat(),
                budget_seconds=budget,training_episodes=0,new_real_transitions=0,RL_updates=0,
                best_full_start_goal_ms=None,initialization_goal_ms=17601,evaluation=[],bounded_functional_test=args.functional_test,
                retention_contract=dict(optimizer_steps_per_update=1,learning_rate=2e-6,anchor_weight=2.,
                    anchor_margin=2.,anchor_rows='all1761',updates_per_transition=.05,max_updates_per_episode=64,
                    candidate_gate='greedy full-start clean NativeGoal<=accepted time before actor replacement',
                    candidate_check_every_episodes=2,initial_actor_RL_updates=0))
    learner=None;private=None
    try:
        proof=json.loads((args.qualification/'result.json').read_text())
        receipt=json.loads((args.preparation/'preparation.json').read_text())
        plan=json.loads((args.preparation/'plan.json').read_text())
        if proof['status']!='PASS_FULL_START_BC' or proof['cleanupErrors'] or len(proof['trials'])!=3:
            raise ValueError('three clean initializer trials required')
        for name,key in [('initialization.pt','checkpoint_sha256'),('route-geometry.npy','route_sha256'),
                         ('demonstrations-TRAINING-ONLY.npz','dataset_sha256')]:
            if sha(args.preparation/name)!=receipt[key]:raise ValueError('preparation changed: '+name)
        if proof['checkpoint_sha256']!=receipt['checkpoint_sha256']:raise ValueError('qualified checkpoint mismatch')
        pins=json.loads((args.preparation/'source-pins.json').read_text())
        for name in ('model.py','features.py'):
            if sha(Path(__file__).parent/name)!=pins[name]:raise ValueError('qualified model ABI changed')
        config=dict(plan['backend_config']);sys.path.insert(0,config['source'])
        # Recheck immutable native backend sources before importing it.
        from tools.record_rl1168.qualify import linux_path
        manifest=Path(linux_path(plan['backend_source_manifest']))
        if sha(manifest)!=plan['backend_source_manifest_sha256']:raise ValueError('native backend manifest changed')
        for item in json.loads(manifest.read_text()):
            if item['path'].startswith(config['source']+'/') and sha(item['path'])!=item['sha256']:
                raise ValueError('native backend source changed')
        from tools.record_rl1168.runtime import isolate_runtime
        from tools.record_rl1168.learner import Learner
        from tools.record_rl1168.episodes import episode
        from tools.record_rl1168.archive import archive_closed_trace
        from tools.record_rl1168.model import ACTIONS
        import numpy as np
        import torch
        torch.set_num_threads(2);torch.manual_seed(args.seed)
        private=Path('/home/kuba/tmrl-full-qr-20261022-011')/(args.output.parent.name+'-'+args.output.name)
        private.mkdir(mode=0o700,exist_ok=False);config=isolate_runtime(config,private)
        initial=torch.load(args.preparation/'initialization.pt',map_location='cpu',weights_only=True)
        learner=Learner(initial,device='cuda',seed=args.seed)
        learner.configure_demonstrations(args.preparation/'demonstrations-TRAINING-ONLY.npz')
        geometry=np.load(args.preparation/'route-geometry.npy',allow_pickle=False)
        with np.load(args.preparation/'demonstrations-TRAINING-ONLY.npz',allow_pickle=False) as data:
            prefix_actions=np.array([ACTIONS[int(action)] for action in data['actions']])
        rng=random.Random(args.seed);failures=0
        write(args.output/'provenance.json',dict(initialization=receipt,qualification=proof,
            plan=plan,seed=args.seed,teacher_prefixes_training_only=True,static_reference_geometry=True,
            code={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},native_config=config))
        while time.monotonic()<training_end and not(args.output/'STOP').exists():
            if args.functional_test and report['training_episodes']>=2:break
            if any(shutil.disk_usage(p).free<g*2**30 for p,g in [('/mnt/c',10),('/mnt/f',4),(str(private),4)]):
                raise RuntimeError('disk reserve')
            number=report['training_episodes'];folder=args.output/f'episode-{number:06d}'
            # Real teacher prefix advances physics; policy receives actual state at handoff.
            # Alternate full starts with suffixes; random actions perturb the latter.
            prefix=prefix_actions[:rng.randrange(200,1601)] if number%2 else ()
            epsilon=.01 if number%2 else (0. if number%4==0 else .003)
            learner.begin_episode()
            actor_state={k:v.detach().cpu().clone() for k,v in learner.actor.state_dict().items()}
            handoff_steps=0
            def training_action(state):
                nonlocal handoff_steps
                action=learner.action(state,epsilon)
                if len(prefix) and handoff_steps<5:
                    steering,gas,brake=ACTIONS[action]
                    steering=max(-1.,min(1.,steering+rng.choice((-.125,.125))))
                    action=ACTIONS.index((steering,gas,brake))
                handoff_steps+=1
                return action
            try:
                result=episode(config,folder,geometry,training_action,
                    deadline=training_end,training=True,prefix_controls=prefix,
                    on_transition=learner.add,stop_file=args.output/'STOP')
                archive_closed_trace(private,result['trace_dir'],folder/'native-trace.tar.gz')
                failures=0
            except ValueError as error:
                # Invalid observations/progress never enter replay; retain the FAIL.
                # A new environment resets progress; repeated failures stop explicitly.
                failures+=1;report['last_episode_error']=repr(error)
                report['training_episodes']+=1
                if failures>=3:raise
                continue
            report['training_episodes']+=1
            goal=result.get('actual_goal_ms')
            rl_updates=learner.accepted_RL_updates
            if not len(prefix) and goal is not None and rl_updates>0 and (report['best_full_start_goal_ms'] is None or goal<=report['best_full_start_goal_ms']):
                best=folder/'driving-policy.pt'
                torch.save(dict(state=actor_state,RL_updates=learner.accepted_RL_updates,
                    auxiliary_imitation_updates=learner.imitation_updates,
                    epsilon=epsilon,actual_goal_ms=goal,policy_kind='BC_then_QR'),best)
                report.update(best_full_start_goal_ms=goal,best_driving_policy=str(best),best_policy_sha256=sha(best))
            for _ in range(min(16 if args.functional_test else 64,max(1,int(result['policy_steps']*.05)))):
                if time.monotonic()>=training_end or(args.output/'STOP').exists():break
                metrics=learner.update()
                if metrics:report['retention_diagnostics']={k:metrics[k] for k in ('auxiliary_margin','anchor_errors_before_update','optimizer_steps')}
            # Keep the proven actor serving collection while the candidate learns.
            # Only a real, greedy, full-start finish can replace that actor.
            if report['training_episodes']%2==0 and time.monotonic()<training_end-15:
                champion={k:v.detach().cpu().clone() for k,v in learner.actor.state_dict().items()}
                candidate={k:v.detach().cpu().clone() for k,v in learner.learner.model.state_dict().items()}
                learner.actor.load_state_dict(candidate)
                candidate_folder=args.output/f'candidate-{number:06d}'
                try:
                    admitted=episode(config,candidate_folder,geometry,lambda state:learner.action(state,0.),deadline=training_end)
                    archive_closed_trace(private,admitted['trace_dir'],candidate_folder/'native-trace.tar.gz')
                finally:
                    learner.actor.load_state_dict(champion)
                accepted=learner.consider_candidate(admitted,candidate)
                torch.save(dict(state=candidate,RL_updates=learner.learner.update_count),candidate_folder/'candidate-policy.pt')
                write(candidate_folder/'admission.json',dict(accepted=accepted,goal_ms=admitted.get('actual_goal_ms'),
                    candidate_RL_updates=learner.learner.update_count,accepted_RL_updates=learner.accepted_RL_updates,
                    accepted_goal_ms=learner.accepted_goal_ms,policy_sha256=sha(candidate_folder/'candidate-policy.pt')))
                report.update(accepted_goal_ms=learner.accepted_goal_ms,accepted_RL_updates=learner.accepted_RL_updates,
                    rejected_candidates=learner.rejected_candidates,last_candidate_goal_ms=admitted.get('actual_goal_ms'))
            saved=learner.checkpoint();temporary=args.output/'latest.tmp'
            torch.save(saved,temporary);temporary.replace(args.output/'latest-full-checkpoint.pt')
            report.update(status='training',new_real_transitions=learner.transitions,
                RL_updates=learner.learner.update_count,auxiliary_imitation_updates=learner.imitation_updates,
                elapsed_seconds=time.monotonic()-started)
            write(args.output/'state.json',report)
        report.setdefault('stop_reason','operator_STOP' if(args.output/'STOP').exists() else 'functional_test_episode_cap' if args.functional_test else 'training_budget_reserve')
        # Final trials are frozen and include ALL failures in the denominator.
        # Select the actual driving snapshot, never the weights updated afterwards.
        report['evaluated_policy_RL_updates']=learner.accepted_RL_updates
        torch.save(dict(state={k:v.detach().cpu().clone() for k,v in learner.actor.state_dict().items()},
            RL_updates=report['evaluated_policy_RL_updates'],route_sha256=receipt['route_sha256'],
            static_reference_geometry=True,policy_kind='BC_then_QR' if learner.accepted_RL_updates else 'BC_fallback'),args.output/'evaluated-policy.pt')
        report['evaluated_policy_sha256']=sha(args.output/'evaluated-policy.pt')
        learner.demo=None
        prefix_actions=None
        def deny_teacher(event,parameters):
            if event!='open' or not isinstance(parameters[0],(str,bytes)):return
            mode=parameters[1] if len(parameters)>1 else None
            if isinstance(mode,str) and mode.startswith(('w','a','x')):return
            if any(token in str(parameters[0]).lower() for token in ('training-only','source-controls','actual-rollout.npz')):
                raise RuntimeError('teacher-control read during evaluation')
        sys.addaudithook(deny_teacher)
        report['status']='evaluating';write(args.output/'state.json',report)
        for trial in range(2 if args.functional_test else 100):
            if time.monotonic()>=deadline-10 or(args.output/'STOP').exists():break
            eval_rng=random.Random(args.seed+100000+trial);epsilon=0. if trial==0 else .005
            row=episode(config,args.output/f'evaluation-{trial:03d}',geometry,
                lambda state:learner.action(state,epsilon,eval_rng),deadline=deadline-10)
            archive_closed_trace(private,row['trace_dir'],args.output/f'evaluation-{trial:03d}'/'native-trace.tar.gz')
            row.update(trial=trial,seed=args.seed+100000+trial,epsilon=epsilon)
            report['evaluation'].append(row);write(args.output/'state.json',report)
        goals=[r['actual_goal_ms'] for r in report['evaluation'] if r['status']=='NATIVE_GOAL']
        report.update(status='completed',evaluation_trials=len(report['evaluation']),
            evaluation_finishes=len(goals),best_evaluation_goal_ms=min(goals) if goals else None,
            wins_vs_17712=sum(g<17712 for g in goals),improvements_over_BC=sum(g<17601 for g in goals),
            teacher_control_reads_during_evaluation=False,learner_updates_during_evaluation=0,
            Windows_TICK_model_verification_pending=True,visual_ghost_pending=True)
    except Exception as error:
        report.update(status='failed',error=repr(error))
    finally:
        if learner:
            import torch
            torch.save(learner.checkpoint(),args.output/'final-full-checkpoint.pt')
        report['elapsed_seconds']=time.monotonic()-started
        write(args.output/'completion.json',report)
        print(json.dumps({k:v for k,v in report.items() if k!='evaluation'}))
    if report['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
