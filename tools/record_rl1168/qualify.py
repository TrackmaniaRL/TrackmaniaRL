"""Bounded, CPU, full-start autonomous native qualification of a frozen policy."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def linux_path(path):return path.replace('F:\\','/mnt/f/').replace('F:/','/mnt/f/').replace('\\','/')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('preparation',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--trials',type=int,default=3,choices=(1,3))
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    plan=json.loads((args.preparation/'plan.json').read_text())
    receipt=json.loads((args.preparation/'preparation.json').read_text())
    report=dict(status='FAILED',policy_kind='BC_ONLY',RL_updates=0,trials=[],cleanupErrors=[],
        teacher_control_reads=False,static_reference_geometry=True,public_WR_claim=False)
    started=time.monotonic()
    try:
        checkpoint=args.preparation/'initialization.pt'
        if sha(checkpoint)!=receipt['checkpoint_sha256'] or sha(args.preparation/'route-geometry.npy')!=receipt['route_sha256']:
            raise ValueError('initializer/route changed')
        manifest=Path(linux_path(plan['backend_source_manifest']))
        if sha(manifest)!=plan['backend_source_manifest_sha256']:raise ValueError('backend manifest changed')
        config=dict(plan['backend_config'])
        for item in json.loads(manifest.read_text()):
            if item['path'].startswith(config['source']+'/') and sha(item['path'])!=item['sha256']:
                raise ValueError('frozen backend source changed: '+item['path'])
        private=Path('/home/kuba/tmrl-full-qr-20261022-011')/(args.preparation.name+'-'+args.output.name)
        private.mkdir(mode=0o700,exist_ok=False)
        from tools.record_rl1168.runtime import isolate_runtime
        config=isolate_runtime(config,private)
        sys.path.insert(0,config['source'])
        import numpy as np
        import torch
        from environment import FallEnvironment
        from tools.record_rl1168.model import build_model,ACTIONS
        from tools.record_rl1168.features import RouteFeatures
        from trackmaniarl.models.contracts import RiskSpec,ValuePhase
        torch.set_num_threads(2)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if saved['RL_updates']!=0:raise ValueError('this admission is for BC initialization only')
        model=build_model();model.load_state_dict(saved['state']);model.eval()
        geometry=np.load(args.preparation/'route-geometry.npy',allow_pickle=False)
        def no_teacher(event,parameters):
            if event!='open' or not isinstance(parameters[0],(str,bytes)):return
            mode=parameters[1] if len(parameters)>1 else None
            if isinstance(mode,str) and mode.startswith(('w','a','x')):return
            if any(word in str(parameters[0]).lower() for word in ('training-only','source-controls','actual-rollout.npz')):
                raise RuntimeError('teacher-control read in frozen evaluation')
        sys.addaudithook(no_teacher)
        for trial in range(args.trials):
            env=None;row=dict(trial=trial,status='FAILED',actual_goal_ms=None)
            report['trials'].append(row)
            begin=time.monotonic()
            try:
                env=FallEnvironment(config,args.output/f'trial-{trial}')
                env.settings=replace(env.settings,trace_dir=private/'traces'/f'trial-{trial}')
                route=RouteFeatures(geometry)
                state,_,_=route.observe(env.reset(),reset=True)
                for step in range(3000):
                    if time.monotonic()-started>120:raise TimeoutError('qualification total120s budget')
                    with torch.no_grad():
                        tensor=torch.from_numpy(state[None])
                        features=model.encode_frames(tensor)
                        values=model.expected_all_actions(features,model.support(features,ValuePhase.EVALUATE),RiskSpec())
                        action=int(values.argmax(-1))
                    native,_,goal,truncated,info=env.step(ACTIONS[action])
                    state,progress,lateral=route.observe(native)
                    row.update(steps=step+1,progress_m=progress,lateral_m=lateral,race_ms=info['race_ms'])
                    if goal or truncated:
                        row.update(status='NATIVE_GOAL' if goal else 'NO_GOAL',actual_goal_ms=info['actual_goal_ms'])
                        break
            finally:
                if env:
                    env.close()
                    row['closure']=json.loads((env.settings.trace_dir/'native-supervisor.json').read_text())
                    closed=row['closure']
                    if closed['returncode']!=0 or closed['remainingOwned'] or closed['cleanupErrors']:
                        report['cleanupErrors'].append(dict(trial=trial,closure=closed))
                row['elapsed_seconds']=time.monotonic()-begin
                (args.output/'result.json').write_text(json.dumps(report,indent=2))
        if sha(checkpoint)!=receipt['checkpoint_sha256']:raise ValueError('checkpoint mutated')
        report['status']='PASS_FULL_START_BC' if not report['cleanupErrors'] and all(
            r['actual_goal_ms']==17601 for r in report['trials']) else 'BC_NOT_QUALIFIED'
    except Exception as error:
        report['error']=repr(error)
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        report['checkpoint_sha256']=receipt['checkpoint_sha256']
        (args.output/'result.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k!='trials'}))
        print(json.dumps([dict(trial=r['trial'],status=r['status'],goal=r.get('actual_goal_ms'),
                              steps=r.get('steps'),elapsed=r.get('elapsed_seconds')) for r in report['trials']]))


if __name__=='__main__':main()
