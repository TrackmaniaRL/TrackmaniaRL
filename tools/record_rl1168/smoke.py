"""At most180s: validate real RL updates and post-update native full starts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('preparation',type=Path);parser.add_argument('qualification',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    started=time.monotonic();report=dict(status='FAILED',bounded_functional_test=True,RL_updates=0,
                                       training_episodes=[],evaluation=[],public_WR_claim=False)
    try:
        proof=json.loads((args.qualification/'result.json').read_text())
        receipt=json.loads((args.preparation/'preparation.json').read_text())
        if proof['status']!='PASS_FULL_START_BC' or proof['cleanupErrors'] or proof['checkpoint_sha256']!=receipt['checkpoint_sha256']:
            raise ValueError('three clean full-start native initializer trials required')
        if len(proof['trials'])!=3:raise ValueError('three initializer trials required')
        plan=json.loads((args.preparation/'plan.json').read_text())
        config=dict(plan['backend_config']);sys.path.insert(0,config['source'])
        from tools.record_rl1168.runtime import isolate_runtime
        from tools.record_rl1168.learner import Learner
        from tools.record_rl1168.episodes import episode
        import numpy as np
        import torch
        torch.set_num_threads(2);torch.manual_seed(1168)
        private=Path('/home/kuba/tmrl-full-qr-20261022-011')/(args.preparation.name+'-'+args.output.name)
        private.mkdir(mode=0o700,exist_ok=False);config=isolate_runtime(config,private)
        initial_path=args.preparation/'initialization.pt'
        if hashlib.sha256(initial_path.read_bytes()).hexdigest()!=receipt['checkpoint_sha256']:raise ValueError('initializer changed')
        initial=torch.load(initial_path,map_location='cpu',weights_only=True)
        learner=Learner(initial,device='cpu')
        learner.configure_demonstrations(args.preparation/'demonstrations-TRAINING-ONLY.npz')
        geometry=np.load(args.preparation/'route-geometry.npy',allow_pickle=False)
        for number,epsilon in enumerate((0.,.005)):
            learner.begin_episode()
            row=episode(config,args.output/f'train-{number}',geometry,
                lambda state:learner.action(state,epsilon),deadline=started+180,
                training=True,on_transition=learner.add)
            report['training_episodes'].append(row)
        metrics=[]
        for update in range(16):
            if time.monotonic()-started>=150:raise TimeoutError('functional update budget')
            metrics.append(learner.update())
        learner.sync_actor()
        saved=learner.checkpoint()
        torch.save(saved,args.output/'smoke-RL.pt')
        before=initial['state'];after=saved['state']
        changed=[k for k in before if not torch.equal(before[k],after[k])]
        if not any('encoder' in k for k in changed) or not any('head' in k for k in changed):
            raise RuntimeError('missing actual full-policy update')
        # Stop all learning; this separate evaluation uses only the frozen actor.
        checksum=hashlib.sha256((args.output/'smoke-RL.pt').read_bytes()).hexdigest()
        for number in range(2):
            report['evaluation'].append(episode(config,args.output/f'eval-{number}',geometry,
                lambda state:learner.action(state,0.),deadline=started+180))
        if hashlib.sha256((args.output/'smoke-RL.pt').read_bytes()).hexdigest()!=checksum:raise ValueError('evaluation changed checkpoint')
        report.update(status='PASS_REAL_RL_FUNCTIONAL_TEST',RL_updates=saved['RL_updates'],
            auxiliary_imitation_updates=saved['auxiliary_imitation_updates'],new_real_transitions=saved['transitions'],
            changed_tensor_names=changed,checkpoint_sha256=checksum,
            learner_updates_during_evaluation=0,training_seconds_authorized=0)
    except Exception as error:
        report['error']=repr(error)
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        (args.output/'result.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k not in ('training_episodes','evaluation','changed_tensor_names')}))
        print(json.dumps([dict(status=r['status'],goal=r.get('actual_goal_ms')) for r in report['evaluation']]))


if __name__=='__main__':main()
