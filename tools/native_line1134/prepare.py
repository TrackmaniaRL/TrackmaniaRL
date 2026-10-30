"""Fresh self-contained report/source tree; preserve the qualified baseline."""
from __future__ import annotations
import hashlib
import gzip
import json
import re
import shutil
import sys
from pathlib import Path
import numpy as np

BASE=Path('/home/kuba/tmrl-full-qr-20261022-011')

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main() -> None:
    root=Path(sys.argv[1]).resolve()
    if not re.fullmatch(r'line-(?:qualification|search)[0-9]{4}',root.name) or root.exists():
        raise RuntimeError('Fresh named campaign required')
    study=root.parent
    protected=study/'autonomous-line1114'
    policy=protected/'pretraining1115/policy-only.pt'
    ghost=study/'native-capture1126/TrackmaniaRL-AI-17.712-NATIVE.Ghost.Gbx'
    if sha(policy)!='bb22eba32ca1da83da6f3346e7ad74e67cc55fdaaa455d486f0792e94221b0d4' or sha(ghost)!='221d9960f6dda43bd71b1c5205ee4e13c87466080e1224a11c63999fe36d78a9':
        raise RuntimeError('Protected baseline policy/ghost changed')
    with np.load(study/'local-search1113/candidate-plans.npz',allow_pickle=False) as z:
        ticks=list(map(int,z['motor_ticks'][8]))
    root.mkdir();(root/'source').mkdir();(root/'protected').mkdir()
    names=['core.py','backend.py','search.py']
    if 'search' in root.name:
        names+=['finalize.py']
    for name in names:
        shutil.copyfile(Path(__file__).parent/name,root/'source'/name)
    for name in ('launch.ps1','keep_awake.ps1','check.py'):
        shutil.copyfile(Path(__file__).parent/name,root/name)
    for source,dest in [(policy,'policy17.712.pt'),(ghost,'native17.712.Ghost.Gbx'),
                        (protected/'macro-catalog.npz','macro-catalog132.npz')]:
        shutil.copyfile(source,root/'protected'/dest)
    writer=BASE/'native-line1134-target/release/native-line1134'
    plan=dict(status='PREPARED_NOT_STARTED',seed=1136,batch_size=128,
              total_budget_seconds=36000,finalization_reserve_seconds=1800,
              target_improvement_ms=100,stop_on_small_improvement=False,
              writer=str(writer),writer_sha256=sha(writer),
              source_sha256={name:sha(root/'source'/name) for name in names},
              native_calibration='native-score-calibration1134: computed17712/17715/null',
              initial_policy_sha256=sha(policy),initial_native_ghost_sha256=sha(ghost),
              method='digital10ms edge shifts, pulses, combined edits, elite-beam crossover; actual native Goal only',
              no_online_RL=True,no_Windows_game=True,old_model_and_native_ghost_preserved=True)
    qualified=study/'line-qualification1135'
    if 'search' in root.name:
        completion=json.loads((qualified/'completion.json').read_text())
        winner=json.loads((qualified/'best.json').read_text())
        if completion['phase']!='qualification_complete':raise RuntimeError('Completed qualification required')
        cached=set()
        for folder in sorted((qualified/'batches').iterdir()):
            receipt=json.loads((folder/'result.json').read_text())
            if (receipt['status']!='PASS_NATIVE_PHYSICS_BATCH_NOT_AUTONOMOUS_EVALUATION'
                or receipt['exit_code']!=0 or not receipt['exact_native_identity_gone']
                or receipt['remaining_descendants'] or receipt['forced'] or receipt['cleanupErrors']):
                raise RuntimeError('Qualification native closure required')
            spec=json.loads(gzip.decompress((folder/'spec.json.gz').read_bytes()))
            cached.update(hashlib.sha256(bytes(c['ticks'])).hexdigest() for c in spec['candidates'])
        motors=json.loads(Path(winner['motors']).read_text())
        if hashlib.sha256(bytes(motors)).hexdigest()!=winner['motor_sha256']:raise RuntimeError('Qualified input fingerprint')
        shutil.copyfile(winner['input_only_ghost'],root/'qualified-inputs-NOT-VISUAL.Ghost.Gbx')
        (root/'qualified-motors.json').write_text(json.dumps(motors))
        (root/'previously-evaluated-input-sha256.json').write_text(json.dumps(sorted(cached)))
        plan['qualified_seed']=dict(actual_native_goal_ms=winner['actual_native_goal_ms'],
            motor_sha256=winner['motor_sha256'],source=str(qualified/'best.json'),
            fresh_five_native_confirmations_required=True,previously_evaluated=len(cached))
        fitted=json.loads((qualified/'finalization.json').read_text())
        if fitted['best_autonomous_goal_ms']!=winner['actual_native_goal_ms'] or fitted['actual_goals_ms']!=[winner['actual_native_goal_ms']]*5:
            raise RuntimeError('Five autonomous qualified seed trials required')
        if sha(Path(fitted['policy']))!=fitted['policy_sha256']:raise RuntimeError('Qualified model hash')
        shutil.copyfile(fitted['policy'],root/'protected/qualified-policy.pt')
        shutil.copyfile(Path(fitted['policy']).parent.parent/'macro-catalog.npz',root/'protected/qualified-macro-catalog.npz')
        plan['qualified_policy_sha256']=fitted['policy_sha256']
        plan['initial_best_autonomous_goal_ms']=fitted['best_autonomous_goal_ms']
        plan['operations_sha256']={name:sha(root/name) for name in ('launch.ps1','keep_awake.ps1','check.py')}
    (root/'baseline-motors.json').write_text(json.dumps(ticks))
    (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(plan))

if __name__=='__main__':main()
