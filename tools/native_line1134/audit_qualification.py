"""Independent CPU readback of completed native and autonomous qualification."""
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root/'source'))
sys.path.insert(0,'/home/kuba/tmrl-full-qr-20261022-011/shortcut-multi-source1094')
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from backend import identity,save
from core import native_goal
import torch
from trackmaniarl.core.builtins import TorchCheckpointCodec

torch.set_num_threads(1)
completion=json.loads((root/'completion.json').read_text())
assert completion['phase']=='qualification_complete'
assert identity(completion['parent']['pid'])!=completion['parent']
records=0
for folder in sorted((root/'batches').iterdir()):
    receipt=json.loads((folder/'result.json').read_text())
    assert receipt['exit_code']==0 and receipt['exact_native_identity_gone']
    assert not any(receipt[k] for k in ('remaining_descendants','forced','cleanupErrors'))
    assert identity(receipt['native_identity']['pid'])!=receipt['native_identity']
    raw=gzip.decompress((folder/'native-output.json.gz').read_bytes())
    assert hashlib.sha256(raw).hexdigest()==receipt['full_output_sha256']
    decoded=json.loads(raw)
    assert len(decoded)==receipt['native_count']
    for row in decoded:native_goal(row)
    records+=len(decoded)
fitted=json.loads((root/'finalization.json').read_text())
output=Path(fitted['policy']).parent.parent
for name,role in [('collection','collect'),('training','train'),('evaluation','evaluate')]:
    report=json.loads((output/name/'supervision.json').read_text())
    assert report['status']=='PASS_ROLE' and report['exits']=={role:0}
    assert not any(report['closure'][k] for k in ('remainingOwned','cleanupErrors','forced'))
    for key in ('worker','supervisor'):
        saved=report['identities'][key];actual=identity(saved['pid'])
        assert actual is None or any(actual[k]!=saved[k] for k in ('pid','birth','uid','sid','pgid'))
policy=Path(fitted['policy'])
assert hashlib.sha256(policy.read_bytes()).hexdigest()==fitted['policy_sha256']
weights=TorchCheckpointCodec().load(policy)
tensors=[v for state in weights['online'].values() for v in state.values()]
assert len(tensors)==90 and all(torch.isfinite(v).all() for v in tensors)
evaluation=json.loads((output/'evaluation/result.json').read_text())
assert evaluation['actual_goals_ms']==[17705]*5 and evaluation['teacher_files_opened']==0 and evaluation['weights_unchanged']
save(root/'independent-audit1138.json',dict(status='PASS_NATIVE_SEARCH_AND_FIVE_AUTONOMOUS_QUALIFICATION_NOT_OVERNIGHT',
    native_records=records,actual_goals_ms=evaluation['actual_goals_ms'],finite_online_tensors=90,
    policy_sha256=fitted['policy_sha256'],regular_optimizer_exported=False,
    remainingOwned=[],cleanupErrors=[],forced=[],visual_ghost_pending=True,
    prior_helper_failures_preserved=['model-finalization','model-finalization1137']))
print('PASS qualification; five autonomous17.705s; visual export remains pending')
