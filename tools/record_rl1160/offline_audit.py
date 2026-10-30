"""CPU-only audit of actual saved observations and expanded policy actions."""
from pathlib import Path
import copy
import hashlib
import json
import sys

OUT=Path(sys.argv[1])
PRIVATE=Path('/home/kuba/tmrl-full-qr-20261022-011')/OUT.name
sys.path.insert(0,str(PRIVATE/'source'))
import numpy as np
import torch
import yaml
from trackmaniarl.core.builtins import TorchCheckpointCodec
from trackmaniarl.models.factory import CompositeValueModelFactory
from trackmaniarl.models.contracts import ValuePhase

torch.set_num_threads(1)
torch.set_grad_enabled(False)
plan=json.loads((OUT/'plan.json').read_text())
study=OUT.parent
qual=study/'line-qualification1135/model-finalization1138'
search=study/'line-search1136/model-finalization'
raw=yaml.safe_load((OUT/'model-spec.yaml').read_text())
oldraw=yaml.safe_load((qual/'model-spec.yaml').read_text())
codec=TorchCheckpointCodec()
models=[]
for config,path in [(oldraw,study/'line-search1136/protected/qualified-policy.pt'),(raw,OUT/'baseline-policy.pt')]:
    model=CompositeValueModelFactory(**config['components']['model_factory']['kwargs']).build().eval()
    policy=codec.load(path)
    for k,v in policy['online'].items():getattr(model,k).load_state_dict(v,strict=True)
    models.append(model)
with np.load(qual/'collection/observations-TRAINING-ONLY.npz',allow_pickle=False) as z:
    names=json.loads(str(z['observation_keys'].item()))
    obs={k:torch.from_numpy(z['observation/'+k][::35].copy()) for k in names}
with np.load(study/'line-search1136/protected/qualified-macro-catalog.npz',allow_pickle=False) as z:old=z['programs'].tolist()
with np.load(OUT/'macro-catalog.npz',allow_pickle=False) as z:new=z['programs'].tolist()
mapping=[new.index(p) for p in old]
logits=[]
for model in models:
    features=model.encode_frames(obs)
    logits.append(model.head.evaluate_all(features,model.support(features,ValuePhase.EVALUATE)).mean(-2))
torch.testing.assert_close(logits[0],logits[1][:,mapping],atol=2e-4,rtol=2e-5)
selected_old=logits[0].argmax(-1).tolist()
selected_new=logits[1].argmax(-1).tolist()
if [old[i] for i in selected_old]!=[new[i] for i in selected_new]:raise RuntimeError('Catalog expansion changed tested greedy motor programs')
with np.load(search/'collection/observations-TRAINING-ONLY.npz',allow_pickle=False) as z:
    keys=json.loads(str(z['observation_keys'].item()))
    groups={}
    for i,a in enumerate(z['actions'].tolist()):
        digest=hashlib.sha256(b''.join(z['observation/'+k][i].tobytes() for k in sorted(keys))).hexdigest()
        groups.setdefault(digest,set()).add(a)
    aliases=sum(len(values)>1 for values in groups.values())
    rows=len(z['actions'])
if aliases:raise RuntimeError('Identical saved observations require different labels; BC not identifiable')
report=dict(status='PASS_OFFLINE_REAL_DATA_AUDIT',CPU_only=True,native_processes_started=0,
    real_observation_checks=len(selected_old),max_old_action_Q_error=float((logits[0]-logits[1][:,mapping]).abs().max()),
    motor_programs_preserved=True,fastest_demo_decisions=rows,conflicting_identical_observations=aliases,
    timing_note='Each 50 ms action explicitly contains five 10 ms commands. These two demonstrated tapes are exactly representable; 50 ms decisions alone do not prove a timing limitation.')
(OUT/'offline-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
