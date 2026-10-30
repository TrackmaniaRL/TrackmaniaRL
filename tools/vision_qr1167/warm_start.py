"""Bounded supervised initialization of the QR policy; never counted as RL."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from tools.vision_qr1167.learning import Learning
from trackmaniarl.models.contracts import RiskSpec,ValuePhase


def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('dataset',type=Path);p.add_argument('output',type=Path);p.add_argument('--phase-balanced',action='store_true');p.add_argument('--max-updates',type=int,choices=(300,1000,3000),default=300);a=p.parse_args()
 if a.output.exists():raise ValueError('fresh initialization output required')
 receipt=json.loads((a.dataset/'receipt.json').read_text())
 payload=(a.dataset/'demonstrations.npz').read_bytes()
 if receipt['status']!='ADMITTED_TEACHER_WINDOWS' or hashlib.sha256(payload).hexdigest()!=receipt['dataset_sha256']:raise ValueError('dataset admission')
 data=np.load(a.dataset/'demonstrations.npz',allow_pickle=False)
 torch.manual_seed(1167)
 learning=Learning(device='cuda');model=learning.learner.model
 obs=torch.from_numpy(data['observations']).to('cuda');labels=torch.from_numpy(data['actions']).long().to('cuda')
 # Fixed chronological held-out fifth; report as same-lap validation only.
 validation=torch.arange(len(labels),device='cuda')%5==0
 train=~validation
 weights=torch.ones(int(train.sum()),device='cuda')
 if a.phase_balanced:
  phases=torch.from_numpy(data['race_ms']//2000).to('cuda')[train]
  _,inverse,counts=torch.unique(phases,return_inverse=True,return_counts=True)
  weights=1/counts[inverse].float();weights=weights/weights.mean()
 optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
 started=time.monotonic();steps=0
 def logits(x):
  features=model.encode_frames(x);support=model.support(features,ValuePhase.EVALUATE)
  return model.expected_all_actions(features,support,RiskSpec())
 initial=None
 while steps<a.max_updates and time.monotonic()-started<60:
  optimizer.zero_grad(set_to_none=True)
  q=logits(obs[train])
  loss=(torch.nn.functional.cross_entropy(q,labels[train],reduction='none')*weights).mean()+.1*q.square().mean()
  if initial is None:initial=float(loss.detach())
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step();steps+=1
  if not all(torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('nonfinite BC weights')
 with torch.no_grad():
  measured_q=logits(obs)
  prediction=measured_q.argmax(-1)
  max_abs_q=float(measured_q.abs().max())
  if max_abs_q>20:raise ValueError('BC Q-scale admission failed')
  train_accuracy=float((prediction[train]==labels[train]).float().mean())
  validation_accuracy=float((prediction[validation]==labels[validation]).float().mean())
  phase_agreement=[]
  for phase in np.unique(data['race_ms']//2000):
   phase_mask=torch.from_numpy(data['race_ms']//2000==phase).to('cuda')
   phase_agreement.append(dict(start_ms=int(phase*2000),rows=int(phase_mask.sum()),
       agreement=float((prediction[phase_mask]==labels[phase_mask]).float().mean())))
 from tools.vision_qr1167.quantile_initialization import initialize_point_mass
 initialize_point_mass(model.head)
 learning.learner.target_model.load_state_dict(model.state_dict())
 a.output.mkdir(exist_ok=False)
 torch.save(learning.checkpoint(),a.output/'initialization.pt')
 report=dict(status='BC_INITIALIZATION_ONLY',rl_updates=0,bc_updates=steps,
  initial_loss=initial,final_loss=float(loss.detach()),train_accuracy=train_accuracy,max_abs_q=max_abs_q,q_regularization=.1,
  same_lap_validation_accuracy=validation_accuracy,generalization_claim=False,quantile_prior='point mass at supervised mean Q; RL subsequently learns return spread',
  dataset_sha256=receipt['dataset_sha256'],policy_sha256=hashlib.sha256((a.output/'initialization.pt').read_bytes()).hexdigest(),
  phase_balanced=a.phase_balanced,phase_agreement=phase_agreement,
  elapsed_seconds=time.monotonic()-started,max_updates=a.max_updates,max_seconds=60,
  autonomous_finish_verified=False)
 (a.output/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
