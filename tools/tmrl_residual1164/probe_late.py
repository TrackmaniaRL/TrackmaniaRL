"""One bounded functional check of a predeclared smaller Gaussian variance."""
from pathlib import Path
import hashlib,json,os,sys,time
out=Path(sys.argv[1]);cfg=json.loads((out/'environment.json').read_text());sys.path.insert(0,cfg['source']);private=Path(cfg['private'])
from trackmaniarl.experiments.full_qr.campaign_processes import CampaignProcessCohort
from trackmaniarl.trackmania.native_headless._runtime.ownership import process_identity
if len(sys.argv)>2:
 import torch
 from model import Policy,decode
 from environment import FallEnvironment
 torch.set_num_threads(1);torch.manual_seed(1166);torch.set_grad_enabled(False)
 model=Policy();source=out/'bc-policy.pt';state=torch.load(source,map_location='cpu',weights_only=True);model.load_state_dict(state['state']);model.residual_std.fill_(.1);model.eval()
 rows=[]
 for trial in range(2):
  torch.manual_seed(116600+trial);folder=out/f'late-check1166-{trial}';env=FallEnvironment(cfg,folder)
  try:
   obs=env.reset()
   for index in range(3001):
    control,*_=model.act(torch.from_numpy(obs).unsqueeze(0),stochastic=True)
    if round(float(obs[0])*30000)<16500:
     s,l=model.base_logits(torch.from_numpy(obs).unsqueeze(0));control=decode(torch.stack((s.argmax(-1),l.argmax(-1)),-1))
    obs,_,done,truncated,info=env.step(control[0].tolist())
    if done or truncated:break
   else:raise RuntimeError('Native30s domain exhausted')
  finally:env.close()
  closure=json.loads((env.settings.trace_dir/'native-supervisor.json').read_text())
  if closure['remainingOwned'] or closure['cleanupErrors'] or closure['returncode']!=0:raise RuntimeError('Probe native closure failed')
  rows.append(dict(trial=trial,actual_goal_ms=info['actual_goal_ms'],steps=index+1,closure=closure))
 report=dict(status='PASS_LOCAL_VARIANCE_FINISHES' if all(r['actual_goal_ms'] is not None for r in rows) else 'LOCAL_VARIANCE_DNF',latent_std=.1,motor_scale=.01,residual_after_ms=16500,baseline_policy_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),RL_updates=0,trials=rows,record_claim=False)
 (out/'late-check1166.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
else:
 for tag in ('train','qualify'):
  path=out/(tag+'-supervision.json')
  if path.exists() and any(json.loads(path.read_text())['closure'][k] for k in ('remainingOwned','cleanupErrors','forced')):raise RuntimeError('Prior ownership closure not clean')
 if json.loads((out/'power-guard-state.json').read_text())['phase']!='released':raise RuntimeError('Prior controller still running')
 cohort=CampaignProcessCohort(private/'late-probe1166',lambda:None);started=time.monotonic();report=dict(parent=process_identity(os.getpid()),status='FAILED')
 try:
  report['process']=cohort.start('probe',[sys.executable,'-B',str(Path(__file__).resolve()),str(out),'worker'],cwd=Path(cfg['source']))
  while not cohort.exited():
   if time.monotonic()-started>180:raise RuntimeError('Bounded variance test deadline')
   time.sleep(.5)
  report['exits']=cohort.exited()
  if report['exits']!={'probe':0}:raise RuntimeError('Probe process failed')
  report['status']='PASS_PROBE_PROCESS_CLOSED'
 finally:
  report['closure']=cohort.close(grace_seconds=30);(out/'late-check1166-supervision.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report))
