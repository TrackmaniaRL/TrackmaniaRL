"""Admission from actual two-carrier Goal agreement, plain validation and tests."""
from pathlib import Path
import hashlib,json,sys
out=Path(sys.argv[1]);cfg=json.loads((out/'environment-independent.json').read_text());sys.path.insert(0,cfg['source'])
from environment import FallEnvironment
from model import Policy,controls
import torch
torch.set_num_threads(1);torch.manual_seed(1161)

def clean(closure):
 return closure.get('remainingOwned')==[] and closure.get('cleanupErrors')==[] and closure.get('forced',[])==[]

for tag in ('qualification-v6','qualification-v7'):
 result=json.loads((out/tag/'result.json').read_text());supervision=json.loads((out/(tag+'-supervision.json')).read_text())
 if result.get('info',{}).get('actual_goal_ms')!=21758 or supervision['exits']!={'qualify':0} or not clean(supervision['closure']) or not clean(result['supervisor']):raise RuntimeError('Two real native source/control-independent finishes and exact closure required')
plain=json.loads((out/'plain-carrier-v1.json').read_text())
if not clean(plain['closure']) or len(plain['records'])!=1 or plain['records'][0]['ValidatedResult']['Time']!=21758 or plain['records'][0]['IsValid'] is not True:raise RuntimeError('Independent unchanged native replay validation required')
env=FallEnvironment(cfg,out/'untrained-causal-policy-test');model=Policy().eval();steps=[]
try:
 obs=env.reset()
 for _ in range(20):
  with torch.no_grad():
   s,b,_=model(torch.from_numpy(obs).unsqueeze(0));action=controls(torch.stack((s.argmax(-1),b.argmax(-1)),-1))[0].tolist()
  obs,_,finished,truncated,info=env.step(action);steps.append(dict(action=action,race_ms=info['race_ms'],finished=finished))
  if finished or truncated:raise RuntimeError('Unexpected terminal in 200ms independent random-policy transport test')
finally:env.close()
closure=json.loads((env.settings.trace_dir/'native-supervisor.json').read_text())
if not clean(closure) or closure.get('returncode')!=0:raise RuntimeError('Random policy native closure failed')
report=dict(status='PASS_FALL05_CAUSAL_PHYSICS_STEP_ENVIRONMENT',map_uid='v96wCr7mP5xEGxNLSnZC6DAp4se',
 observation='17 measured native physical scalars; race time, quaternion, position, velocity, rpm, gear and current controls; no reference trajectory lookup',
 action='model-selected steering [-127,127]/127, binary brake, gas1 every10ms',
 reset='literal native2190 pre-input bootstrap; race origin2200; model controls initial interval as well',
 actual_native_goals_ms=[21758,21758],unchanged_plain_validated_goal_ms=21758,
 unrelated_carrier_same_physics=True,untrained_causal_network_actual_steps=steps,untrained_policy_closure=closure,
 full_V5_motor_or_UI_qualification_claimed=False,Windows_TICK_model_deployment_verified=False,RL_started=False,
 readiness_scope='Qualified for this pinned Linux engine/map/reset and measured physical-state observation contract; no claim of client UI or public ranking acceptance')
(out/'environment-admission.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('untrained_causal_network_actual_steps','untrained_policy_closure')}))
