"""Three real native episodes with strict original watchdogs and eager closure."""
from pathlib import Path
import copy,hashlib,json,sys,time
out=Path(sys.argv[1]);plan=json.loads((out/'plan.json').read_text());private=Path(plan['private']);sys.path.insert(0,plan['source'])
import torch,yaml
from trackmaniarl.models.factory import CompositeValueModelFactory
from trackmaniarl.core.builtins import TorchCheckpointCodec
from trackmaniarl.algorithms.value_based.policy import DiscreteValuePolicy,DiscreteValuePolicyConfig
from trackmaniarl.core.contracts import PolicyMode
from trackmaniarl.models.contracts import RiskSpec
from trackmaniarl.trackmania.native_headless.full_qr.factory import FullQrEnvironmentFactory
from trackmaniarl.trackmania.native_headless._runtime.ownership import process_identity,same_identity
torch.set_num_threads(1);torch.set_grad_enabled(False)
raw=yaml.safe_load((private/'config.yaml').read_text());model=CompositeValueModelFactory(**raw['components']['model_factory']['kwargs']).build().eval();saved=TorchCheckpointCodec().load(private/'warm-policy.pt')['learner']
for k,v in saved['online'].items():getattr(model,k).load_state_dict(v,strict=True)
policy=DiscreteValuePolicy(model,torch.device('cpu'),DiscreteValuePolicyConfig(0,None,RiskSpec(),RiskSpec()))
cfg=copy.deepcopy(raw['components']['environment']['kwargs']['config']);cfg['runtime'].update(trace_dir=str(private/'smoke/traces'),work_dir=str(Path(cfg['runtime']['work_dir'])/'smoke'));cfg['trace_archive_root']=str(out/'smoke/archives')
env=FullQrEnvironmentFactory(cfg).create(seed=17);rows=[];started=time.monotonic()
try:
 for episode in range(3):
  policy.reset_episode();obs,_=env.reset(seed=17+episode);identity=process_identity(env.provider._session.pid)
  for step in range(3001):
   if time.monotonic()-started>540:raise RuntimeError('Bounded three-episode validation deadline')
   obs,reward,done,truncated,info=env.step(int(policy.act(obs,PolicyMode.EVALUATION)))
   if done or truncated:break
  else:raise RuntimeError('No actual terminal within native150s horizon')
  if env.provider._session is not None or same_identity(identity,process_identity(identity['pid'])):raise RuntimeError('Native lease survived terminal transition')
  receipt=env.provider.archive_receipts[-1]
  rows.append(dict(episode=episode,steps=step+1,reason=info['termination_reason'],race_time_ms=info['race_time_ms'],identity=identity,native_gone_before_actor_wait=True,archive=receipt))
  # Actual actor-side waiting now occurs after exact native closure.
  time.sleep(1)
finally:env.close()
report=dict(status='PASS_THREE_REAL_EPISODES_AND_EAGER_CLOSURE',trials=rows,watchdog_seconds=240,driver_episode_limit=1,weights_source_sha256=plan['original_checkpoint_sha256'],training_updates=0)
(out/'smoke-result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
