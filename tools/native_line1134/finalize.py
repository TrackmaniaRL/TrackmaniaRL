"""Bounded real native collection, BC distillation and five autonomous trials.

Input-only search tapes never masquerade as visual replays. Preserve the original
native ghost until a fresh visual export is independently verified.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
import re
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any

BASE=Path('/home/kuba/tmrl-full-qr-20261022-011')
SOURCE=BASE/'shortcut-multi-source1094'
ROOT=Path(sys.argv[1]).resolve()
STUDY=ROOT.parent
ATTEMPT=os.environ.get('TMRL_FINALIZATION_ATTEMPT','model-finalization')
if not re.fullmatch(r'model-finalization(?:[0-9]{4})?',ATTEMPT):raise RuntimeError('Invalid own finalization namespace')
OUT=ROOT/ATTEMPT
PRIVATE=BASE/(ROOT.name+'-'+ATTEMPT)
sys.path.insert(0,str(SOURCE))
sys.dont_write_bytecode=True
if len(sys.argv)>2 and sys.argv[2]=='train':
    os.environ.pop('CUDA_VISIBLE_DEVICES',None)
else:
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['OMP_NUM_THREADS']='2'

def save(p: Path,v: object) -> None:
    stage=p.with_suffix('.stage');stage.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');stage.replace(p)

def read(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8-sig'))

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def stopped(role: str) -> bool:
    return any(p.exists() for p in (ROOT/'STOP',OUT/'STOP',OUT/role/'STOP'))

def config(role: str) -> dict[str,Any]:
    value=read(OUT/'environment.json')
    value['runtime'].update(trace_dir=str(PRIVATE/role/'traces'),
        work_dir=str(BASE/'waypoint-runtime002/build/server-probe'/(ROOT.name+'-'+role)),timeout_s=150)
    value.update(trace_archive_root=str(OUT/role/'closed-traces'),
                 keep_trace_archives=5 if role=='evaluation' else 1,driver_episode_limit=1)
    return value

def collect() -> None:
    import numpy as np
    import torch
    from trackmaniarl.experiments.shortcut_macro import AutonomousMacroShortcutFactory,actual_goal_ms
    from trackmaniarl.trackmania.native_headless._runtime.ownership import process_identity,same_identity
    torch.set_num_threads(1);torch.set_grad_enabled(False)
    plan=read(OUT/'plan.json');cfg=config('collection');environment=AutonomousMacroShortcutFactory(cfg).create(seed=0)
    result: dict[str,Any]={'status':'FAILED','learner_updates':0}
    rows=[];actions=[];native=[];started=time.monotonic()
    try:
        observation,_=environment.reset(seed=0)
        native.append(process_identity(environment.provider._session.pid))
        with np.load(OUT/'controls-TRAINING-ONLY.npz',allow_pickle=False) as archive:
            sequence=archive['actions'].tolist()
        for i,action in enumerate(sequence):
            if time.monotonic()-started>180 or stopped('collection'):raise RuntimeError('Collection deadline/STOP')
            rows.append({k:v.detach().cpu().numpy().copy() for k,v in observation.items()});actions.append(action)
            observation,_,terminated,truncated,info=environment.step(int(action))
            save(OUT/'collection/progress.json',{'decisions':i+1,'observed_physics_ms':info['observed_physics_time_ms']})
            if terminated or truncated:
                actual=actual_goal_ms(environment) if info['termination_reason']=='finished' else None
                if actual!=plan['target_goal_ms']:raise RuntimeError(f'Actual collection Goal {actual} differs from validated {plan["target_goal_ms"]}')
                result.update(status='PASS_REAL_NATIVE_OBSERVATIONS_FOR_BC',actual_goal_ms=actual,
                              decisions=len(actions),physical_time_ms=info['observed_physics_time_ms'])
                break
        else:raise RuntimeError('Plan did not finish within supplied controls')
        arrays={'schema':np.asarray('tmrl.observation-only-bc/v1'),'actions':np.asarray(actions,dtype=np.int64),
                'observation_keys':np.asarray(json.dumps(sorted(rows[0])))}
        arrays.update({'observation/'+k:np.stack([r[k] for r in rows]) for k in rows[0]})
        np.savez_compressed(OUT/'collection/observations-TRAINING-ONLY.npz',**arrays)
    finally:
        environment.close()
        result['archives']=environment.provider.archive_receipts
        result['remaining_native']=[p for p in native if same_identity(p,process_identity(p['pid']))]
        save(OUT/'collection/result.json',result)
    if result['remaining_native']:raise RuntimeError('Collection native closure')

def train() -> None:
    import numpy as np
    import torch
    import yaml
    from trackmaniarl.core.builtins import TorchCheckpointCodec
    from trackmaniarl.models.factory import CompositeValueModelFactory
    from trackmaniarl.experiments.shortcut_pretraining import BehaviorCloningPretrainer
    torch.set_num_threads(2);torch.manual_seed(1136)
    plan=read(OUT/'plan.json')
    if read(OUT/'collection/result.json')['status']!='PASS_REAL_NATIVE_OBSERVATIONS_FOR_BC':raise RuntimeError('Real observations required')
    raw=yaml.safe_load((OUT/'model-spec.yaml').read_text())
    if not torch.cuda.is_available():raise RuntimeError('Bounded BC requires available CUDA; no slow unqualified fallback')
    model=CompositeValueModelFactory(**raw['components']['model_factory']['kwargs']).build().to('cuda')
    codec=TorchCheckpointCodec();original=codec.load(ROOT/'protected/policy17.712.pt')
    if sha(ROOT/'protected/policy17.712.pt')!='bb22eba32ca1da83da6f3346e7ad74e67cc55fdaaa455d486f0792e94221b0d4':raise RuntimeError('Protected policy hash')
    with np.load(ROOT/'protected/macro-catalog132.npz',allow_pickle=False) as z:old=z['programs'].copy()
    with np.load(OUT/'macro-catalog.npz',allow_pickle=False) as z:new=z['programs'].copy()
    lookup={tuple(p):i for i,p in enumerate(old.tolist())}
    mapping=torch.tensor([lookup.get(tuple(p),int(np.count_nonzero(old!=p,axis=1).argmin())) for p in new],dtype=torch.long)
    prior=original['online']['head'];state=model.head.state_dict()
    weights=prior['advantage.weight'].reshape(64,len(old),-1)[:,mapping].clone()
    bias=prior['advantage.bias'].reshape(64,len(old))[:,mapping].clone()
    for i,p in enumerate(new):
        if tuple(p) not in lookup:bias[:,i]-=2
    state['advantage.weight']=weights.reshape(64*len(new),-1)
    state['advantage.bias']=bias.reshape(64*len(new))
    state['value.weight']=prior['value.weight']+weights.mean(1)-prior['advantage.weight'].reshape(64,len(old),-1).mean(1)
    state['value.bias']=prior['value.bias']+bias.mean(1)-prior['advantage.bias'].reshape(64,len(old)).mean(1)
    for name,values in original['online'].items():getattr(model,name).load_state_dict(state if name=='head' else values,strict=True)
    with np.load(OUT/'collection/observations-TRAINING-ONLY.npz',allow_pickle=False) as z:
        obs={k:torch.from_numpy(z['observation/'+k].copy()).to('cuda') for k in json.loads(str(z['observation_keys'].item()))}
        actions=torch.from_numpy(z['actions'].copy()).to('cuda',dtype=torch.long)
    trainer=BehaviorCloningPretrainer(model,actions,learning_rate=3e-5)
    best=None;consecutive=0;started=time.monotonic();history=[]
    for update in range(1,2001):
        if stopped('training'):raise RuntimeError('BC STOP')
        if time.monotonic()-started>480:break
        metrics=trainer.update(obs,actions)
        if not all(math.isfinite(x) for x in metrics.values()):raise RuntimeError('Nonfinite BC metrics')
        if update%10==0:save(OUT/'training/progress.json',dict(updates=update,**metrics))
        if update%25:continue
        model.eval()
        with torch.no_grad():
            logits=trainer.logits(obs);selected=logits.gather(1,actions[:,None]).squeeze(1)
            margin=float((selected-logits.scatter(1,actions[:,None],-torch.inf).max(1).values).min())
            mistakes=int((logits.argmax(1)!=actions).sum())
        history.append(dict(update=update,mistakes=mistakes,margin=margin));save(OUT/'training/fit-history.json',history)
        if mistakes==0:
            if best is None or margin>best[0]:
                online={name:{k:v.detach().cpu().clone() for k,v in getattr(model,name).state_dict().items()} for name in original['online']}
                best=(margin,update,online)
            consecutive=consecutive+1 if margin>=.5 else 0
        else:consecutive=0
        if update>=100 and consecutive>=5:break
    if best is None or best[0]<.1:raise RuntimeError('No fitted policy with safe margin')
    margin,selected,online=best
    tensors=[v for state in online.values() for v in state.values()]
    if len(tensors)!=90 or not all(torch.isfinite(v).all() for v in tensors):raise RuntimeError('90 finite online tensors required')
    policy=dict(schema='tmrl.autonomous-shortcut-policy-only/v1',architecture_fingerprint=model.architecture_fingerprint(),
                online=online,training_updates=selected,source_checkpoint_sha256=sha(ROOT/'protected/policy17.712.pt'),
                macro_catalog_sha256=sha(OUT/'macro-catalog.npz'))
    codec.save(policy,OUT/'training/policy-only.pt')
    save(OUT/'training/result.json',dict(status='PASS_BC_PENDING_AUTONOMOUS_EVALUATION',updates=update,selected=selected,
        minimum_margin=margin,policy_sha256=sha(OUT/'training/policy-only.pt'),new_optimizer=True,online_RL_updates=0,
        finite_online_tensors=90,wall_seconds=time.monotonic()-started,old_action_count=len(old),new_action_count=len(new),device='cuda_float32'))

def evaluate() -> None:
    import torch
    from trackmaniarl.core.builtins import TorchCheckpointCodec
    from trackmaniarl.core.spec import RunSpec
    from trackmaniarl.core.runtime import import_symbol
    from trackmaniarl.core.contracts import EvaluatorRuntimeRequest,PolicyMode
    from trackmaniarl.algorithms.value_based.policy import DiscreteValuePolicy,DiscreteValuePolicyConfig
    from trackmaniarl.models.contracts import RiskSpec
    from trackmaniarl.models.factory import CompositeValueModelFactory
    from trackmaniarl.experiments.shortcut_macro import AutonomousMacroShortcutFactory,AutonomousMacroEvaluator,actual_goal_ms
    from trackmaniarl.trackmania.native_headless._runtime.ownership import process_identity,same_identity
    torch.set_num_threads(1);torch.set_grad_enabled(False)
    spec=RunSpec.from_yaml(OUT/'model-spec.yaml');target=read(OUT/'plan.json')['target_goal_ms']
    checkpoint=OUT/'training/policy-only.pt';checkpoint_sha=sha(checkpoint)
    weights=TorchCheckpointCodec().load(checkpoint)
    blocked=[]
    def deny(event: str,args: tuple[Any,...]) -> None:
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        value=str(args[0]).lower()
        if any(x in value for x in ('training-only','motors.json','/batches/','/winners/','jimmy-17715','candidate-plans')):
            blocked.append(value);raise RuntimeError('Autonomous actor attempted demonstration access')
    sys.addaudithook(deny)
    model=CompositeValueModelFactory(**spec.components.model_factory.kwargs).build().eval()
    if model.architecture_fingerprint()!=weights['architecture_fingerprint']:raise RuntimeError('Architecture fingerprint')
    for name,values in weights['online'].items():getattr(model,name).load_state_dict(values,strict=True)
    frozen={k:v.detach().clone() for k,v in model.state_dict().items()}
    policy=DiscreteValuePolicy(model,torch.device('cpu'),DiscreteValuePolicyConfig(0,None,RiskSpec(),RiskSpec()))
    made=[];identities=[];goals=[];calls=0;started=time.monotonic()
    class ObservedFactory(AutonomousMacroShortcutFactory):
        def create(self,**kwargs: Any) -> Any:
            nonlocal calls
            env=super().create(**kwargs);made.append(env);reset,step=env.reset,env.step
            def observed_reset(**kw: Any) -> Any:
                value=reset(**kw);identities.append(process_identity(env.provider._session.pid));return value
            def observed_step(action: int) -> Any:
                nonlocal calls
                if time.monotonic()-started>480 or stopped('evaluation'):raise RuntimeError('Evaluation deadline/STOP')
                value=step(action);calls+=1
                with (OUT/'evaluation/actions.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(decision=calls,action=action,terminated=value[2],truncated=value[3],info=value[4]),allow_nan=False)+'\n')
                save(OUT/'evaluation/progress.json',dict(actual_decisions=calls,clock=time.monotonic()))
                if value[2] or value[3]:goals.append(actual_goal_ms(env) if value[4]['termination_reason']=='finished' else None)
                return value
            env.reset,env.step=observed_reset,observed_step
            return env
    pipeline=import_symbol(spec.components.feature_pipeline.class_path)(**spec.components.feature_pipeline.kwargs)
    evaluator=AutonomousMacroEvaluator(EvaluatorRuntimeRequest(spec.evaluation.model_copy(update={'trials_per_map':5}),
        ObservedFactory(config('evaluation')),pipeline,max_episode_steps=3000,run_dir=OUT/'evaluation'))
    evaluator.set_checkpoint(checkpoint)
    report:dict[str,Any]={'status':'FAILED','expert_access_attempts_blocked':blocked,'learner_updates':0}
    try:
        report['metrics']=evaluator.evaluate(policy)
        if len(goals)!=5 or any(type(g) is not int or g>target for g in goals) or len(set(goals))!=1:
            raise RuntimeError(f'Autonomous actual Goal qualification failed: {goals}')
        if blocked or sha(checkpoint)!=checkpoint_sha or any(not torch.equal(v,model.state_dict()[k]) for k,v in frozen.items()):
            raise RuntimeError('Teacher access or policy mutation during evaluation')
        report.update(status='PASS_FIVE_AUTONOMOUS_ACTUAL_NATIVE_GOALS',actual_goals_ms=goals,
                      policy_sha256=checkpoint_sha,teacher_files_opened=0,weights_unchanged=True)
    finally:
        for env in made:env.close()
        report['remaining_native']=[p for p in identities if same_identity(p,process_identity(p['pid']))]
        save(OUT/'evaluation/result.json',report)
    if report['remaining_native']:raise RuntimeError('Evaluation native closure')

def main() -> None:
    if len(sys.argv)>2:
        {'collect':collect,'train':train,'evaluate':evaluate}[sys.argv[2]]();return
    import numpy as np
    import yaml
    from trackmaniarl.experiments.full_qr.campaign_processes import CampaignProcessCohort
    from trackmaniarl.trackmania.native_headless._runtime.ownership import process_identity
    def terminate(signum: int,frame: Any) -> None:
        (OUT/'STOP').touch(exist_ok=True)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate)
    if OUT.exists() or PRIVATE.exists():raise RuntimeError('Never duplicate finalization')
    OUT.mkdir();PRIVATE.mkdir(mode=0o700)
    winner=read(ROOT/'best.json');ticks=read(Path(winner['motors']))
    if hashlib.sha256(bytes(ticks)).hexdigest()!=winner['motor_sha256']:raise RuntimeError('Winner input fingerprint')
    with np.load(ROOT/'protected/macro-catalog132.npz',allow_pickle=False) as z:old=[tuple(map(int,p)) for p in z['programs']]
    programs=[tuple(ticks[i:i+5]) for i in range(0,len(ticks),5)]
    catalog=sorted(set(old+programs));lookup={p:i for i,p in enumerate(catalog)}
    np.savez_compressed(OUT/'macro-catalog.npz',schema=np.asarray('tmrl.autonomous-macro-action50ms-native10ms/v1'),programs=np.asarray(catalog,dtype=np.int16))
    np.savez_compressed(OUT/'controls-TRAINING-ONLY.npz',actions=np.asarray([lookup[p] for p in programs],dtype=np.int64))
    raw=yaml.safe_load((STUDY/'autonomous-line1114/inference1114.yaml').read_text())
    raw['components']['model_factory']['kwargs']['head']['kwargs']['config']['action_count']=len(catalog)
    cfg=copy.deepcopy(raw['components']['environment']['kwargs']['config'])
    cfg.pop('training_demonstration',None);cfg.pop('reference_controls',None)
    cfg['macro_catalog']=dict(path=str(OUT/'macro-catalog.npz'),sha256=sha(OUT/'macro-catalog.npz'))
    cfg['event_source']['kwargs'].update(motor_reference_path=str(STUDY.parent/'motor-reference1079/reference.json'),
        motor_reference_sha256='6e0d5e9171eda93a8d9fa0ad47381f96eb86850616d405b66694b7b482d44bb4')
    raw['components']['environment']['kwargs']['config']=cfg
    (OUT/'model-spec.yaml').write_text(yaml.safe_dump(raw,sort_keys=False))
    save(OUT/'environment.json',cfg)
    save(OUT/'plan.json',dict(target_goal_ms=winner['actual_native_goal_ms'],motor_sha256=winner['motor_sha256'],
                             action_count=len(catalog),source_policy=sha(ROOT/'protected/policy17.712.pt')))
    for role,limit in [('collect',240),('train',540),('evaluate',540)]:
        folder=OUT/({'collect':'collection','train':'training','evaluate':'evaluation'}[role]);folder.mkdir()
        cohort=CampaignProcessCohort(PRIVATE/role,lambda:(folder/'STOP').touch(exist_ok=True))
        report:dict[str,Any]={'status':'RUNNING','parent':process_identity(os.getpid())};start=time.monotonic()
        try:
            report['identities']=cohort.start(role,[sys.executable,'-B',str(Path(__file__)),str(ROOT),role],cwd=SOURCE)
            save(folder/'launch.json',report)
            while not cohort.exited():
                if time.monotonic()-start>limit:raise RuntimeError('Bounded finalization role timeout')
                time.sleep(.5)
            report['exits']=cohort.exited()
            if report['exits']!={role:0}:raise RuntimeError(f'{role} failed')
            report['status']='PASS_ROLE'
        finally:
            report['closure']=cohort.close(grace_seconds=10);save(folder/'supervision.json',report)
        if any(report['closure'][k] for k in ('remainingOwned','cleanupErrors','forced')):raise RuntimeError('Finalization role exact closure')
    evaluation=read(OUT/'evaluation/result.json')
    save(ROOT/'finalization.json',dict(status='AUTONOMOUS_MODEL_QUALIFIED_VISUAL_GHOST_EXPORT_PENDING',
        best_autonomous_goal_ms=evaluation['actual_goals_ms'][0],policy=str(OUT/'training/policy-only.pt'),
        policy_sha256=sha(OUT/'training/policy-only.pt'),actual_goals_ms=evaluation['actual_goals_ms'],
        input_only_ghost=winner['input_only_ghost'],original_native_ghost_retained=str(ROOT/'protected/native17.712.Ghost.Gbx'),
        visual_ghost_pending=True))

if __name__=='__main__':main()
