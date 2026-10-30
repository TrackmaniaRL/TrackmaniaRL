"""Fresh BC -> native qualification -> actual on-policy PPO -> frozen100 suite."""
from pathlib import Path
import copy,hashlib,json,math,os,shutil,sys,time
import numpy as np
import torch
from torch.nn import functional as F
from model import Policy,controls,advantages
from environment import FallEnvironment,sha


def save(path,value):
    path=Path(path);tmp=path.with_suffix('.stage');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def checkpoint(path,model,bc_updates,rl_updates,extra=None):
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if not all(torch.isfinite(v).all() for v in state.values()):raise RuntimeError('Nonfinite policy checkpoint')
    target=Path(path);temporary=target.with_suffix('.stage')
    torch.save(dict(schema='fall05-causal-native17-ppo/v1',state=state,bc_updates=bc_updates,RL_updates=rl_updates,
        observation='fall05-causal-native17/bootstrap2190/v1',action='steer-i8+binary-brake; gas1; one model decision every10ms',extra=extra),temporary)
    temporary.replace(target)


def fit(out,device):
    demo=np.load(out/'qualification-v6/actual-rollout.npz',allow_pickle=False)
    observations=torch.from_numpy(demo['observations'][:-1]).float().to(device)
    applied=torch.from_numpy(demo['controls']).float().to(device)
    actions=torch.stack(((applied[:,0]*127).round().long()+127,applied[:,2].long()),-1)
    if not torch.equal(controls(actions),applied):raise RuntimeError('Demonstrated controls must be exactly representable')
    model=Policy().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    started=time.monotonic();streak=0;history=[]
    weights=[]
    for column,n in ((0,255),(1,2)):
        counts=torch.bincount(actions[:,column],minlength=n);w=torch.zeros(n,device=device);present=counts>0;w[present]=len(actions)/(present.sum()*counts[present]);weights.append(w)
    for update in range(1,15001):
        if (out/'STOP').exists() or time.monotonic()-started>1200:break
        model.train();s,b,_=model(observations)
        loss=F.cross_entropy(s,actions[:,0],weight=weights[0])+F.cross_entropy(b,actions[:,1],weight=weights[1])
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);optimizer.step()
        if update%25:continue
        model.eval()
        with torch.no_grad():
            s,b,_=model(observations);errors=int(((s.argmax(-1)!=actions[:,0])|(b.argmax(-1)!=actions[:,1])).sum())
            margins=[]
            for logits,target in ((s,actions[:,0]),(b,actions[:,1])):
                selected=logits.gather(1,target[:,None]).squeeze(1);other=logits.scatter(1,target[:,None],-torch.inf).max(-1).values;margins.append(float((selected-other).min()))
        row=dict(updates=update,errors=errors,min_margin=min(margins),loss=float(loss.detach()));history.append(row);save(out/'bc-history.json',history)
        streak=streak+1 if errors==0 and min(margins)>=5 else 0
        if errors==0:checkpoint(out/'bc-policy.pt',model,update,0,row)
        if streak>=3:break
    if not history or history[-1]['errors']!=0 or history[-1]['min_margin']<1:raise RuntimeError('BC failed to fit measured native decisions; no unqualified RL launch')
    checkpoint(out/'bc-policy.pt',model,update,0,history[-1])
    save(out/'bc-result.json',dict(status='FIT_GATE_PASSED_NATIVE_MODEL_EVALUATION_PENDING',updates=update,actual_demonstration_goal_ms=21758,RL_updates=0,wall_seconds=time.monotonic()-started))
    return model,update


def episode(model,cfg,out,number,*,stochastic,training=False):
    env=FallEnvironment(cfg,out/f'episode-{number:06d}')
    observations=[];actions=[];logps=[];values=[];rewards=[];terminals=[]
    model.eval();info=None
    try:
        obs=env.reset()
        for step in range(3002):
            if (out/'STOP').exists():raise RuntimeError('Own training STOP')
            x=torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                sd,bd,value=model.distributions(x)
                action=torch.stack((sd.sample(),bd.sample()),-1) if stochastic else torch.stack((sd.probs.argmax(-1),bd.probs.argmax(-1)),-1)
                logp=sd.log_prob(action[:,0])+bd.log_prob(action[:,1]);control=controls(action)[0].tolist()
            obs,reward,terminated,truncated,info=env.step(control)
            observations.append(x[0]);actions.append(action[0]);logps.append(logp[0]);values.append(value[0]);rewards.append(reward);terminals.append(float(terminated or truncated))
            if terminated or truncated:break
        else:raise RuntimeError('Physical episode cap exceeded')
    finally:env.close()
    row=dict(episode=number,actual_goal_ms=info['actual_goal_ms'] if info else None,finished=bool(info and info['finished']),steps=len(actions),stochastic=stochastic)
    closure=json.loads((env.settings.trace_dir/'native-supervisor.json').read_text())
    if closure.get('remainingOwned')!=[] or closure.get('cleanupErrors')!=[] or closure.get('returncode')!=0:raise RuntimeError('Native episode closure failed; preserve all evidence')
    row['closure']=closure
    save(out/f'episode-{number:06d}/summary.json',row)
    # Retain complete qualification and candidate traces. Other closed training
    # episodes keep actual observations/commands and closure receipts, with only
    # this new campaign's large transient engine files removed.
    preserve=type(row['actual_goal_ms']) is int and row['actual_goal_ms']<=21757
    if not preserve:
        for target,parent in ((env.settings.trace_dir,Path(cfg['private'])/'traces'),(env.settings.work_dir,Path(cfg['runtime_root'])/'build/server-probe')):
            if target.name!=f'episode-{number:06d}' or target.resolve()!=target or not target.is_relative_to(parent.resolve()):raise RuntimeError('Own trace retention path escaped new campaign')
            if target.exists():shutil.rmtree(target)
    if not training:return row,None
    values.append(torch.tensor(0.))
    adv,returns=advantages(torch.tensor(rewards),torch.stack(values),torch.tensor(terminals))
    return row,dict(observation=torch.stack(observations),action=torch.stack(actions),old_logp=torch.stack(logps),advantage=adv,returns=returns)


def ppo_update(model,optimizer,data,device):
    count=len(data['action']);mean=data['advantage'].mean();std=data['advantage'].std().clamp_min(1e-6)
    metrics=[]
    for epoch in range(4):
        for ids in torch.randperm(count).split(256):
            x=data['observation'][ids].to(device);action=data['action'][ids].to(device);old=data['old_logp'][ids].to(device);adv=(data['advantage'][ids].to(device)-mean.to(device))/std.to(device);target=data['returns'][ids].to(device)
            s,b,value=model.distributions(x);logp=s.log_prob(action[:,0])+b.log_prob(action[:,1]);ratio=(logp-old).exp()
            actor=-torch.minimum(ratio*adv,ratio.clamp(.9,1.1)*adv).mean()
            critic=F.mse_loss(value,target);entropy=(s.entropy()+b.entropy()).mean();loss=actor+.5*critic-.001*entropy
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite actual PPO loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
            metrics.append(dict(actor=float(actor.detach()),critic=float(critic.detach()),entropy=float(entropy.detach()),gradient_norm=float(norm)))
    return metrics


def main(out):
    cfg=json.loads((out/'environment-independent.json').read_text());device=torch.device('cuda')
    if not torch.cuda.is_available():raise RuntimeError('CUDA preflight failed; do not silently change training execution')
    torch.set_num_threads(2);torch.manual_seed(1161)
    model,bc_updates=fit(out,device);model=model.cpu();number=0
    baseline=[]
    for _ in range(3):
        row,_=episode(model,cfg,out,number,stochastic=False);number+=1;baseline.append(row)
    save(out/'bc-native-evaluation.json',dict(trials=baseline))
    if any(r['actual_goal_ms']!=21758 for r in baseline):raise RuntimeError('Neural imitation did not reproduce actual21758; RL admission failed')
    # This new optimizer consumes only actual online returns and old policy
    # probabilities below. No demonstrations participate in PPO loss.
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-5);rl_updates=0;total_steps=0;started=time.monotonic();history=[];best=None
    before={k:v.clone() for k,v in model.state_dict().items()}
    while time.monotonic()-started<14400 and not (out/'STOP').exists():
        model=model.cpu();row,data=episode(model,cfg,out,number,stochastic=True,training=True);number+=1;total_steps+=row['steps']
        if rl_updates>0 and type(row['actual_goal_ms']) is int and row['actual_goal_ms']<=21757:
            checkpoint(out/'candidate-RL-policy.pt',model,bc_updates,rl_updates,row)
            save(out/'candidate.json',dict(**row,RL_updates=rl_updates,policy_sha256=sha(out/'candidate-RL-policy.pt'),confirmation_pending=True))
        model.to(device);metrics=ppo_update(model,optimizer,data,device);rl_updates+=len(metrics);model.cpu()
        history.append(dict(**row,RL_updates=rl_updates,total_online_steps=total_steps,last_update=metrics[-1]));save(out/'rl-history.json',history)
        checkpoint(out/'latest-RL-policy.pt',model,bc_updates,rl_updates,history[-1])
        save(out/'progress.json',dict(phase='ACTUAL_PPO',RL_updates=rl_updates,online_steps=total_steps,episodes=len(history),monotonic=time.monotonic(),last_goal_ms=row['actual_goal_ms']))
        if (out/'candidate-RL-policy.pt').exists():break
        if len(history)%5==0:
            evaluated,_=episode(model,cfg,out,number,stochastic=False);number+=1
            if evaluated['actual_goal_ms'] is not None and (best is None or evaluated['actual_goal_ms']<best):
                best=evaluated['actual_goal_ms'];checkpoint(out/'best-RL-policy.pt',model,bc_updates,rl_updates,evaluated)
            if best is not None and best<=21757:break
        if len(history)>=10 and all(not r['finished'] for r in history[-10:]):break
    if rl_updates<=0 or not any(not torch.equal(before[k],v) for k,v in model.state_dict().items()):raise RuntimeError('No measured RL parameter update')
    selected=out/'candidate-RL-policy.pt' if (out/'candidate-RL-policy.pt').exists() else out/'best-RL-policy.pt' if (out/'best-RL-policy.pt').exists() else out/'latest-RL-policy.pt'
    weights=torch.load(selected,map_location='cpu',weights_only=True)
    if weights['RL_updates']<=0:raise RuntimeError('Only a checkpoint with actual RL updates may enter final evaluation')
    model.load_state_dict(weights['state']);model.eval();frozen={k:v.clone() for k,v in model.state_dict().items()};checksum=sha(selected)
    del optimizer,data
    def deny_teacher(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)) and any(t in str(args[0]).lower() for t in ('training-only','actual-rollout.npz','source-controls')):raise RuntimeError('Frozen policy evaluation attempted teacher access')
    # The environment writes its own new rollout evidence; reads of earlier
    # demonstrations are blocked separately from those write-only files.
    def deny_reads(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        mode=args[1] if len(args)>1 else None
        if isinstance(mode,str) and mode.startswith(('w','a','x')):return
        deny_teacher(event,args)
    sys.addaudithook(deny_reads)
    trials=[];eval_start=time.monotonic()
    for trial in range(100):
        if time.monotonic()-eval_start>3600:raise RuntimeError('Bounded final100 suite expired')
        torch.manual_seed(1161000+trial)
        row,_=episode(model,cfg,out,number,stochastic=trial!=0);number+=1;trials.append(row)
        save(out/'final-evaluation-progress.json',dict(trials=trials,policy_sha256=checksum))
    if sha(selected)!=checksum or any(not torch.equal(frozen[k],v) for k,v in model.state_dict().items()):raise RuntimeError('Frozen evaluation policy changed')
    success=[r for r in trials if type(r['actual_goal_ms']) is int and r['actual_goal_ms']<=21757]
    save(out/'training-result.json',dict(status='RL_GOAL_TARGET_REACHED_NATIVE_EXPORT_PENDING' if success else 'RL_COMPLETED_TARGET_NOT_REACHED',policy=str(selected),policy_sha256=checksum,
        BC_updates=bc_updates,RL_updates=rl_updates,online_steps=total_steps,trials=trials,successful_trials=len(success),total_trials=100,
        inference_teacher_access=False,weights_unchanged=True,public_WR_claim=False,stochastic_policy_disclosed=True))


if __name__=='__main__':main(Path(sys.argv[1]))
