from pathlib import Path
import hashlib,json,shutil,sys,time
import numpy as np
import torch
from torch.nn import functional as F
from model import Policy,decode,advantages
from atomic_json import save
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def checkpoint(path,model,bc_updates,rl_updates,extra):
    weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if not all(torch.isfinite(v).all() for v in weights.values()):raise RuntimeError('Nonfinite checkpoint')
    path=Path(path);stage=path.with_suffix('.stage');torch.save(dict(schema='tmrl-causal22-rotation-neural-base-residual-PPO/v2',state=weights,BC_updates=bc_updates,RL_updates=rl_updates,extra=extra),stage);stage.replace(path)

def fit(out,device):
    demo=np.load(out/'qualification-source/actual-rollout.npz',allow_pickle=False)
    x=torch.from_numpy(demo['observations'][:-1]).float().to(device);controls=torch.from_numpy(demo['controls']).float().to(device)
    labels=torch.stack(((controls[:,0]*127).round().long()+127,controls[:,1].long()*2+controls[:,2].long()),-1)
    if not torch.equal(decode(labels),controls):raise RuntimeError('Actual demonstrated motor commands are not exactly representable')
    model=Policy().to(device);prior=0
    if (out/'initial-bc-policy.pt').exists():
        initialized=torch.load(out/'initial-bc-policy.pt',map_location=device,weights_only=True)
        if initialized['RL_updates']!=0:raise RuntimeError('BC continuation requires a pure imitation initializer')
        current=model.state_dict()
        for key,value in initialized['state'].items():
            if key not in current:raise RuntimeError('Unexpected initializer tensor '+key)
            if value.shape==current[key].shape:current[key]=value
            elif key in ('encoder.0.weight','critic.0.weight') and value.shape[-1]==39 and current[key].shape[-1]==44:
                adapted=torch.zeros_like(current[key]);adapted[:,0]=value[:,0];adapted[:,10:22]=value[:,5:17];adapted[:,22:44]=value[:,17:39];current[key]=adapted
            else:raise RuntimeError('Unsupported initializer shape '+key)
        model.load_state_dict(current,strict=True);prior=initialized['BC_updates']
        model.residual_std.fill_(.1)
        with torch.no_grad():
            s,l=model.base_logits(x);errors=int(((s.argmax(-1)!=labels[:,0])|(l.argmax(-1)!=labels[:,1])).sum());margins=[]
            for logits,target in ((s,labels[:,0]),(l,labels[:,1])):
                margins.append(float((logits.gather(1,target[:,None]).squeeze(-1)-logits.scatter(1,target[:,None],-torch.inf).max(-1).values).min()))
        if errors==0 and min(margins)>=5:
            row=dict(updates=prior,errors=0,min_margin=min(margins),continued_from_validated_BC=True);save(out/'bc-history.json',[row]);checkpoint(out/'bc-policy.pt',model,prior,0,row);return model.cpu(),prior
    optimizer=torch.optim.Adam(list(model.encoder.parameters())+list(model.steering.parameters())+list(model.longitudinal.parameters()),lr=3e-4)
    weights=[]
    for col,n in ((0,255),(1,4)):
        count=torch.bincount(labels[:,col],minlength=n);w=torch.zeros(n,device=device);present=count>0;w[present]=len(labels)/(present.sum()*count[present]);weights.append(w)
    history=[];started=time.monotonic()
    for update in range(1,15001):
        if (out/'STOP').exists() or time.monotonic()-started>1200:break
        s,l=model.base_logits(x);loss=F.cross_entropy(s,labels[:,0],weight=weights[0])+F.cross_entropy(l,labels[:,1],weight=weights[1])
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);optimizer.step()
        if update%25:continue
        with torch.no_grad():
            s,l=model.base_logits(x);errors=int(((s.argmax(-1)!=labels[:,0])|(l.argmax(-1)!=labels[:,1])).sum());margins=[]
            for logits,target in ((s,labels[:,0]),(l,labels[:,1])):
                chosen=logits.gather(1,target[:,None]).squeeze(-1);other=logits.scatter(1,target[:,None],-torch.inf).max(-1).values;margins.append(float((chosen-other).min()))
        row=dict(updates=prior+update,errors=errors,min_margin=min(margins),loss=float(loss.detach()));history.append(row);save(out/'bc-history.json',history);save(out/'progress.json',dict(phase='IMITATION_INITIALIZATION',**row,RL_updates=0,monotonic=time.monotonic()))
        if errors==0:checkpoint(out/'bc-policy.pt',model,prior+update,0,row)
        if errors==0 and min(margins)>=5:break
    if not history or history[-1]['errors'] or history[-1]['min_margin']<2:raise RuntimeError('Exact neural imitation gate failed; no unqualified PPO launch')
    checkpoint(out/'bc-policy.pt',model,prior+update,0,history[-1]);return model.cpu(),prior+update

def episode(model,cfg,out,number,*,stochastic,training=False):
    from environment import FallEnvironment
    folder=out/f'episode-{number:06d}';env=FallEnvironment(cfg,folder);xs=[];zs=[];logs=[];values=[];rewards=[];terminals=[];info=None
    try:
        obs=env.reset()
        for _ in range(3002):
            if (out/'STOP').exists():raise RuntimeError('Owned experiment STOP')
            x=torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():control,z,logp,value=model.act(x,stochastic=stochastic)
            obs,reward,done,truncated,info=env.step(control[0].tolist())
            xs.append(x[0]);zs.append(z[0]);logs.append(logp[0]);values.append(value[0]);rewards.append(reward);terminals.append(float(done or truncated))
            if done or truncated:break
        else:raise RuntimeError('Native30s episode cap')
    finally:env.close()
    closure=json.loads((env.settings.trace_dir/'native-supervisor.json').read_text())
    if closure.get('returncode')!=0 or closure.get('remainingOwned')!=[] or closure.get('cleanupErrors')!=[]:raise RuntimeError('Native episode closure failed')
    row=dict(episode=number,actual_goal_ms=info['actual_goal_ms'],finished=info['finished'],steps=len(zs),stochastic=stochastic,closure=closure)
    save(folder/'summary.json',row)
    if type(row['actual_goal_ms']) is not int or row['actual_goal_ms']>17600:
        for target,parent in ((env.settings.trace_dir,Path(cfg['private'])/'traces'),(env.settings.work_dir,Path(cfg['runtime_root'])/'build/server-probe')):
            if target.resolve()!=target or not target.is_relative_to(parent.resolve()) or target.name!=folder.name:raise RuntimeError('Closed own trace path escaped namespace')
            if target.exists():shutil.rmtree(target)
    if not training:return row,None
    values.append(torch.tensor(0.));adv,returns=advantages(torch.tensor(rewards),torch.stack(values),torch.tensor(terminals))
    return row,dict(x=torch.stack(xs),z=torch.stack(zs),logp=torch.stack(logs),advantage=adv,returns=returns)

def ppo_update(model,optimizers,data,device):
    actor_optimizer,critic_optimizer=optimizers
    advantage=(data['advantage']-data['advantage'].mean())/data['advantage'].std().clamp_min(1e-6);metrics=[]
    for _ in range(2):
        for ids in torch.randperm(len(data['z'])).split(256):
            dist,value=model.residual_distribution(data['x'][ids].to(device));logp=dist.log_prob(data['z'][ids].to(device));ratio=(logp-data['logp'][ids].to(device)).exp();a=advantage[ids].to(device)
            active=(torch.round(data['x'][ids,0]*30000)>=16500).to(device)
            actor=-(torch.minimum(ratio*a,ratio.clamp(.95,1.05)*a)*active).sum()/active.sum().clamp_min(1);critic=F.mse_loss(value,data['returns'][ids].to(device))
            if not torch.isfinite(actor+critic):raise RuntimeError('Nonfinite actual PPO loss')
            if active.any():
                actor_optimizer.zero_grad(set_to_none=True);actor.backward();torch.nn.utils.clip_grad_norm_(model.residual.parameters(),.5,error_if_nonfinite=True);actor_optimizer.step()
            critic_optimizer.zero_grad(set_to_none=True);critic.backward();torch.nn.utils.clip_grad_norm_(model.critic.parameters(),1,error_if_nonfinite=True);critic_optimizer.step();metrics.append(dict(actor=float(actor.detach()),critic=float(critic.detach())))
    return metrics

def main(out):
    cfg=json.loads((out/'environment.json').read_text());torch.set_num_threads(2);torch.manual_seed(1164)
    if not torch.cuda.is_available():raise RuntimeError('Explicit CUDA training required')
    device=torch.device('cuda');model,bc=fit(out,device);number=0;baseline=[]
    for _ in range(3):
        row,_=episode(model,cfg,out,number,stochastic=False);number+=1;baseline.append(row)
    save(out/'bc-native-evaluation.json',dict(trials=baseline))
    if any(r['actual_goal_ms']!=17601 for r in baseline):raise RuntimeError('Greedy neural model did not reproduce17601 in3 fresh native trials')
    model.freeze_base();base_before={k:v.clone() for k,v in model.state_dict().items() if k.startswith(('encoder.','steering.','longitudinal.'))};residual_before={k:v.clone() for k,v in model.residual.state_dict().items()}
    optimizer=(torch.optim.Adam(model.residual.parameters(),lr=1e-6),torch.optim.Adam(model.critic.parameters(),lr=1e-4));updates=0;started=time.monotonic();history=[];best=None
    while time.monotonic()-started<14400 and not (out/'STOP').exists():
        torch.manual_seed(1164000+number);row,data=episode(model,cfg,out,number,stochastic=True,training=True);number+=1
        if updates>0 and type(row['actual_goal_ms']) is int and (best is None or row['actual_goal_ms']<best):
            best=row['actual_goal_ms'];checkpoint(out/'best-RL-policy.pt',model,bc,updates,row)
        if updates>0 and type(row['actual_goal_ms']) is int and row['actual_goal_ms']<=17600:
            checkpoint(out/'candidate-RL-policy.pt',model,bc,updates,row);save(out/'candidate.json',dict(**row,RL_updates=updates,confirmation_pending=True))
        model.to(device);metrics=ppo_update(model,optimizer,data,device);updates+=len(metrics);model.cpu();history.append(dict(**row,RL_updates=updates,last_update=metrics[-1]));save(out/'rl-history.json',history);checkpoint(out/'latest-RL-policy.pt',model,bc,updates,history[-1]);save(out/'progress.json',dict(phase='ACTUAL_RESIDUAL_PPO',RL_updates=updates,episodes=len(history),last_goal_ms=row['actual_goal_ms'],monotonic=time.monotonic()))
        if (out/'candidate-RL-policy.pt').exists():break
        if len(history)>=10 and all(not r['finished'] for r in history[-10:]):break
    if updates<=0 or not any(not torch.equal(residual_before[k],v) for k,v in model.residual.state_dict().items()):raise RuntimeError('No actual RL actor parameter update')
    if any(not torch.equal(v,model.state_dict()[k]) for k,v in base_before.items()):raise RuntimeError('Frozen pretrained base changed')
    selected=out/'candidate-RL-policy.pt' if (out/'candidate-RL-policy.pt').exists() else out/'best-RL-policy.pt' if (out/'best-RL-policy.pt').exists() else out/'latest-RL-policy.pt';weights=torch.load(selected,map_location='cpu',weights_only=True);model.load_state_dict(weights['state']);model.eval();checksum=sha(selected);frozen={k:v.clone() for k,v in model.state_dict().items()};trials=[];eval_started=time.monotonic()
    if weights['RL_updates']<=0 or not any(not torch.equal(residual_before[k],v) for k,v in model.residual.state_dict().items()):raise RuntimeError('Selected actor has no actual PPO parameter change')
    def block_teacher(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        mode=args[1] if len(args)>1 else None
        if isinstance(mode,str) and mode.startswith(('w','a','x')):return
        if any(t in str(args[0]).lower() for t in ('source-controls','actual-rollout.npz','training-only','/winners/')):raise RuntimeError('Frozen model attempted teacher read')
    sys.addaudithook(block_teacher)
    for trial in range(100):
        if time.monotonic()-eval_started>3600:raise RuntimeError('Bounded100-trial native evaluation expired')
        torch.manual_seed(11641000+trial);row,_=episode(model,cfg,out,number,stochastic=trial!=0);number+=1;trials.append(row);save(out/'final-evaluation-progress.json',dict(trials=trials,policy_sha256=checksum))
    if sha(selected)!=checksum or any(not torch.equal(frozen[k],v) for k,v in model.state_dict().items()):raise RuntimeError('Frozen evaluation weights changed')
    winners=[r for r in trials if type(r['actual_goal_ms']) is int and r['actual_goal_ms']<=17600]
    save(out/'training-result.json',dict(status='RL_TARGET_REACHED_EXPORT_PENDING' if winners else 'RL_COMPLETED_TARGET_NOT_REACHED',policy=str(selected),policy_sha256=checksum,BC_updates=bc,selected_checkpoint_RL_updates=weights['RL_updates'],total_run_RL_updates=updates,trials=trials,successful_trials=len(winners),total_trials=100,teacher_reads_in_evaluation=False,pretrained_base_unchanged=True,stochastic_policy_disclosed=True,public_WR_claim=False))

if __name__=='__main__':main(Path(sys.argv[1]))
