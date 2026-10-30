"""Reward-driven Double QR-DQN with separately counted imitation retention."""
from collections import deque
import random
import numpy as np
import torch
from trackmaniarl.algorithms.value_based.learner import DiscreteValueLearner
from trackmaniarl.core.data import TrainingBatch
from trackmaniarl.models.contracts import RiskSpec,ValuePhase
from tools.record_rl1168.model import build_model,ACTIONS
from tools.vision_qr1167.returns import driving_return
from trackmaniarl.algorithms.value_based.updater import ValueUpdater


class RetainingUpdater(ValueUpdater):
    """One Adam step on the sum of TD and retention losses, not two momenta steps."""
    def __init__(self,learner,demo,weight=2.):
        super().__init__(learner);self.demo=demo;self.retention_loss=0.;self.anchor_errors=0;self.weight=weight

    def _objective_loss(self,forward):
        loss=super()._objective_loss(forward)
        if self.demo is None:return loss
        x,y=self.demo;model=self.learner.model
        f=model.encode_frames(x)
        q=model.expected_all_actions(f,model.support(f,ValuePhase.EVALUATE),RiskSpec())
        correct=q.gather(1,y[:,None]).squeeze(-1)
        margin=torch.full_like(q,2.);margin.scatter_(1,y[:,None],0.)
        anchor=(q+margin).max(-1).values.sub(correct).clamp_min(0).mean()
        self.retention_loss=float(anchor.detach())
        self.anchor_errors=int((q.argmax(-1)!=y).sum().detach())
        return loss+self.weight*anchor


class Learner:
    def __init__(self,initializer,*,device='cuda',seed=1168,retention_weight=2.):
        if not np.isfinite(retention_weight) or retention_weight<0:raise ValueError('invalid retention weight')
        self.retention_weight=float(retention_weight)
        self.rng=random.Random(seed)
        self.learner=DiscreteValueLearner(build_model(),seed=seed,learning_rate=2e-6,
            target_update_interval=500,execution={'device':device,'precision':'float32'})
        self.learner.setup({'seed':seed})
        self.learner.model.load_state_dict(initializer['state'])
        self.learner.target_model.load_state_dict(initializer['state'])
        self.replay=deque(maxlen=20000);self.episode_ids=deque(maxlen=20000)
        self.episode=0;self.transitions=0;self.imitation_updates=0
        self.bc_updates=initializer['BC_updates']
        self.actor=build_model().cpu().eval();self.sync_actor()
        self.demo=None
        self.accepted_RL_updates=0;self.accepted_goal_ms=17601;self.rejected_candidates=0

    def consider_candidate(self,result,state):
        valid=(result.get('status')=='NATIVE_GOAL' and result.get('clean_closure') is True
            and result.get('full_start') is True and result.get('training') is False
            and result.get('actual_goal_ms') is not None
            and result['actual_goal_ms']<=self.accepted_goal_ms)
        if valid:
            self.actor.load_state_dict(state)
            self.accepted_goal_ms=result['actual_goal_ms']
            self.accepted_RL_updates=self.learner.update_count
        else:self.rejected_candidates+=1
        return valid

    def sync_actor(self):
        self.actor.load_state_dict({k:v.detach().cpu() for k,v in self.learner.model.state_dict().items()})

    @torch.no_grad()
    def action(self,state,epsilon,rng=None):
        rng=rng or self.rng
        if rng.random()<epsilon:return rng.randrange(len(ACTIONS))
        x=torch.from_numpy(np.asarray(state,dtype=np.float32)[None])
        f=self.actor.encode_frames(x)
        return int(self.actor.expected_all_actions(f,self.actor.support(f,ValuePhase.EVALUATE),RiskSpec()).argmax(-1))

    def begin_episode(self):self.episode+=1

    def add(self,state,action,reward,following,terminal):
        self.replay.append((state.copy(),action,float(reward),following.copy(),bool(terminal),False))
        self.episode_ids.append(self.episode);self.transitions+=1

    def configure_demonstrations(self,path):
        data=np.load(path,allow_pickle=False)
        self.demo=(torch.from_numpy(data['observations']).to(self.learner.device),
                   torch.from_numpy(data['actions']).long().to(self.learner.device))

    def update(self,batch_size=128):
        if len(self.replay)<batch_size:return None
        rows=list(self.replay);ids=list(self.episode_ids)
        chosen=self.rng.sample(range(len(rows)),batch_size)
        samples=[driving_return(rows,ids,index,horizon=50,gamma=1.) for index in chosen]
        obs,act,rew,nxt,done,discounts,horizons=zip(*samples)
        batch=TrainingBatch(data={},observations=torch.from_numpy(np.stack(obs)),
            actions=torch.tensor(act),rewards=torch.tensor(rew,dtype=torch.float32),
            next_observations=torch.from_numpy(np.stack(nxt)),terminated=torch.tensor(done,dtype=torch.bool),
            truncated=torch.zeros(batch_size,dtype=torch.bool),bootstrap_discounts=torch.tensor(discounts,dtype=torch.float32),
            transition_ids=chosen,metadata={'gamma':1.,'return_lengths':list(horizons),'return_horizon_max':50})
        updater=RetainingUpdater(self.learner,self.demo if self.retention_weight else None,self.retention_weight)
        metrics,_=updater.update(batch)
        if self.demo is not None and self.retention_weight:self.imitation_updates+=1
        metrics=dict(metrics,auxiliary_margin=updater.retention_loss,auxiliary_weight=self.retention_weight,optimizer_steps=1,
                     anchor_errors_before_update=updater.anchor_errors)
        if not all(torch.isfinite(p).all() for p in self.learner.model.parameters()):raise ValueError('nonfinite RL weights')
        return dict(metrics)

    def checkpoint(self):
        return dict(schema='full-control-native-QR1168/retention-v2',state_role='unqualified_learning_candidate',learner=self.learner.state_dict(),
            replay=list(self.replay),episode_ids=list(self.episode_ids),episode=self.episode,
            retention_weight=self.retention_weight,transitions=self.transitions,RL_updates=self.learner.update_count,
            BC_initialization_updates=self.bc_updates,auxiliary_imitation_updates=self.imitation_updates,
            rng=self.rng.getstate(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            accepted_actor_state={k:v.detach().cpu().clone() for k,v in self.actor.state_dict().items()},
            accepted_RL_updates=self.accepted_RL_updates,accepted_goal_ms=self.accepted_goal_ms,
            rejected_candidates=self.rejected_candidates,
            state={k:v.detach().cpu() for k,v in self.learner.model.state_dict().items()})
