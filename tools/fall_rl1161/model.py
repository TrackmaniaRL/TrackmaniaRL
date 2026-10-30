"""Causal physical-state policy. No reference trajectory or action tape in forward."""
import torch
from torch import nn
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('frequencies',torch.tensor([.5,1,2,4,8,16,32,64,128,256,512]))
        self.encoder=nn.Sequential(nn.Linear(39,512),nn.SiLU(),nn.Linear(512,512),nn.SiLU(),nn.Linear(512,256),nn.SiLU())
        self.steering=nn.Linear(256,255);self.brake=nn.Linear(256,2);self.value=nn.Linear(256,1)

    def forward(self,x):
        phase=x[...,:1]*30*2*torch.pi*self.frequencies
        features=self.encoder(torch.cat((x,phase.sin(),phase.cos()),-1))
        return self.steering(features),self.brake(features),self.value(features).squeeze(-1)

    def distributions(self,x):
        steering,brake,value=self(x)
        # Fixed before the run. The same stochastic policy is frozen in final
        # evaluation; action probabilities never depend on a teacher or reward.
        s=steering.softmax(-1)*.9998+.0002/255
        b=brake.softmax(-1)*.9998+.0002/2
        return Categorical(probs=s),Categorical(probs=b),value


def controls(actions):
    if actions.shape[-1]!=2:raise ValueError('Steering class and brake class required')
    s,b=actions.unbind(-1)
    if bool(((s<0)|(s>254)|(b<0)|(b>1)).any()):raise ValueError('Policy action outside motor domain')
    return torch.stack(((s.float()-127)/127,torch.ones_like(s,dtype=torch.float32),b.float()),-1)


def advantages(rewards,values,terminals,gamma=.999,lam=.95):
    if len(values)!=len(rewards)+1 or len(terminals)!=len(rewards):raise ValueError('Bootstrap and terminal shapes differ')
    result=torch.zeros_like(rewards);last=0.
    for i in range(len(rewards)-1,-1,-1):
        alive=1.-terminals[i]
        delta=rewards[i]+gamma*values[i+1]*alive-values[i]
        last=delta+gamma*lam*alive*last;result[i]=last
    return result,result+values[:-1]
