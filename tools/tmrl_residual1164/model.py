"""Autonomous neural driving plus bounded residual PPO; no trajectory lookup."""
import torch
from torch import nn
from torch.distributions import Normal
OBS_DIM=22
FEATURE_DIM=44

def decode(classes):
    steer,longitudinal=classes.unbind(-1)
    if ((steer<0)|(steer>254)|(longitudinal<0)|(longitudinal>3)).any():raise ValueError('Native motor class domain')
    return torch.stack(((steer.float()-127)/127,(longitudinal//2).float(),(longitudinal%2).float()),-1)

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('frequencies',torch.tensor([.5,1,2,4,8,16,32,64,128,256,512]))
        self.encoder=nn.Sequential(nn.Linear(FEATURE_DIM,512),nn.SiLU(),nn.Linear(512,512),nn.SiLU(),nn.Linear(512,256),nn.SiLU())
        self.steering=nn.Linear(256,255);self.longitudinal=nn.Linear(256,4)
        self.residual=nn.Sequential(nn.Linear(256,64),nn.Tanh(),nn.Linear(64,1))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
        self.critic=nn.Sequential(nn.Linear(FEATURE_DIM,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU(),nn.Linear(128,1))
        self.register_buffer('residual_std',torch.tensor(.1))

    def features(self,x):
        phase=x[...,:1]*30*2*torch.pi*self.frequencies
        return torch.cat((x,phase.sin(),phase.cos()),-1)

    def base_logits(self,x):
        encoded=self.encoder(self.features(x))
        return self.steering(encoded),self.longitudinal(encoded)

    def residual_distribution(self,x):
        # PPO cannot move the pretrained classifier through critic gradients.
        encoded=self.encoder(self.features(x)).detach()
        mean=.2*torch.tanh(self.residual(encoded).squeeze(-1))
        return Normal(mean,self.residual_std),self.critic(self.features(x)).squeeze(-1)

    def act(self,x,*,stochastic):
        steering,longitudinal=self.base_logits(x)
        classes=torch.stack((steering.argmax(-1),longitudinal.argmax(-1)),-1)
        distribution,value=self.residual_distribution(x)
        latent=distribution.sample() if stochastic else distribution.mean
        control=decode(classes)
        # Latent Normal action and this fixed bounded motor transform define
        # the policy. Log probabilities refer to the latent sampled action.
        gate=(torch.round(x[...,0]*30000)>=16500).float()
        control=control.clone();control[...,0]=(control[...,0]+gate*.01*latent.tanh()).clamp(-1,1)
        return control,latent,distribution.log_prob(latent),value

    def freeze_base(self):
        for module in (self.encoder,self.steering,self.longitudinal):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
                parameter.grad=None

def advantages(rewards,values,terminals,gamma=.999,lam=.995):
    if len(values)!=len(rewards)+1:raise ValueError('Missing value bootstrap')
    result=torch.zeros_like(rewards);last=0.
    for i in range(len(rewards)-1,-1,-1):
        alive=1.-terminals[i];delta=rewards[i]+gamma*values[i+1]*alive-values[i]
        last=delta+gamma*lam*alive*last;result[i]=last
    return result,result+values[:-1]
