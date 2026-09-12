"""Trainable CAT-conditioned motor-token policy and separate privileged critic."""
import copy
import torch
from torch import nn
from torch.distributions import Normal


class WholeBodyPolicy(nn.Module):
    def __init__(self, template, device="cpu"):
        super().__init__()
        self.student=copy.deepcopy(template).to(device)
        self.log_std=nn.Parameter(torch.full((64,),-2.5,device=device))
        self.critic=nn.Sequential(nn.Linear(1279,1024),nn.SiLU(),nn.Linear(1024,512),nn.SiLU(),
            nn.Linear(512,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU(),nn.Linear(128,1)).to(device)

    def mean(self,o):
        s=self.student
        x=torch.cat((s.features(o["cat"]),((o["obs"]-s.mean)/s.std).clamp(-10,10),o["base"],o["mode"][:,None]),-1)
        return s.adapter(x)

    def distribution(self,o):
        return Normal(self.mean(o),self.log_std.clamp(-4.,-.5).exp())

    def value(self,o):
        normalized=((o["obs"]-self.student.mean)/self.student.std).clamp(-10,10)
        return self.critic(torch.cat((normalized,o["privileged"]),-1)).squeeze(-1)

    def targets(self,o,z):
        s=self.student
        raw=s.decoder(torch.cat((o["base"]+2*torch.tanh(z),o["obs"]),-1))
        return raw.clamp(-s.contract["clip"],s.contract["clip"])*s.scale+s.offset

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]


def learner_state(policy):
    """Only trainable weights/stats; original frozen release stays hash-addressed."""
    return dict(adapter=policy.student.adapter.state_dict(),critic=policy.critic.state_dict(),
        log_std=policy.log_std.detach().clone(),mean=policy.student.mean.clone(),std=policy.student.std.clone())


def restore_learner(policy,state):
    policy.student.adapter.load_state_dict(state["adapter"],strict=True)
    policy.critic.load_state_dict(state["critic"],strict=True)
    with torch.no_grad():
        policy.log_std.copy_(state["log_std"])
        policy.student.mean.copy_(state["mean"]);policy.student.std.copy_(state["std"])
