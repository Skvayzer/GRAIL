#!/usr/bin/env python3
"""CPU arithmetic check through the actual frozen GRAIL decoder, not navigation training."""
import argparse
import copy
import json
from pathlib import Path
import torch
from torch import nn
from torch.distributions import Normal
from cat_distill import load
from cat_parallel_policy import WholeBodyPolicy,learner_state,restore_learner
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("output",type=Path);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(913)
    root=Path(__file__).resolve().parent
    prototype,data,_=load(SimpleNamespace(run=root/"runs/20260912_cat_distill_flat_v4"))
    prototype.adapter=nn.Sequential(nn.Linear(1158,512),nn.SiLU(),nn.Linear(512,256),nn.SiLU(),
        nn.Linear(256,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,64))
    nn.init.zeros_(prototype.adapter[-1].weight);nn.init.zeros_(prototype.adapter[-1].bias)
    model=WholeBodyPolicy(prototype)
    frozen=model.student.frozen_hash()
    ids=torch.where((data["mode"]==1)&~data["validation"])[0][:64]
    obs={k:data[k][ids] for k in ("obs","cat","base","mode")}
    # Critic-only arithmetic input, explicitly synthetic; not used as a rollout.
    obs["privileged"]=torch.randn(len(ids),250)*.1
    target=data["target"][ids]
    opt=torch.optim.Adam(model.trainable(),lr=3e-4)
    reports=[]
    for kind in ("transfer","dagger","ppo"):
        with torch.no_grad():
            old=model.distribution(obs);z=old.sample();old_logp=old.log_prob(z).sum(-1)
            teacher=Normal(old.loc+.01,old.scale.clone())
        opt.zero_grad()
        dist=model.distribution(obs)
        prediction=model.targets(obs,dist.loc)
        if kind=="transfer": loss=(prediction[:,model.student.legs]-target[:,model.student.legs]).square().mean()
        elif kind=="dagger": loss=torch.distributions.kl_divergence(teacher,dist).sum(-1).mean()
        else:
            ratio=(dist.log_prob(z).sum(-1)-old_logp).exp()
            torch.testing.assert_close(ratio,torch.ones_like(ratio),atol=1e-6,rtol=0)
            advantage=torch.linspace(-1,1,len(ids))
            loss=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()-.003*dist.entropy().sum(-1).mean()
        loss+=.5*model.value(obs).square().mean()
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.trainable(),1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise ValueError("Nonfinite gradient")
        opt.step()
        reports.append(dict(stage=kind,loss=float(loss.detach()),gradient_norm=float(norm),joint_outputs=prediction.shape[-1]))
    def update():
        opt.zero_grad();loss=model.targets(obs,model.mean(obs)).square().mean()+model.value(obs).square().mean()
        loss.backward();opt.step()
        return torch.cat([p.detach().flatten().clone() for p in model.trainable()])
    state=copy.deepcopy(learner_state(model));optim=copy.deepcopy(opt.state_dict())
    next1=update();restore_learner(model,state);opt.load_state_dict(copy.deepcopy(optim));next2=update()
    torch.testing.assert_close(next1,next2,atol=0,rtol=0)
    if model.student.frozen_hash()!=frozen:raise ValueError("Frozen model changed")
    result=dict(stages=reports,bit_exact_learner_resume=True,frozen_unchanged=True,device="cpu",
        physics_rollout=False,navigation_training=False,robot_actuation=False,
        evidence="original GRAIL decoder, CPU arithmetic only; generated Isaac optimizer smoke still required")
    with a.output.open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
