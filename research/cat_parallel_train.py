#!/usr/bin/env python3
"""CAT-generated multi-environment whole-body training in Isaac GPU PhysX.

Stages: CAT-teacher transfer -> family-specialist PPO -> specialist/generalist
DAgger -> generalist PPO. Pinned CAT rollout/reward/PPO settings, explicit
whole-body architectural adaptations, simulation-only. No SDK/ROS imports.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def memory_fraction(limit_gib,total_bytes):
    """Torch allocator cap only; PhysX/Kit need additional GPU memory headroom."""
    if not math.isfinite(limit_gib) or not 0<limit_gib<total_bytes/2**30:
        raise ValueError("Torch memory limit must be positive and below GPU capacity")
    return limit_gib*2**30/total_bytes


def resume_phase_progress(checkpoint,source_config,phase_names):
    """Preserve transitions, not old rollout indices, when resizing vector envs."""
    if [tuple(p[:2]) for p in source_config["phases"]]!=list(phase_names):
        raise ValueError("Resume phase sequence differs from source run")
    if "phase_transitions" in checkpoint:
        progress=list(checkpoint["phase_transitions"])
        budgets=list(checkpoint["phase_budget_transitions"])
    else:
        chunk=source_config["args"]["num_envs"]*source_config["native_recipe"]["policy_config"]["unroll_length"]
        budgets=[p[2]*chunk for p in source_config["phases"]]
        phase=checkpoint["phase"]
        if not 0<=phase<=len(budgets): raise ValueError("Invalid legacy resume phase")
        progress=[budget if i<phase else 0 for i,budget in enumerate(budgets)]
        if phase<len(progress): progress[phase]=checkpoint["next_update"]*chunk
    if (len(progress)!=len(phase_names) or len(budgets)!=len(phase_names)
            or any(type(v) is not int or v<0 for v in progress)
            or any(type(v) is not int or v<=0 for v in budgets)
            or sum(progress)!=checkpoint["total_steps"]):
        raise ValueError("Resume transition accounting mismatch")
    return progress,budgets


def gpu_robot_deployments():
    """Read only: identify potentially timing-critical robot users of this GPU."""
    result=subprocess.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],
        text=True,capture_output=True,check=True)
    found=[]
    for line in result.stdout.splitlines():
        if not line.strip().isdigit():continue
        pid=int(line.strip())
        try: command=Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0",b" ").decode()
        except (FileNotFoundError,PermissionError):continue
        if "g1_deploy" in command or "robot_deploy" in command:
            found.append(dict(pid=pid,command=command))
    return found


def process_gpu_memory_gib():
    """NVIDIA-reported process use, including non-Torch simulator allocations."""
    result=subprocess.run(["nvidia-smi","--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits"],text=True,capture_output=True,check=True)
    for line in result.stdout.splitlines():
        pid,used=(part.strip() for part in line.split(",",1))
        if pid==str(os.getpid()): return float(used)/1024
    raise RuntimeError("Cannot read training process GPU memory")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("--bank",type=Path,default=ROOT/"runs/20260912_cat_generated_bank_v2")
    p.add_argument("--context",type=Path,default=ROOT/"runs/20260912T095135_246506Z_stair_p1_cat_audit")
    p.add_argument("--num-envs",type=int,default=2048,help="16/64/128 or a multiple of 256 up to 32768")
    p.add_argument("--benchmark",action="store_true")
    p.add_argument("--benchmark-steps",type=int,default=128)
    p.add_argument("--smoke-updates",type=int,default=0)
    p.add_argument("--transfer-updates",type=int,default=None)
    p.add_argument("--specialist-ppo-updates",type=int,default=None)
    p.add_argument("--dagger-updates",type=int,default=None)
    p.add_argument("--generalist-ppo-updates",type=int,default=None)
    p.add_argument("--hours",type=float,default=24.)
    p.add_argument("--torch-memory-limit-gib",type=float,default=14.,
        help="Torch allocator ceiling; not a total-process limit (Kit/PhysX allocate separately)")
    p.add_argument("--min-free-gpu-gib",type=float,default=2.,
        help="Checkpointed stop below this device headroom; minimum supported reserve is 1 GiB")
    p.add_argument("--wandb-mode",choices=("online","offline","disabled"),default="online")
    p.add_argument("--resume",type=Path)
    p.add_argument("--seed",type=int,default=20260912)
    p.add_argument("--no-randomization",action="store_true",help="Benchmark diagnostics only")
    p.add_argument("--accept-isaac-eula",action="store_true")
    p.add_argument("--detach",action="store_true")
    p.add_argument("--reviewed-deployment-pid",type=int,action="append",default=[],
        help="Only after operator confirms this exact GPU deployment PID is not controlling hardware")
    p.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.num_envs not in (16,64,128) and (not 256<=a.num_envs<=32768 or a.num_envs%256):
        p.error("Environment count must be 16/64/128 or a multiple of 256 up to 32768")
    if not a.accept_isaac_eula or not 0<a.hours<=48 or a.smoke_updates<0 or not math.isfinite(a.torch_memory_limit_gib) or a.torch_memory_limit_gib<=0:
        p.error("Explicit EULA acceptance and bounded runtime required")
    if not math.isfinite(a.min_free_gpu_gib) or not 1.<=a.min_free_gpu_gib<=16.:
        p.error("Device free-memory reserve must be 1..16 GiB")
    if a.no_randomization and not a.benchmark:
        p.error("Randomization cannot be disabled for distributional training")
    a.run=a.run.resolve(); a.bank=a.bank.resolve(); a.context=a.context.resolve()
    if a.resume: a.resume=a.resume.resolve()
    deployments=gpu_robot_deployments()
    if any(d["pid"] not in a.reviewed_deployment_pid for d in deployments):
        p.error("Potential real-robot GPU deployment detected; operator review required: "+str([d["pid"] for d in deployments]))
    if not a.worker:
        a.run.mkdir(parents=True,exist_ok=False)
        command=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],"--worker"]
        command=[c for c in command if c!="--detach"]
        with (a.run/"worker.log").open("x") as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (a.run/"process.json").write_text(json.dumps(dict(pid=child.pid,command=command,
            created=time.time(),robot_actuation=False),indent=2)+"\n")
        print(json.dumps(dict(pid=child.pid,run=str(a.run),log=str(a.run/"worker.log"))),flush=True)
        if a.detach:
            return
        try:
            code=child.wait(timeout=min(a.hours*3600+180,900) if a.benchmark else a.hours*3600+180)
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL); child.wait()
            raise
        if code or not (a.run/"complete.json").exists():
            raise SystemExit("Worker failed; inspect "+str(a.run/"worker.log"))
        return
    os.environ["OMNI_KIT_ACCEPT_EULA"]="Yes"
    os.environ["WANDB_MODE"]=a.wandb_mode
    os.environ["OMP_NUM_THREADS"]="4"
    os.environ["OPENBLAS_NUM_THREADS"]="4"
    os.environ["TMPDIR"]=str(a.run/"tmp")
    (a.run/"tmp").mkdir()
    import tempfile
    tempfile.tempdir=os.environ["TMPDIR"]
    sys.argv=sys.argv[:1]
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device="cuda:0",livestream=0).app
    try:
        train(a)
        (a.run/"complete.json").write_text(json.dumps(dict(complete=True,time=time.time(),robot_actuation=False))+"\n")
    except BaseException as error:
        import traceback
        traceback.print_exc()
        (a.run/"error.json").write_text(json.dumps(dict(error=repr(error),time=time.time()))+"\n")
        raise
    finally:
        app.close(skip_cleanup=True)


def train(a):
    import numpy as np
    import torch
    from torch import nn
    from torch.distributions import Normal
    from cat_distill_model import load_grail,DistillStudent,sha
    from cat_direct_native import teacher
    from cat_parallel_env import ParallelCatEnv
    from cat_parallel_policy import WholeBodyPolicy,learner_state,restore_learner
    from gear_sonic.research.residual_learning import generalized_advantage
    torch.set_num_threads(4); torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    device="cuda:0"
    free_bytes,total_bytes=torch.cuda.mem_get_info()
    fraction=memory_fraction(a.torch_memory_limit_gib,total_bytes)
    if free_bytes<4*2**30:
        raise ValueError("Less than 4 GiB GPU headroom at learner startup; refusing new training")
    torch.cuda.set_per_process_memory_fraction(fraction,device)
    print("GPU_BUDGET",json.dumps(dict(torch_limit_gib=a.torch_memory_limit_gib,
        free_gib=free_bytes/2**30,total_gib=total_bytes/2**30,
        scope="Torch allocator only; Kit/PhysX memory separately monitored")),flush=True)
    actor,contract=load_grail(a.context)
    resume_checkpoint=torch.load(a.resume,map_location="cpu",weights_only=False) if a.resume else None
    expert=teacher().to(device)
    prototype=DistillStudent(actor,contract,expert.cpu()).to(device)
    # CAT-sized hidden stack for distributional transfer; not the small pilot MLP.
    prototype.adapter=nn.Sequential(nn.Linear(1158,512),nn.SiLU(),nn.Linear(512,256),nn.SiLU(),
        nn.Linear(256,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,64)).to(device)
    nn.init.zeros_(prototype.adapter[-1].weight);nn.init.zeros_(prototype.adapter[-1].bias)
    expert=expert.to(device); del actor
    env=ParallelCatEnv(a.run,a.bank,contract,a.num_envs,device,a.seed,not a.no_randomization)
    obs=env.observe()
    if a.benchmark:
        started=time.monotonic(); visits=[]
        with torch.no_grad():
            for i in range(a.benchmark_steps):
                target=env.teacher_targets(expert,obs)
                obs,reward,term,trunc,info=env.step(target)
                if not torch.isfinite(obs["cat"]).all() or not torch.isfinite(obs["obs"]).all():
                    raise ValueError("Nonfinite batch observation")
                if i%32==0:
                    print("BENCH",i, float(reward.mean()),len(env.completed),flush=True)
        torch.cuda.synchronize()
        result=dict(num_envs=a.num_envs,steps=a.benchmark_steps,
            env_steps_per_second=a.num_envs*a.benchmark_steps/(time.monotonic()-started),
            cuda_peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30,
            cuda_free_gb=torch.cuda.mem_get_info()[0]/2**30,
            scene_count_visited=int((env.bank.visit_count>0).sum()),episodes=len(env.completed),
            completed_episodes=env.completed,robot_actuation=False)
        (a.run/"benchmark.json").write_text(json.dumps(result,indent=2)+"\n")
        print("BENCHMARK",json.dumps({k:v for k,v in result.items() if k!="completed_episodes"}),flush=True)
        return

    # Statistics are frozen before likelihood-based optimization. Burn-in uses
    # generated scenes and native teacher, never the old side1 dataset.
    if resume_checkpoint is None:
        moments=[]
        env.family="all"; env.reset(torch.arange(a.num_envs,device=device)); obs=env.observe()
        with torch.no_grad():
            for _ in range(32):
                moments.append(obs["obs"])
                obs,*_=env.step(env.teacher_targets(expert,obs))
        values=torch.cat(moments)
        prototype.mean.copy_(values.mean(0)); prototype.std.copy_(values.std(0).clamp_min(.05))
        del moments,values
    else:
        # Restore the original fitted statistics; don't waste a new burn-in on
        # values that would immediately be overwritten by learner restoration.
        prototype.mean.copy_(resume_checkpoint["models"]["generalist"]["mean"])
        prototype.std.copy_(resume_checkpoint["models"]["generalist"]["std"])
    models={f:WholeBodyPolicy(prototype,device) for f in ("lateral","low","overhead","mixed","generalist")}
    optimizers={f:torch.optim.Adam(m.trainable(),lr=3e-4) for f,m in models.items()}
    frozen=prototype.frozen_hash()
    del prototype
    # Original GRAIL action/token replay remains a separately labelled loss.
    retention_path=ROOT/"runs/20260912_cat_distill_flat_v4"
    retention_meta=json.loads((retention_path/"dataset.json").read_text())
    if sha(retention_path/"dataset.npz")!=retention_meta["dataset_sha256"]:
        raise ValueError("Changed GRAIL retention recording")
    with np.load(retention_path/"dataset.npz",allow_pickle=False) as data:
        keep=(data["mode"]==0)&~data["validation"]
        retention={k:torch.tensor(data[k][keep],device=device) for k in ("obs","cat","base","mode","target")}
    source=json.loads((ROOT/"artifacts/cat_release/logs_v1/generalist_v1/checkpoints/config.json").read_text())
    cfg=source["policy_config"]
    # Transition budgets do not silently grow when GPU environment count grows.
    transition_budgets={}
    for name,budget in (("transfer_updates",4194304),("specialist_ppo_updates",16777216),
                        ("dagger_updates",8388608),("generalist_ppo_updates",134217728)):
        if getattr(a,name) is None:
            setattr(a,name,math.ceil(budget/(a.num_envs*cfg["unroll_length"])))
            transition_budgets[name]=budget
        else:
            transition_budgets[name]=getattr(a,name)*a.num_envs*cfg["unroll_length"]
    phases=[]
    phase_budgets=[]
    for family in ("lateral","low","overhead","mixed"):
        phases += [(family,"transfer",a.transfer_updates),(family,"ppo",a.specialist_ppo_updates)]
        phase_budgets += [transition_budgets["transfer_updates"],transition_budgets["specialist_ppo_updates"]]
    phases += [("generalist","dagger",a.dagger_updates),("generalist","ppo",a.generalist_ppo_updates)]
    phase_budgets += [transition_budgets["dagger_updates"],transition_budgets["generalist_ppo_updates"]]
    if a.smoke_updates:
        phases=[("lateral","transfer",a.smoke_updates),("lateral","ppo",1),
            ("generalist","dagger",1),("generalist","ppo",1)]
        phase_budgets=[p[2]*a.num_envs*cfg["unroll_length"] for p in phases]
    phase_progress=[0]*len(phases)
    if resume_checkpoint is not None:
        source_config=json.loads((a.resume.parent/"config.json").read_text())
        phase_progress,phase_budgets=resume_phase_progress(resume_checkpoint,source_config,[p[:2] for p in phases])
    config=dict(schema="cat-generated-whole-body-training-v1",args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        phase_budget_transitions=phase_budgets,initial_phase_transitions=list(phase_progress),
        budget_semantics="Preserve transitions on resize; last complete vector rollout can overshoot each stage by less than one batch",
        resume_checkpoint_sha256=sha(a.resume) if a.resume else None,
        source_git_revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        source_files_sha256={name:sha(ROOT/name) for name in ("cat_parallel_train.py","cat_parallel_env.py",
            "cat_parallel_policy.py","cat_parallel_core.py","cat_distill_model.py","cat_direct_isaac.py","cat_direct_native.py")},
        native_recipe=source,phases=phases,bank_sha256=env.bank.hash,
        observation="CAT 162D noisy/delayed field + GRAIL 1029D; virtual reference object, no live LiDAR",
        action="64D pre-tanh Gaussian -> bounded motor tokens -> frozen GRAIL 29-joint decoder",
        adapter_hidden_sizes=[512,256,128,64],
        teacher="released CAT generalist teaches initial family students; our family specialists teach generalist",
        deviations=["torch PPO/PhysX port, not JAX/MJX bit equivalence", "GRAIL 64D latent action and 1029D proprioception plus native CAT 250D privileged critic input",
            "CAT leg-target MSE transfer; whole-body specialist KL DAgger", "single-GPU scaled batch",
            "physical floor and CAT SDF-only clutter; full GRAIL terrain environment remains a later task"],
        robot_actuation=False)
    (a.run/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    import wandb
    wb=wandb.init(entity="skvayzer",project="grail-cat",name=a.run.name,dir=str(a.run),
        config=config,mode=a.wandb_mode,job_type="generated-clutter-whole-body")
    (a.run/"wandb.json").write_text(json.dumps(dict(id=wb.id,url=wb.url))+"\n")
    total_steps,total_updates=0,0
    if a.resume:
        checkpoint=resume_checkpoint
        if checkpoint["bank_sha256"]!=env.bank.hash or checkpoint["contract"]!=contract or tuple(checkpoint["frozen_hashes"])!=frozen:
            raise ValueError("Resume provenance mismatch")
        for key in models:
            restore_learner(models[key],checkpoint["models"][key])
            optimizers[key].load_state_dict(checkpoint["optimizers"][key])
        torch.set_rng_state(checkpoint["cpu_rng"].cpu()); torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["cuda_rng"]])
        env.rng.set_state(checkpoint["env_rng"].cpu())
        env.bank.success_ema.copy_(checkpoint["success_ema"])
        env.bank.visit_count.copy_(checkpoint["visit_count"])
        total_steps,total_updates=checkpoint["total_steps"],checkpoint["total_updates"]
    started=time.monotonic(); stop_requested=False
    def stop_handler(*_):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGTERM,stop_handler);signal.signal(signal.SIGINT,stop_handler)
    log=(a.run/"metrics.jsonl").open("x")
    def save(phase,update):
        path=a.run/f"checkpoint_{total_steps:012d}.pt"
        if path.exists():
            return path
        checkpoint=dict(schema="cat-generated-checkpoint-v1",models={k:learner_state(m) for k,m in models.items()},
            optimizers={k:o.state_dict() for k,o in optimizers.items()},cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),env_rng=env.rng.get_state(),
            success_ema=env.bank.success_ema,visit_count=env.bank.visit_count,
            phase=phase,next_update=update,total_steps=total_steps,
            phase_transitions=list(phase_progress),phase_budget_transitions=list(phase_budgets),
            total_updates=total_updates,bank_sha256=env.bank.hash,contract=contract,frozen_hashes=frozen,
            resume_semantics="learner/optimizer/RNG continuation; fresh randomized episodes, not bit-exact PhysX replay",robot_actuation=False)
        temporary=path.with_suffix(".pt.partial")
        torch.save(checkpoint,temporary); os.replace(temporary,path)
        (a.run/"latest.json").write_text(json.dumps(dict(checkpoint=path.name,sha256=sha(path),total_steps=total_steps))+"\n")
        return path
    @torch.no_grad()
    def evaluate(policy,family):
        nonlocal stop_requested
        previous_family=env.family
        env.family="all" if family=="generalist" else family
        env.split="validation"
        env.completed=[]
        env.reset(torch.arange(a.num_envs,device=device));evaluation_obs=env.observe()
        used=set(env.scene.tolist())
        evaluated_steps=0
        for _ in range(env.config["episode_length"]):
            target=policy.targets(evaluation_obs,policy.mean(evaluation_obs))[:,env.grail_order]
            evaluation_obs,*_=env.step(target)
            used.update(env.scene.tolist())
            evaluated_steps+=1
            if evaluated_steps%32==0:
                if (any(d["pid"] not in a.reviewed_deployment_pid for d in gpu_robot_deployments())
                        or torch.cuda.mem_get_info()[0]<a.min_free_gpu_gib*2**30):
                    print("Evaluation interrupted for shared-GPU safety/headroom",flush=True)
                    stop_requested=True
                if stop_requested or time.monotonic()-started>a.hours*3600:
                    break
        episodes=env.completed
        result=dict(total_steps=total_steps,family=family,teacher_fraction=0.,
            split="held_out_geometry",scenes=len(used),episodes=len(episodes),
            success_rate=sum(e["success"] for e in episodes)/max(1,len(episodes)),
            fall_rate=sum(e["fall"] for e in episodes)/max(1,len(episodes)),
            collision_rate=sum(e["collision"] for e in episodes)/max(1,len(episodes)),
            running_unfinished_envs=a.num_envs,episode_records=episodes,
            evaluated_steps=evaluated_steps,full_horizon=evaluated_steps==env.config["episode_length"])
        (a.run/f"evaluation_{total_steps:012d}.json").write_text(json.dumps(result,indent=2)+"\n")
        wb.log({"eval/"+k:v for k,v in result.items() if isinstance(v,(int,float))},step=total_steps)
        env.completed=[];env.split="train";env.family=previous_family
        env.reset(torch.arange(a.num_envs,device=device))
        return env.observe()
    for phase_id,(family,kind,updates) in enumerate(phases):
        if phase_progress[phase_id]>=phase_budgets[phase_id]: continue
        policy=models[family]; optimizer=optimizers[family]
        env.family="all" if family=="generalist" else family
        env.split="train";env.max_difficulty=1.
        env.reset(torch.arange(a.num_envs,device=device));obs=env.observe()
        chunk=a.num_envs*cfg["unroll_length"]
        first_update=phase_progress[phase_id]//chunk
        remaining=math.ceil((phase_budgets[phase_id]-phase_progress[phase_id])/chunk)
        for update in range(first_update,first_update+remaining):
            tick=time.monotonic(); rows=[]
            beta=max(0.,1.-phase_progress[phase_id]/max(1,phase_budgets[phase_id]*.75)) if kind=="transfer" else 0.
            for _ in range(cfg["unroll_length"]):
                with torch.no_grad():
                    dist=policy.distribution(obs)
                    z=dist.sample() if kind=="ppo" else dist.loc
                    value=policy.value(obs)
                    target=policy.targets(obs,z)
                    label=env.teacher_targets(expert,obs)
                    applied=target[:,env.grail_order]
                    if beta:
                        assisted=env.uniform((a.num_envs,1),0,1)<beta
                        applied=torch.where(assisted,label,applied)
                    teacher_mean=torch.zeros_like(z)
                    teacher_std=torch.zeros_like(z)
                    if kind=="dagger":
                        for name in ("lateral","low","overhead","mixed"):
                            mask=torch.tensor([env.bank.rows[i]["family"]==name for i in env.scene.tolist()],device=device)
                            if mask.any():
                                subset={k:v[mask] for k,v in obs.items()}
                                td=models[name].distribution(subset)
                                teacher_mean[mask]=td.loc;teacher_std[mask]=td.scale
                    nxt,reward,terminated,truncated,info=env.step(applied)
                    next_value=policy.value(info["final_obs"])
                    rows.append(dict(**obs,z=z,old_logp=dist.log_prob(z).sum(-1),value=value,
                        reward=reward,next_value=next_value,terminated=terminated,truncated=truncated,
                        target=label[:,[env.names.index(n) for n in contract["joints"]]],
                        teacher_mean=teacher_mean,teacher_std=teacher_std))
                    obs=nxt
            data={k:torch.stack([r[k] for r in rows]) for k in rows[0]}
            adv,returns=generalized_advantage(data["reward"],data["value"],data["next_value"],
                data["terminated"],data["truncated"],gamma=cfg["discounting"],lam=cfg["gae_lambda"])
            data["advantages"]=(adv-adv.mean())/(adv.std()+1e-8); data["returns"]=returns
            flat={k:v.flatten(0,1) for k,v in data.items()}; count=len(flat["z"])
            minibatch=max(16,count//cfg["num_minibatches"])
            statistics=[]
            for epoch in range(cfg["num_updates_per_batch"]):
                permutation=torch.randperm(count,device=device)
                for batch_index,ids in enumerate(permutation.split(minibatch)):
                    b={k:v[ids] for k,v in flat.items()}; inputs={k:b[k] for k in obs}
                    dist=policy.distribution(inputs)
                    value=policy.value(inputs)
                    predicted=policy.targets(inputs,dist.loc)
                    leg_loss=(predicted[:,policy.student.legs]-b["target"][:,policy.student.legs]).square().mean()
                    posture=(predicted[:,policy.student.upper]-policy.student.offset[policy.student.upper]).square().mean()
                    value_loss=.5*(value-b["returns"]).square().mean()
                    if kind=="ppo":
                        ratio=(dist.log_prob(b["z"]).sum(-1)-b["old_logp"]).exp()
                        if epoch==0 and batch_index==0 and (ratio-1).abs().max()>.002:
                            raise ValueError("Unchanged-policy PPO likelihood mismatch")
                        actor_loss=-torch.minimum(ratio*b["advantages"],ratio.clamp(1-cfg["clipping_epsilon"],1+cfg["clipping_epsilon"])*b["advantages"]).mean()
                        actor_loss-=cfg["entropy_cost"]*dist.entropy().sum(-1).mean()
                    elif kind=="dagger":
                        td=Normal(b["teacher_mean"],b["teacher_std"].clamp_min(1e-5))
                        actor_loss=torch.distributions.kl_divergence(td,dist).sum(-1).mean()
                    else:
                        actor_loss=leg_loss+.25*posture
                    ri=torch.randint(len(retention["mode"]),(min(64,len(ids)),),device=device)
                    rb={k:v[ri] for k,v in retention.items()}
                    retention_loss=(policy.student(**{k:rb[k] for k in ("obs","cat","base","mode")})-rb["target"]).square().mean()
                    loss=actor_loss+value_loss+retention_loss
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    grad=torch.nn.utils.clip_grad_norm_(policy.trainable(),cfg["max_grad_norm"])
                    if not torch.isfinite(loss) or not torch.isfinite(grad):
                        raise ValueError("Nonfinite learner; checkpoint preserved, aborting")
                    optimizer.step();total_updates+=1
                    statistics.append(torch.stack((loss.detach(),leg_loss.detach(),retention_loss.detach(),value_loss.detach())))
            total_steps+=count
            phase_progress[phase_id]+=count
            phase_complete=phase_progress[phase_id]>=phase_budgets[phase_id]
            average=torch.stack(statistics).mean(0).tolist()
            episodes=env.completed;env.completed=[]
            metrics=dict(total_steps=total_steps,optimizer_updates=total_updates,phase=phase_id,family=family,kind=kind,
                phase_update=update,teacher_fraction=beta,loss=average[0],leg_mse=average[1],retention_mse=average[2],
                phase_transitions=phase_progress[phase_id],phase_budget_transitions=phase_budgets[phase_id],
                value_loss=average[3],reward=float(data["reward"].mean()),
                env_steps_per_second=count/(time.monotonic()-tick),episodes=len(episodes),
                success_rate=sum(e["success"] for e in episodes)/max(1,len(episodes)),
                fall_rate=sum(e["fall"] for e in episodes)/max(1,len(episodes)),
                collision_rate=sum(e["collision"] for e in episodes)/max(1,len(episodes)),
                unique_train_scenes_visited=sum(int(env.bank.visit_count[i])>0 for i,r in enumerate(env.bank.rows) if r["split"]=="train"),
                unique_validation_scenes_visited=sum(int(env.bank.visit_count[i])>0 for i,r in enumerate(env.bank.rows) if r["split"]=="validation"),
                cuda_allocated_gb=torch.cuda.memory_allocated()/2**30,
                cuda_reserved_gb=torch.cuda.memory_reserved()/2**30,
                cuda_peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30,
                gpu_process_gib=process_gpu_memory_gib(),
                cuda_free_gb=torch.cuda.mem_get_info()[0]/2**30,elapsed_s=time.monotonic()-started)
            log.write(json.dumps(metrics)+"\n");log.flush()
            (a.run/"status.json").write_text(json.dumps(dict(**metrics,state="training",robot_actuation=False))+"\n")
            wb.log({k:v for k,v in metrics.items() if isinstance(v,(int,float))},step=total_steps)
            print("TRAIN",json.dumps(metrics),flush=True)
            if any(d["pid"] not in a.reviewed_deployment_pid for d in gpu_robot_deployments()):
                print("GPU robot deployment appeared; saving and stopping for operator review",flush=True)
                stop_requested=True
            import shutil
            if shutil.disk_usage(a.run).free<5*2**30:
                print("Low free disk: saving and stopping before filling the filesystem",flush=True)
                stop_requested=True
            if metrics["cuda_free_gb"]<a.min_free_gpu_gib:
                print("GPU headroom below configured reserve: saving and stopping",flush=True)
                stop_requested=True
            if not stop_requested and not a.smoke_updates and ((update+1)%250==0 or phase_complete):
                save(phase_id,update+1)
                (a.run/"status.json").write_text(json.dumps(dict(**metrics,state="evaluating",robot_actuation=False))+"\n")
                obs=evaluate(policy,family)
            if update%25==0 or phase_complete or stop_requested or time.monotonic()-started>a.hours*3600:
                if policy.student.frozen_hash()!=frozen:
                    raise ValueError("Frozen decoder or teacher modified")
                save(phase_id,update+1)
            if stop_requested or time.monotonic()-started>a.hours*3600:
                (a.run/"status.json").write_text(json.dumps(dict(**metrics,state="stopped_checkpointed",robot_actuation=False))+"\n")
                log.close();wb.finish();return
        save(phase_id+1,0)
    (a.run/"status.json").write_text(json.dumps(dict(total_steps=total_steps,
        optimizer_updates=total_updates,state="completed",robot_actuation=False))+"\n")
    log.close();wb.finish()


if __name__=="__main__":
    main()
