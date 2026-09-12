#!/usr/bin/env python3
"""Compare every torch reward term against the unchanged native JAX methods.

CPU only; run with Click-and-Traverse/.venv/bin/python. No simulator or training.
"""
import ast
import json
import os
from pathlib import Path
import sys
import types
os.environ["JAX_PLATFORMS"]="cpu"
os.environ["OMP_NUM_THREADS"]="2"
import numpy as np
import jax
import jax.numpy as jp
import torch
from cat_parallel_core import native_rewards,sha


def native():
    root=Path(__file__).resolve().parents[2]/"Click-and-Traverse"
    wanted={"_get_reward","_reward_orientation","_reward_tracking_root_field","_cost_body_motion",
        "_reward_body_rotation","_cost_foot_contact","_cost_foot_clearance","_cost_foot_slip",
        "_cost_foot_balance","_cost_straight_knee","_cost_foot_far","_cost_joint_pos_limits",
        "_cost_torque","_cost_smoothness_joint","_cost_smoothness_action","_re_gf0","_re_sdf"}
    obj=types.SimpleNamespace(); hashes={}
    for name in ("env_loco.py","env_cat.py"):
        path=root/"cat_ppo/envs/g1"/name
        hashes[name]=sha(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.FunctionDef) and node.name in wanted:
                node.returns=None;node.decorator_list=[]
                for arg in node.args.args:arg.annotation=None
                space=dict(jp=jp,jax=jax,np=np)
                exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),"exec"),space)
                setattr(obj,node.name,types.MethodType(space[node.name],obj))
    return obj,hashes


def main():
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parent
    cfg=json.loads((root/"artifacts/cat_release/logs_v1/generalist_v1/checkpoints/config.json").read_text())["env_config"]
    obj,hashes=native()
    obj._config=types.SimpleNamespace(ang_vel_yaw=cfg["ang_vel_yaw"],torso_height=cfg["torso_height"],
        reward_config=types.SimpleNamespace(**cfg["reward_config"]))
    obj._head_site_id=0;obj._feet_site_id=jp.array([3,4]);obj.body_id_pelvis=0
    obj.body_ids_left_leg=jp.array([0,1]);obj.body_ids_right_leg=jp.array([2,3]);obj._knee_indices=[3,9]
    obj._foot_linvel_sensor_adr=jp.array([[0,1,2],[3,4,5]]);obj.obs_joint_ids=jp.arange(23);obj.dt=.02
    obj._soft_lowers=jp.full((29,),-1.);obj._soft_uppers=jp.full((29,),1.)
    rng=np.random.default_rng(728)
    n=32
    t=lambda shape:torch.tensor(rng.normal(size=shape),dtype=torch.float32)
    s=dict(command=t((n,4)),lin_vel=t((n,3)),torso_ang_nav=t((n,3)),sites=t((n,11,3)),
        site_vel=t((n,11,3)),fields=dict(gf=t((n,11,3)),sdf=t((n,11,1))),gait=t((n,2)).sign(),
        pelvis_rpy=t((n,3)),torso_rpy=t((n,3)),legs_nav=torch.eye(3).repeat(n,4,1,1),
        feet_contact=t((n,2))>0,com=t((n,3)),action=t((n,12)),qpos=t((n,29)),qvel=t((n,29)),
        previous_qvel=t((n,29)),obs_ids=list(range(23)),knee_ids=[3,9],last_action=t((n,12)),
        last_last_action=t((n,12)),soft_lower=torch.full((29,),-1.),soft_upper=torch.full((29,),1.),
        torque=t((n,29)),feet_sensor_vel=t((n,2,3)))
    s["command"][:,0]=(s["command"][:,0]>0).float()
    s["command"][0]=0
    reward,terms=native_rewards(s,cfg["reward_config"]["scales"])
    errors={k:0. for k in terms}
    for i in range(n):
        a=lambda v:jp.array(v[i].numpy())
        data=types.SimpleNamespace(site_xpos=a(s["sites"]),subtree_com=a(s["com"])[None],
            xmat=a(s["legs_nav"]),qpos=jp.concatenate((jp.zeros(7),a(s["qpos"]))),
            qvel=jp.concatenate((jp.zeros(6),a(s["qvel"]))),actuator_force=a(s["torque"]),
            sensordata=a(s["feet_sensor_vel"]).reshape(-1))
        info=dict(command=a(s["command"]),navi_pelvis_rpy=a(s["pelvis_rpy"]),navi_torso_rpy=a(s["torso_rpy"]),
            global_lin_vel=a(s["lin_vel"]),navi_torso_ang_vel=a(s["torso_ang_nav"]),navi2world_rot=jp.eye(3),
            navi2world_pose=jp.eye(4),gait_mask=a(s["gait"]),foot_height=.07,last_joint_vel=a(s["previous_qvel"]),
            last_act=a(s["last_action"]),last_last_act=a(s["last_last_action"]))
        for group,ids in (("head",[0]),("pelv",[1]),("tors",[2]),("feet",[3,4]),("hands",[5,6]),("knees",[7,8]),("shlds",[9,10])):
            info[group+"gf"]=a(s["fields"]["gf"])[jp.array(ids)]
            info[group+"df"]=a(s["fields"]["sdf"])[jp.array(ids)]
            info[group+"_pos"]=a(s["sites"])[jp.array(ids)]
            info[group+"_vel"]=a(s["site_vel"])[jp.array(ids)]
        info["head_pos"]=info["head_pos"].reshape(3)
        reference=obj._get_reward(data,a(s["action"]),info,jp.array(False),a(s["feet_contact"]))
        for k in terms:
            got=float(terms[k][i]);expected=float(reference[k])
            np.testing.assert_allclose(got,expected,rtol=2e-5,atol=2e-5,err_msg=k)
            errors[k]=max(errors[k],abs(got-expected))
    result=dict(samples=n,all_terms_match=True,max_absolute_error=errors,native_sources=hashes)
    print(json.dumps(result,indent=2))
    if len(sys.argv)>1:
        path=Path(sys.argv[1])
        with path.open("x") as f:json.dump(result,f,indent=2)


if __name__=="__main__":main()
