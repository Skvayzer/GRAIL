"""GPU-batched Isaac native-G1 CAT task. Imported after AppLauncher only.

Procedurally generated per-episode fields, native torque PD and CAT randomization.
The physical floor supports the robot; clutter is SDF-only as in CAT training.
"""
import json
import math
from pathlib import Path
import numpy as np
import torch
from cat_parallel_core import FieldBank, BatchHistory, pack_cat, native_rewards, rpy
from gear_sonic.research.cat_bridge import CatObservationBridge, heading_matrix, normalize_fields
from gear_sonic.research.cat_command import field_command, gait_step, delayed_sites
from gear_sonic.research.learner_state import quaternion_matrix


class ParallelCatEnv:
    def __init__(self, run, bank, grail_contract, num_envs=256, device="cuda:0", seed=0, randomize=True):
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.actuators import IdealPDActuatorCfg
        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
        from isaacsim.core.cloner import GridCloner
        from isaacsim.core.utils.extensions import enable_extension
        from pxr import UsdGeom, UsdPhysics
        from cat_direct_native import make_player, prepare_robot_xml
        from cat_direct_isaac import add_native_pair_shapes, configure_contacts
        run = Path(run)
        self.device, self.n, self.randomize = device, num_envs, randomize
        self.rng = torch.Generator(device=device).manual_seed(seed)
        self.bank = FieldBank(bank, device)
        self.family, self.split, self.max_difficulty = "lateral", "train", 1.
        self.config = json.loads((Path(__file__).parent/"artifacts/cat_release/logs_v1/generalist_v1/checkpoints/config.json").read_text())["env_config"]
        player, consts, cat_contract = make_player("side1") # Robot contract only; never used as a training field.
        model = player.mj_model
        self.native_names = [model.actuator(i).name for i in range(model.nu)]
        enable_extension("isaacsim.asset.importer.mjcf")
        xml = prepare_robot_xml(run)
        asset = MjcfConverter(MjcfConverterCfg(asset_path=str(xml), usd_dir=str(run/"robot_usd"),
            usd_file_name="cat_robot.usd", fix_base=False, make_instanceable=False,
            import_inertia_tensor=True, self_collision=True))
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.002, device=device,
            render_interval=1000000, physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1., dynamic_friction=1., restitution=0.),
            physx=sim_utils.PhysxCfg(solver_type=1, gpu_max_rigid_contact_count=2**21,
                gpu_max_rigid_patch_count=2**18)))
        ground = sim_utils.GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1., dynamic_friction=1., restitution=0.))
        ground.func("/World/Ground", ground)
        cloner = GridCloner(spacing=5.)
        cloner.define_base_env("/World/envs")
        paths = cloner.generate_paths("/World/envs/env", num_envs)
        UsdGeom.Xform.Define(self.sim.stage, paths[0])
        gains = lambda v: dict(zip(self.native_names,map(float,v)))
        cfg = ArticulationCfg(prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=asset.usd_path, activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(linear_damping=0.,angular_damping=0.,
                    max_linear_velocity=100.,max_angular_velocity=100.,max_depenetration_velocity=10.),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                    solver_position_iteration_count=8,solver_velocity_iteration_count=4)),
            init_state=ArticulationCfg.InitialStateCfg(pos=tuple(consts.DEFAULT_QPOS[:3]),
                rot=tuple(consts.DEFAULT_QPOS[3:7]),joint_pos=gains(consts.DEFAULT_QPOS[7:]),joint_vel={".*":0.}),
            actuators={"native_torque":IdealPDActuatorCfg(joint_names_expr=[".*"],stiffness=0.,damping=0.,
                effort_limit=1000.,effort_limit_sim=1000.,velocity_limit_sim=1000.,
                armature=gains(model.dof_armature[6:]),friction=0.,dynamic_friction=0.,viscous_friction=0.)})
        self.robot = Articulation(cfg)
        prefix = paths[0]+"/Robot"
        world = self.sim.stage.GetPrimAtPath(prefix+"/worldBody")
        if world and world.HasAPI(UsdPhysics.ArticulationRootAPI):
            world.SetActive(False)
        add_native_pair_shapes(self.sim.stage,model,prefix)
        colliders = [p for p in self.sim.stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
        configure_contacts(self.sim.stage,model,colliders,prefix)
        origins = cloner.clone(source_prim_path=paths[0],prim_paths=paths,replicate_physics=True,
            base_env_path="/World/envs",root_path="/World/envs/env_",copy_from_source=True)
        cloner.filter_collisions(self.sim.get_physics_context().prim_path,"/World/collisions",paths,global_paths=["/World/Ground"])
        self.origins = torch.tensor(origins,device=device,dtype=torch.float32)
        # Match body paths by suffix; contact sensor resolves the actual imported hierarchy.
        foot_paths = [str(p.GetPath()) for p in self.sim.stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)
            and str(p.GetPath()).startswith(prefix+"/") and p.GetName()=="left_ankle_roll_link"]
        if len(foot_paths)!=1:
            raise ValueError("Native foot contact sensor path missing")
        expression = foot_paths[0].replace(paths[0],"/World/envs/env_.*").replace("left_ankle_roll_link",".*_ankle_roll_link")
        self.contacts = ContactSensor(ContactSensorCfg(prim_path=expression,update_period=.0,history_length=1))
        self.sim.reset()
        self.robot.reset(); self.contacts.reset()
        self.names = self.robot.joint_names
        order = [self.native_names.index(n) for n in self.names]
        self.tensor = lambda v: torch.as_tensor(v,device=device,dtype=torch.float32)
        self.kp, self.kd = self.tensor(consts.KPs[order]),self.tensor(consts.KDs[order])
        self.torque_limit = self.tensor(model.actuator_ctrlrange[:,1][order])
        self.soft_lower, self.soft_upper = self.tensor(player._soft_lowers[order]), self.tensor(player._soft_uppers[order])
        self.default = self.tensor(player._default_qpos[order])
        self.robot.write_joint_friction_coefficient_to_sim(self.tensor(model.dof_frictionloss[6:][order]).expand(num_envs,-1),
            self.tensor(model.dof_frictionloss[6:][order]).expand(num_envs,-1),
            self.tensor(model.dof_damping[6:][order]).expand(num_envs,-1))
        self.bridge = CatObservationBridge(cat_contract,self.names,self.robot.body_names)
        self.history = BatchHistory(num_envs,grail_contract,self.names,device)
        self.leg_ids = self.bridge.action_ids
        self.grail_order = [grail_contract["joints"].index(n) for n in self.names]
        self.pelvis = self.robot.body_names.index("pelvis")
        self.torso = self.robot.body_names.index(cat_contract["sites"][2]["body"])
        self.leg_bodies = [self.robot.body_names.index(n) for n in player.body_names_left_leg+player.body_names_right_leg]
        self.knees = [self.names.index(n) for n in ("left_knee_joint","right_knee_joint")]
        self.foot_sensor_order = [self.contacts.body_names.index(n) for n in ("left_ankle_roll_link","right_ankle_roll_link")]
        self.masses = self.robot.root_physx_view.get_masses().to(device)
        self.targets = self.default.repeat(num_envs,1)
        self.last_action = torch.zeros(num_envs,12,device=device)
        self.last_last_action = self.last_action.clone()
        self.steps = torch.zeros(num_envs,dtype=torch.long,device=device)
        self.scene = torch.zeros_like(self.steps)
        self.phase = torch.zeros(num_envs,2,device=device)
        self.phase_dt = torch.zeros(num_envs,1,device=device)
        self.stop = torch.full_like(self.steps,100)
        self.command = torch.zeros(num_envs,4,device=device)
        self.kp_scale = torch.ones(num_envs,1,device=device); self.kd_scale=self.kp_scale.clone()
        self.rfi = torch.zeros_like(self.targets)
        self.push_interval = torch.full_like(self.steps,400)
        self.episode_reward = torch.zeros(num_envs,device=device)
        self.episode_violated = torch.zeros(num_envs,dtype=torch.bool,device=device)
        self.goal_dwell = torch.zeros_like(self.steps)
        self.completed = []
        self.reset(torch.arange(num_envs,device=device))
        (run/"task_contract.json").write_text(json.dumps(dict(num_envs=num_envs,bank_sha256=self.bank.hash,
            cat_contract=cat_contract,grail_contract=grail_contract,physics="GPU PhysX",
            geometry="180+ generated layouts; per-episode resampling, not side1 replicas",
            deviations=["PhysX solver instead of MJX", "29-joint GRAIL latent actions instead of 12 CAT increments",
                        "self-contact physics enabled; SDF/head/upvector termination; no explicit self-contact termination query"],
            robot_actuation=False),indent=2)+"\n")

    def uniform(self, shape, low, high):
        return low+(high-low)*torch.rand(shape,device=self.device,generator=self.rng)

    def reset(self, ids):
        if not len(ids):
            return
        count=len(ids)
        self.scene[ids]=self.bank.choose(count,self.family,self.split,self.max_difficulty,self.rng)
        self.bank.visit_count.scatter_add_(0,self.scene[ids],torch.ones_like(ids))
        root=self.robot.data.default_root_state[ids].clone()
        root[:,:3]+=self.origins[ids]; root[:,2]=.8+self.origins[ids,2]
        joints=self.default.repeat(count,1)
        if self.randomize:
            root[:,:2]+=self.uniform((count,2),-1.,1.)
            yaw=self.uniform((count,),-math.pi/2,math.pi/2)
            root[:,3:7]=torch.stack((torch.cos(yaw/2),yaw*0,yaw*0,torch.sin(yaw/2)),-1)
            root[:,7:13]=self.uniform((count,6),-.5,.5)
            joints=(joints*self.uniform((count,29),.5,1.5)).clamp(self.soft_lower,self.soft_upper)
        self.robot.write_root_pose_to_sim(root[:,:7],env_ids=ids)
        self.robot.write_root_velocity_to_sim(root[:,7:13],env_ids=ids)
        self.robot.write_joint_state_to_sim(joints,torch.zeros_like(joints),env_ids=ids)
        self.robot.reset(ids); self.contacts.reset(ids)
        # Native reset randomizes physical qpos, but starts motor-target history
        # at the nominal posture, not at the randomized joint measurements.
        self.targets[ids]=self.default; self.last_action[ids]=0; self.last_last_action[ids]=0
        self.steps[ids]=0; self.stop[ids]=100; self.command[ids]=0; self.history.ready[ids]=False
        self.episode_reward[ids]=0; self.episode_violated[ids]=False; self.goal_dwell[ids]=0
        self.phase[ids]=torch.tensor([0.,math.pi],device=self.device)
        if self.randomize:
            self.phase[ids]=torch.where(self.uniform((count,1),0,1)>.5,
                self.phase[ids],self.phase[ids].flip(-1))
        self.phase_dt[ids]=2*math.pi*.02*self.uniform((count,1),1.3,1.5)
        self.kp_scale[ids]=self.uniform((count,1),.75,1.25) if self.randomize else 1.
        self.kd_scale[ids]=self.uniform((count,1),.75,1.25) if self.randomize else 1.
        self.rfi[ids]=.1*self.uniform((count,29),.5,1.5)*self.torque_limit if self.randomize else 0.
        self.push_interval[ids]=(self.uniform((count,),5,10)/.02).round().long()
        self.sim.forward(); self.robot.update(.002)
        physical=self.bridge.read_articulation(self.robot.data,self.origins)
        if not hasattr(self,"cached_root"):
            self.cached_root=physical["sites"][:,1].clone()
            self.cached_rotation=physical["pelvis_rotation"].clone()
            self.previous_sites=physical["sites"].clone()
        self.cached_root[ids]=physical["sites"][ids,1]
        self.cached_rotation[ids]=physical["pelvis_rotation"][ids]
        self.previous_sites[ids]=physical["sites"][ids]
        fields=self.bank.sample(self.scene,physical["sites"])
        gf,bf=normalize_fields(fields["gf"][ids],fields["bf"][ids],torch.ones(count,device=self.device))
        self.command[ids]=field_command(gf,bf)

    def observe(self, advance_history=True):
        p=self.bridge.read_articulation(self.robot.data,self.origins)
        sites=p["sites"]; root=sites[:,1]
        current=self.bank.sample(self.scene,sites)
        update=(self.steps%5)==0
        self.cached_root[update]=root[update]; self.cached_rotation[update]=p["pelvis_rotation"][update]
        delayed=delayed_sites(sites,root,p["pelvis_rotation"],self.cached_root,self.cached_rotation)
        delayed_field=self.bank.sample(self.scene,delayed)
        current["gf"],current["bf"]=normalize_fields(current["gf"],current["bf"],self.command[:,0])
        delayed_field["gf"],delayed_field["bf"]=normalize_fields(delayed_field["gf"],delayed_field["bf"],self.command[:,0])
        cmd=field_command(delayed_field["gf"],delayed_field["bf"]); cmd[:,0]=self.command[:,0]
        noise=None
        if self.randomize:
            noise=[self.uniform((self.n,width),-scale,scale) for width,scale in ((3,.2),(3,.05),(23,.03),(23,1.5))]
        cat=pack_cat(p,self.bridge.obs_ids,p["joint_pos"].new_tensor(self.bridge.default_obs),
            self.last_action,self.targets[:,self.leg_ids],cmd,self.phase,delayed_field,noise)
        # CAT's original 250D noiseless privileged critic packet, in native
        # group order/world coordinates, plus explicit GRAIL proprioception.
        nav=heading_matrix(p["pelvis_rotation"])
        body_rot=quaternion_matrix(self.robot.data.body_quat_w)
        torso_rpy=rpy(nav.transpose(-1,-2)@body_rot[:,self.torso])
        site_vel=getattr(self,"site_velocity",torch.zeros_like(sites))
        gait=torch.where(self.phase.cos()>.6,1.,torch.where(self.phase.cos()<-.6,-1.,0.))
        contacts=self.contacts.data.net_forces_w[:,self.foot_sensor_order].norm(dim=-1)>1.
        field_groups=[]; start=0
        from gear_sonic.research.cat_bridge import GROUP_SIZES
        for size in GROUP_SIZES:
            field_groups.extend(current[k][:,start:start+size].flatten(1) for k in ("gf","bf","sdf"))
            start+=size
        local_vel=(p["pelvis_rotation"].transpose(-1,-2)@self.robot.data.body_lin_vel_w[:,self.pelvis,...,None]).squeeze(-1)
        privileged=torch.cat([p["gyro"],p["gravity"],p["joint_pos"][:,self.bridge.obs_ids]-p["joint_pos"].new_tensor(self.bridge.default_obs),
            p["joint_vel"][:,self.bridge.obs_ids],self.last_action,self.targets[:,self.leg_ids],self.command,
            torch.full((self.n,1),.07,device=self.device),self.phase.cos(),self.phase.sin(),local_vel,
            *field_groups,sites[:,0],site_vel[:,0],sites[:,1],sites[:,2],sites[:,3:5].flatten(1),site_vel[:,3:5].flatten(1),
            sites[:,5:7].flatten(1),site_vel[:,5:7].flatten(1),sites[:,7:9].flatten(1),sites[:,9:11].flatten(1),
            torso_rpy[:,:2],gait,contacts.float(),self.kp_scale,self.kd_scale,self.rfi],-1)
        if privileged.shape!=(self.n,250):
            raise ValueError("Native privileged observation packing mismatch")
        if advance_history:
            self.grail_obs=self.history.sample(p,root,self.targets)
        self.physical,self.fields=p,current
        return dict(obs=self.grail_obs.clone(),cat=cat,privileged=privileged,
            base=torch.zeros(self.n,64,device=self.device),mode=torch.ones(self.n,device=self.device))

    def teacher_targets(self, expert, observation):
        action=expert(observation["cat"])
        result=self.default.repeat(self.n,1)
        result[:,self.leg_ids]=self.bridge.leg_targets(action,self.targets[:,self.leg_ids])
        return result

    def step(self, targets):
        previous_vel=self.robot.data.joint_vel.clone()
        old=self.targets.clone()
        self.targets=targets.clamp(self.soft_lower,self.soft_upper).detach()
        action=(self.targets[:,self.leg_ids]-old[:,self.leg_ids])/.5
        if self.randomize:
            push=((self.steps+1)%self.push_interval)==0
            ids=push.nonzero().flatten()
            if len(ids):
                velocity=self.robot.data.root_vel_w[ids].clone()
                angle=self.uniform((len(ids),),0,2*math.pi)
                velocity[:,:2]+=torch.stack((angle.cos(),angle.sin()),-1)*self.uniform((len(ids),1),.1,1.)
                self.robot.write_root_velocity_to_sim(velocity,env_ids=ids)
        for _ in range(10):
            torque=self.kp*self.kp_scale*(self.targets-self.robot.data.joint_pos)-self.kd*self.kd_scale*self.robot.data.joint_vel
            if self.randomize:
                torque+=self.uniform(torque.shape,-1,1)*self.rfi
            torque=torque.clamp(-self.torque_limit,self.torque_limit)
            self.robot.set_joint_effort_target(torque); self.robot.write_data_to_sim()
            self.sim.step(render=False); self.robot.update(.002); self.contacts.update(.002)
        self.steps+=1
        p=self.bridge.read_articulation(self.robot.data,self.origins)
        sites=p["sites"]; vel=(sites-self.previous_sites)/.02
        fields=self.bank.sample(self.scene,sites)
        newcommand=field_command(fields["gf"],fields["bf"])
        command,phase,stop=gait_step(newcommand,self.command,self.phase,self.stop,self.phase_dt)
        self.command,self.phase,self.stop=command,phase,stop
        fields["gf"],fields["bf"]=normalize_fields(fields["gf"],fields["bf"],command[:,0])
        nav=heading_matrix(p["pelvis_rotation"])
        body_rot=quaternion_matrix(self.robot.data.body_quat_w)
        com=(self.robot.data.body_com_pos_w*self.masses[...,None]).sum(1)/self.masses.sum(-1,keepdim=True)-self.origins
        feet_contact=self.contacts.data.net_forces_w[:,self.foot_sensor_order].norm(dim=-1)>1.
        torso_ang=(nav.transpose(-1,-2)@self.robot.data.body_ang_vel_w[:,self.torso,...,None]).squeeze(-1)
        s=dict(command=command,lin_vel=self.robot.data.body_lin_vel_w[:,self.pelvis],torso_ang_nav=torso_ang,
            sites=sites,site_vel=vel,fields=fields,gait=torch.where(phase.cos()>.6,1.,torch.where(phase.cos()<-.6,-1.,0.)),
            pelvis_rpy=rpy(nav.transpose(-1,-2)@p["pelvis_rotation"]),
            torso_rpy=rpy(nav.transpose(-1,-2)@body_rot[:,self.torso]),
            legs_nav=nav[:,None].transpose(-1,-2)@body_rot[:,self.leg_bodies],feet_contact=feet_contact,
            com=com,action=action,qpos=p["joint_pos"],qvel=p["joint_vel"],previous_qvel=previous_vel,
            obs_ids=self.bridge.obs_ids,knee_ids=self.knees,last_action=self.last_action,
            last_last_action=self.last_last_action,soft_lower=self.soft_lower,soft_upper=self.soft_upper,
            torque=torque,feet_sensor_vel=vel[:,3:5])
        reward,terms=native_rewards(s,self.config["reward_config"]["scales"])
        violated=(fields["sdf"].squeeze(-1)<0).any(-1)
        fell=(sites[:,0,2]<.7)|(p["pelvis_rotation"][:,2,2]<0)
        finite=torch.isfinite(self.robot.data.root_state_w).all(-1)&torch.isfinite(p["joint_pos"]).all(-1)&torch.isfinite(p["joint_vel"]).all(-1)
        terminated=fell|((self.steps>=50)&violated)|~finite
        timeout=self.steps>=self.config["episode_length"]
        near=(sites[:,1,:2]-sites.new_tensor([2.,0.])).norm(dim=-1)<.2
        self.goal_dwell=torch.where(near & ~terminated,self.goal_dwell+1,0)
        # Preserve CAT's full 1000-step episode. Crossing an exit or briefly
        # touching the goal must not hide a later fall (as in the old pilot).
        success=timeout&(self.goal_dwell>=50)&~self.episode_violated
        truncated=timeout
        self.episode_reward+=reward; self.episode_violated|=(self.steps>=50)&violated
        done=terminated|truncated
        ids=done.nonzero().flatten()
        for i in ids.tolist():
            scene=int(self.scene[i]); good=bool(success[i] and not terminated[i])
            self.completed.append(dict(scene=scene,family=self.bank.rows[scene]["family"],split=self.split,
                success=good,fall=bool(fell[i]),collision=bool(violated[i]),steps=int(self.steps[i]),
                reward=float(self.episode_reward[i])))
            if self.split=="train":
                self.bank.success_ema[scene]=.95*self.bank.success_ema[scene]+.05*float(good)
        self.last_last_action=self.last_action.clone(); self.last_action=action
        self.previous_sites=sites.clone()
        self.site_velocity=vel
        # Bootstrap timeout with the terminal observation BEFORE resetting.
        next_obs=self.observe()
        final_obs={k:v.clone() for k,v in next_obs.items()}
        if len(ids):
            self.reset(ids)
            refreshed=self.observe(advance_history=False)
            # Only reset slots initialize new histories; ongoing slots are unchanged.
            p=self.bridge.read_articulation(self.robot.data,self.origins)
            # sample() would advance every slot twice. Restore histories for ongoing envs.
            saved=[h.clone() for h in self.history.history]
            new=self.history.sample(p,p["sites"][:,1],self.targets)
            ongoing=~done
            for h,old_h in zip(self.history.history,saved):
                h[ongoing]=old_h[ongoing]
            self.grail_obs[ids]=new[ids]
            refreshed["obs"]=self.grail_obs.clone()
            for k in next_obs:
                next_obs[k][ids]=refreshed[k][ids]
        return next_obs,reward,terminated,truncated,dict(final_obs=final_obs,terms=terms,
            out_of_domain=(~fields["in_domain"]).float().mean())
