"""Distribution, batched packet/history and reward regressions; no simulator."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import copy
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cat_parallel_bank import recipes,FAMILIES
from cat_parallel_core import FieldBank,BatchHistory,pack_cat,native_rewards
from cat_distill_model import ProprioHistory
from gear_sonic.research.cat_bridge import CatFieldSampler,CatObservationBridge,JOINTS,OBS_JOINTS
from cat_parallel_policy import WholeBodyPolicy,learner_state,restore_learner


class ParallelTests(unittest.TestCase):
    def test_real_gradient_paths_and_checkpoint_resume(self):
        from test_cat_distill import fixture
        template,batch,contract=fixture()
        policy=WholeBodyPolicy(template)
        obs={k:batch[k] for k in ("obs","cat","base","mode")}
        obs["privileged"]=torch.randn(8,250)*.05
        optimizer=torch.optim.Adam(policy.trainable(),lr=3e-4)
        frozen=policy.student.frozen_hash()
        target=batch["target"]
        def update():
            optimizer.zero_grad()
            predicted=policy.targets(obs,policy.mean(obs))
            loss=(predicted-target).square().mean()+policy.value(obs).square().mean()
            loss.backward();optimizer.step()
            return torch.cat([p.detach().flatten().clone() for p in policy.trainable()])
        update()
        state=copy.deepcopy(learner_state(policy));opt=copy.deepcopy(optimizer.state_dict())
        first=update()
        restore_learner(policy,state);optimizer.load_state_dict(copy.deepcopy(opt))
        second=update()
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        self.assertEqual(frozen,policy.student.frozen_hash())
        self.assertTrue(all(p.grad is None for p in policy.student.decoder.parameters()))
        self.assertTrue(all(p.grad is None for p in policy.student.features.parameters()))

    def test_recipe_diversity_and_disjoint_splits(self):
        rows=recipes()
        self.assertEqual(len(rows),240)
        self.assertEqual(len({r["seed"] for r in rows}),240)
        for family in FAMILIES:
            self.assertEqual(sum(r["family"]==family and r["split"]=="train" for r in rows),48)
            self.assertEqual(sum(r["family"]==family and r["split"]=="validation" for r in rows),12)
        self.assertEqual({r["difficulty"] for r in rows},{.2,.4,.6,.8})

    def test_banked_sampling_matches_independent_per_scene_sampler(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rng=np.random.default_rng(3);rows=[];fields=[]
            for i in range(3):
                f=dict(gf=rng.normal(size=(6,7,8,3)).astype(np.float32),
                    bf=rng.normal(size=(6,7,8,3)).astype(np.float32),sdf=rng.normal(size=(6,7,8)).astype(np.float32))
                np.savez(root/f"{i}.npz",**f)
                rows.append(dict(file=f"{i}.npz",sha256=hashlib.sha256((root/f"{i}.npz").read_bytes()).hexdigest(),
                    origin=[0,0,0],resolution=.04,shape=[6,7,8],split="train" if i<2 else "validation",family="lateral",difficulty=.4))
                fields.append({k:torch.tensor(v) for k,v in f.items()})
            (root/"bank.json").write_text(json.dumps(dict(complete=True,robot_actuation=False,scenes=rows)))
            bank=FieldBank(root)
            scene=torch.tensor([2,0,1,2]);points=torch.rand(4,11,3)*.4-.05
            actual=bank.sample(scene,points)
            for i,s in enumerate(scene.tolist()):
                expected=CatFieldSampler(fields[s],[0,0,0],.04).sample(points[i])
                for key in expected:
                    torch.testing.assert_close(actual[key][i],expected[key],rtol=0,atol=0)
            picked=bank.choose(100,"lateral")
            self.assertFalse((picked==2).any())
            self.assertTrue((bank.choose(10,"lateral","validation")==2).all())

    def test_batch_history_matches_single_env_and_resets_independently(self):
        from scipy.spatial.transform import Rotation
        names=[f"j{i}" for i in range(29)]
        contract=dict(joints=names[::-1],offset=[.1]*29,scale=[.5]*29)
        batch=BatchHistory(2,contract,names,"cpu")
        independent=[ProprioHistory(contract,names) for _ in range(2)]
        rng=np.random.default_rng(6)
        for step in range(12):
            q=rng.normal(size=(2,36))*.1;q[:,3:7]=Rotation.from_euler("z",[.2,-.3]).as_quat(scalar_first=True)
            v=rng.normal(size=(2,35))*.1;targets=rng.normal(size=(2,29))*.1
            rot=Rotation.from_quat(q[:,3:7],scalar_first=True).as_matrix()
            physical=dict(pelvis_rotation=torch.tensor(rot,dtype=torch.float32),gyro=torch.tensor(v[:,3:6],dtype=torch.float32),
                joint_pos=torch.tensor(q[:,7:],dtype=torch.float32),joint_vel=torch.tensor(v[:,6:],dtype=torch.float32),
                gravity=torch.tensor(-rot[:,2],dtype=torch.float32))
            if step==7:
                batch.ready[1]=False;independent[1]=ProprioHistory(contract,names)
            got=batch.sample(physical,torch.tensor(q[:,:3],dtype=torch.float32),torch.tensor(targets,dtype=torch.float32))
            for i in range(2):
                np.testing.assert_allclose(got[i],independent[i].sample(q[i],v[i],targets[i]),atol=2e-7,rtol=1e-6)

    def test_fast_pack_matches_verified_bridge(self):
        n=16;names=list(OBS_JOINTS)+[f"extra{i}" for i in range(6)]
        contract=dict(schema="cat-observation-action-contract-v1",action_joints=JOINTS,
            observation_joints=OBS_JOINTS,default_joint_positions={k:.1 for k in names})
        bridge=CatObservationBridge(contract,names)
        p=dict(joint_pos=torch.randn(n,29),joint_vel=torch.randn(n,29),
            pelvis_rotation=torch.eye(3).repeat(n,1,1),gyro=torch.randn(n,3),gravity=torch.randn(n,3))
        last=torch.randn(n,12);targets=torch.randn(n,12);cmd=torch.randn(n,4);phase=torch.randn(n,2)
        fields=dict(gf=torch.randn(n,11,3),bf=torch.randn(n,11,3),sdf=torch.randn(n,11,1))
        got=pack_cat(p,bridge.obs_ids,torch.tensor(bridge.default_obs),last,targets,cmd,phase,fields)
        expected=bridge.pack(**p,last_action=last,previous_targets=targets,command_world=cmd,
            foot_height=torch.full((n,1),.07),phase=phase,**fields)
        torch.testing.assert_close(got,expected,rtol=0,atol=0)


if __name__=="__main__":unittest.main()
