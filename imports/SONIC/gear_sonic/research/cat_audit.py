"""Opt-in CAT/GRAIL integration and sampled-reference rejection, not learning."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .body_envelope import world_probe_centers
from .cat_geometry import CatFields, Placement, cat_mesh_arrays
from .mesh_distance import ClosedMeshDistance, reference_clearance_summary
from .usd_envelope import imported_collision_probes


class CatAudit:
    def __init__(self, wrapper, output):
        from pxr import UsdGeom, UsdPhysics
        from isaaclab.sim.utils.stage import get_current_stage
        self.env = wrapper.env
        self.robot = self.env.scene["robot"]
        cfg = wrapper.config
        self.placement = Placement(tuple(cfg.research_cat_translation), float(cfg.research_cat_yaw))
        self.fields = CatFields(cfg.research_cat_scene, self.placement, self.env.device)
        vertices, faces = cat_mesh_arrays(cfg.research_cat_scene)
        self.mesh = ClosedMeshDistance(vertices, faces, self.env.device)
        stage = get_current_stage()
        self.probes, inventory = imported_collision_probes(stage, "/World/envs/env_0/Robot", self.robot.body_names)
        self.links = sorted(set(p.link for p in self.probes))
        self.output = Path(output)
        self.minimum = {link: None for link in self.links}
        self.peak_force = {link: 0. for link in self.links}
        self.samples = []
        self.reset_samples = 0
        self.completed = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        self.failed = torch.zeros_like(self.completed)
        self.timeouts = torch.zeros_like(self.completed)
        for i in range(self.env.num_envs):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/ResearchCAT/Obstacles")
            if UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() != "none":
                raise ValueError("Physical CAT geometry is convexified")
            actual = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
            expected = self.placement.vectors_to_world(torch.eye(3)).numpy()
            origin = self.env.scene.env_origins[i].cpu().numpy()
            if not (np.allclose(actual[:3, :3], expected, atol=1e-5) and
                    np.allclose(actual[3, :3], np.array(self.placement.translation)+origin, atol=1e-4)):
                raise ValueError("Physical CAT transform differs from its field placement")
        self.report = dict(schema="grail-cat-reference-audit-v1", simulation_only=True,
            diagnostic_scope="first episode per environment only, not subsequent GUI replays",
            policy_changed=False, observations_changed=False, rewards_changed=False,
            source_scene=str(self.fields.directory), source_files=self.fields.meta["files"],
            placement=asdict(self.placement), probe_count=len(self.probes), imported_colliders=inventory,
            coverage="conservative cover of imported robot colliders, not visual/real-robot geometry",
            distance_contract="signed distance to exact closed CAT mesh minus probe radius; NOT terrain distance",
            field_contract="CAT voxel fields are unknown outside their sample domain; mesh exterior is a separate query",
            contact_sampling="policy-step physics history; auto-reset samples excluded; terminal contacts not fully audited",
            terrain_support_verified=False, avoidance_training_ready=False)
        self.report["reference"] = self.reference_check(wrapper.motion_command)
        self.report["preflight_accepted"] = self.report["reference"]["sampled_reference_clear"]
        self.save()
        if not self.report["preflight_accepted"]:
            raise RuntimeError("CAT intersects/approaches the reference collider cover. No policy steps allowed; inspect cat_audit.json")

    def gaps(self, centers, radii):
        distance, _, valid = self.mesh.query(self.placement.to_local(centers))
        if not valid.all():
            raise ValueError("Mesh distance unavailable: do not treat unknown as free")
        return distance-radii

    def reference_check(self, motion):
        names = list(self.env.cfg.isaaclab_to_mujoco_mapping["isaaclab_joints"])
        ids = motion.motion_ids.unique()
        if len(ids) != 1:
            raise ValueError("Bounded reference check currently requires a single paired reference")
        total = int(motion.motion_lib.get_time_step_total(ids)[0])
        values, field_valid, root_positions, probe_positions = [], [], [], []
        for start in range(0, total, 128):
            steps = torch.arange(start, min(total, start+128), device=self.env.device)
            batch_ids = ids.expand(len(steps))
            pos = motion.motion_lib.get_body_pos_w_full(batch_ids, steps)
            quat = motion.motion_lib.get_body_quat_w_full(batch_ids, steps)
            centers, radii = world_probe_centers(self.probes, names, pos, quat)
            probe_positions.append(centers.cpu())
            values.append(self.gaps(centers, radii))
            field_valid.append(self.fields.sample(centers)["valid"])
            root_positions.append(pos[:, names.index(motion.cfg.anchor_body)])
        result = reference_clearance_summary(torch.cat(values), [p.link for p in self.probes])
        roots = torch.cat(root_positions)
        sweep = self.output.with_name("reference_sweep.npz")
        np.savez_compressed(sweep, centers=torch.cat(probe_positions).numpy(),
                            radii=radii.cpu().numpy(), anchors=roots.cpu().numpy(),
                            links=np.array([p.link for p in self.probes]))
        result.update(field_valid_fraction=float(torch.cat(field_valid).float().mean()),
                      anchor_xyz_min=roots.amin(0).tolist(), anchor_xyz_max=roots.amax(0).tolist(),
                      reference_body_names=names,
                      sweep_file=sweep.name, sweep_sha256=hashlib.sha256(sweep.read_bytes()).hexdigest())
        return result

    def sample(self, actions, dones):
        if not torch.isfinite(actions).all():
            raise ValueError("Nonfinite actions")
        done = dones.reshape(-1).bool()
        active = ~(done | self.completed)
        first_done = done & ~self.completed
        # Isaac's reset flags are the returned pre-reset episode outcome, unlike
        # the already-reset body poses/contact buffers. Keep success distinct
        # from merely reaching *some* termination.
        self.failed |= first_done & self.env.reset_terminated.bool()
        self.timeouts |= first_done & self.env.reset_time_outs.bool()
        self.completed |= done
        self.reset_samples += int(done.sum())
        if not active.any():
            return
        centers, radii = world_probe_centers(self.probes, self.robot.body_names,
            self.robot.data.body_pos_w, self.robot.data.body_quat_w)
        centers -= self.env.scene.env_origins[:, None, :]
        gaps = self.gaps(centers, radii)
        fields = self.fields.sample(centers)
        for link in self.links:
            idx = [i for i, p in enumerate(self.probes) if p.link == link]
            value = float(gaps[active][:, idx].amin())
            self.minimum[link] = value if self.minimum[link] is None else min(value, self.minimum[link])
            history = self.env.scene["research_cat_contact_"+link].data.force_matrix_w_history
            if history is None or history.shape[-2] != 1 or not torch.isfinite(history).all():
                raise ValueError("Missing or invalid link-to-CAT physical contact sensor")
            self.peak_force[link] = max(self.peak_force[link], float(torch.linalg.vector_norm(history[active], dim=-1).amax()))
        self.samples.append(dict(step=len(self.samples), active_envs=int(active.sum()),
            min_clearance_m=float(gaps[active].amin()), field_valid_fraction=float(fields["valid"][active].float().mean())))

    def save(self):
        self.report.update(samples=self.samples, min_clearance_by_link_m=self.minimum,
                           peak_CAT_normal_force_by_link_N=self.peak_force, skipped_reset_samples=self.reset_samples,
                           completed_envs=self.completed.tolist(), failure_terminated_envs=self.failed.tolist(),
                           timeout_envs=self.timeouts.tolist(), valid_diagnostic_run=bool(self.samples),
                           rollout_completed_without_failure=bool(self.completed.all() and not self.failed.any()))
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(self.report, indent=2, allow_nan=False)+"\n")

    def finish(self):
        self.save()
        print(f"CAT reference/physics audit: {self.output}", flush=True)
