"""Opt-in articulated per-physics-step contact audit, including terminal steps.

Raw PhysX contact tensors are read before automatic reset. No policy, reward,
observation, action, reset or physics-step modifications. One environment and
static terrain only. Phase labels are reference-kinematic candidates, not truth.
"""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .contact_accounting import ContactLimits, PhysicsStepTap, bind_contact_names, classify_contact, reference_phase, sole_regions, unpack_contacts
from .scene_audit import read_reference, read_snapshot
from .terrain_snapshot import rotation_wxyz
from .terrain_surface import TerrainSurface


class ContactAudit:
    def __init__(self, wrapper, cat_audit, output):
        import omni.physics.tensors as tensors
        self.env, self.motion, self.cat = wrapper.env, wrapper.motion_command, cat_audit
        if self.env.num_envs != 1:
            raise ValueError("Contact audit requires one environment")
        self.output = Path(output)
        self.limits = ContactLimits()
        v, f, self.terrain = read_snapshot(self.output.parent)
        self.surface = TerrainSurface(v, f, ground_z=self.terrain["ground"]["height"], device=self.env.device)
        self.terrain_only = TerrainSurface(v, f, device=self.env.device)
        self.robot = self.env.scene["robot"]
        self.origin = self.env.scene.env_origins[0].cpu().numpy().copy()
        self.links = cat_audit.links
        self.partners = ["terrain", "ground", "cat"]
        paths = ["/World/envs/env_0/Robot/"+link for link in self.links]
        filters = [self.terrain["rigid_body_prim"], self.terrain["ground"]["prim"], "/World/envs/env_0/ResearchCAT/Obstacles"]
        self.physics_view = tensors.create_simulation_view("torch")
        self.physics_view.set_subspace_roots("/")
        self.view = self.physics_view.create_rigid_contact_view(paths, [filters]*len(paths), max_contact_data_count=8192)
        # Names, not guessed order, bind the raw buffers to semantic labels.
        names, filter_names = list(self.view.sensor_names), list(self.view.filter_names)
        if self.view.sensor_count != len(paths) or self.view.filter_count != 3:
            raise ValueError(f"Unresolved contact sensors/filters: {names}, {filter_names}")
        self.sensor_links = bind_contact_names(names, filter_names, paths, filters)
        self.body_indices = [self.robot.body_names.index(link) for link in self.sensor_links]
        self.sole = sole_regions(self.cat.probes, self.cat.report["imported_colliders"], self.limits)
        self.phase = self.reference_phases()
        self.steps, self.policy_steps, self.details, self.codes = [], [], [], []
        self.counts = Counter()
        self.by_link = {link: Counter() for link in self.links}
        self.peak = {link: {p: 0. for p in self.partners} for link in self.links}
        self.pending = 0
        self.done = False
        self.last_counter = int(self.env._sim_step_counter)
        self.tap = PhysicsStepTap(self.env.scene, self.capture)
        self.report = dict(schema="grail-cat-articulated-contact-audit-v1", simulation_only=True,
            policy_changed=False, observations_changed=False, rewards_changed=False, physics_steps_added=0,
            capture_point="after scene.update(dt=physics_dt), before reward/termination/reset",
            diagnostic_scope="first episode, one environment, static terrain/ground/CAT pairs",
            requested_sensor_paths=paths, requested_filter_paths=filters,
            sensors=names, filter_names=filter_names, limits=asdict(self.limits),
            physics_dt=self.env.physics_dt, decimation=self.env.cfg.decimation,
            phase_source="independent reference sole clearance/ankle speed heuristic; not annotated contact truth",
            phase_truth_verified=False, contact_permissions_granted=False, avoidance_training_ready=False,
            body_coverage="all imported collision-bearing links; no self contacts/friction/payload in this audit",
            terrain_hash=self.terrain["sha256"], cat_files=self.cat.fields.meta["files"],
            sole_regions={k: [a.tolist() for a in value] for k, value in self.sole.items()},
            contact_api="https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.RigidContactView.get_contact_data")

    def reference_phases(self):
        _, centers, _, _ = read_reference(self.output.parent)
        names = list(self.env.cfg.isaaclab_to_mujoco_mapping["isaaclab_joints"])
        motion_ids = self.motion.motion_ids.unique()
        if len(motion_ids) != 1:
            raise ValueError("One fixed paired reference required")
        steps = torch.arange(len(centers), device=self.env.device)
        speed = self.motion.motion_lib.get_body_lin_vel_w_full(motion_ids.expand(len(steps)), steps)
        result, diagnostics = {}, {}
        for link in self.sole:
            indices = [i for i, p in enumerate(self.cat.probes) if p.link == link]
            p = torch.as_tensor(centers[:, indices], device=self.env.device).clone()
            radii = p.new_tensor([self.cat.report["imported_colliders"][self.cat.probes[i].collision_index]["radius"] for i in indices])
            p[..., 2] -= radii
            rays = p.clone()
            rays[..., 2] += .12
            h, n, valid = self.surface.support_below(rays, max_drop=2.)
            gap = p[..., 2]-h
            valid &= n[..., 2] >= self.limits.min_up_normal
            # Missing footprint samples remain unknown, never guessed stance.
            known = valid.all(-1).cpu().numpy()
            gap = gap.amin(-1).cpu().numpy()
            velocity = torch.linalg.vector_norm(speed[:, names.index(link)], dim=-1).cpu().numpy()
            labels = [reference_phase(g, s, k, self.limits) for g, s, k in zip(gap, velocity, known)]
            # Conservative transition guard over one action interval.
            result[link] = [a if i+1 < len(labels) and a == labels[i+1] else "unknown" for i, a in enumerate(labels)]
            diagnostics[link] = dict(Counter(result[link]))
        self.phase_distribution = diagnostics
        return result

    def __enter__(self):
        self.tap.__enter__()
        return self

    def __exit__(self, kind, error, trace):
        self.tap.__exit__(kind, error, trace)
        self.save(error=str(error) if error is not None else None)

    def capture(self, dt, **kwargs):
        counter = int(self.env._sim_step_counter)
        if self.done or not np.isclose(dt, self.env.physics_dt) or counter != self.last_counter+1:
            raise RuntimeError("Unexpected/repeated physics update in contact audit")
        self.last_counter = counter
        self.pending += 1
        if self.pending > self.env.cfg.decimation:
            raise RuntimeError("More physics updates than the configured policy decimation")
        obj = self.env.scene["object"]
        if not np.allclose(obj.data.root_pos_w[0].cpu().numpy()-self.origin, self.terrain["live_position"], atol=1e-4):
            raise RuntimeError("Terrain moved during static contact audit")
        if not np.isclose(abs(np.dot(obj.data.root_quat_w[0].cpu().numpy(), self.terrain["live_quaternion_wxyz"])), 1., atol=1e-4):
            raise RuntimeError("Terrain rotated during static contact audit")
        raw = [value.cpu().numpy().copy() for value in self.view.get_contact_data(dt)]
        matrix = self.view.get_contact_force_matrix(dt).cpu().numpy().copy()
        contacts, error = unpack_contacts(raw, matrix)
        self.peak_error = max(getattr(self, "peak_error", 0.), error)
        frame = int((self.motion.motion_start_time_steps+self.motion.time_steps)[0])
        poses = self.robot.data.body_pos_w[0, self.body_indices].cpu().numpy()-self.origin
        quats = self.robot.data.body_quat_w[0, self.body_indices].cpu().numpy()
        rotations = [rotation_wxyz(q) for q in quats]
        points = contacts["point"]-self.origin
        query = torch.as_tensor(points, dtype=torch.float32, device=self.env.device)
        if len(points):
            td, tn, tv = self.terrain_only.distance(query)
            cd, cn, cv = self.cat.mesh.query(self.cat.placement.to_local(query))
            cn = self.cat.placement.vectors_to_world(cn)
            td, tn, tv, cd, cn, cv = [v.cpu().numpy() for v in (td, tn, tv, cd, cn, cv)]
        step_counts = Counter()
        for i in range(len(points)):
            si, pi = int(contacts["sensor"][i]), int(contacts["partner"][i])
            link, partner = self.sensor_links[si], self.partners[pi]
            local = (points[i]-poses[si]) @ rotations[si]
            phase = self.phase[link][frame] if link in self.phase and 0 <= frame < len(self.phase[link]) else "unknown"
            if partner == "ground":
                distance, normal, valid = abs(points[i, 2]-self.surface.ground_z), np.array([0., 0., 1.]), True
            elif partner == "terrain":
                # Query the actual paired terrain, never substitute nearby ground.
                distance, normal, valid = td[i], tn[i], bool(tv[i])
            else:
                distance, normal, valid = cd[i], cn[i], bool(cv[i])
            agreement = float(np.dot(normal, contacts["normal"][i])) if valid else float("nan")
            lo, hi = self.sole.get(link, (np.zeros(3), np.zeros(3)))
            code = classify_contact(partner, link in self.sole, phase, local, lo, hi,
                contacts["normal"][i], float(contacts["force"][i]), float(contacts["separation"][i]),
                float(distance) if valid else float("nan"), agreement, self.limits)
            if code not in self.codes:
                self.codes.append(code)
            self.counts[code] += 1
            self.by_link[link][code] += 1
            step_counts[code] += 1
            self.details.append([len(self.steps), len(self.policy_steps), frame, si, pi, contacts["force"][i],
                *points[i], *contacts["normal"][i], contacts["separation"][i], self.codes.index(code),
                *local, distance, agreement])
        for si, link in enumerate(self.sensor_links):
            for pi, partner in enumerate(self.partners):
                self.peak[link][partner] = max(self.peak[link][partner], float(np.linalg.norm(matrix[si, pi])))
        self.steps.append(dict(physics_counter=counter, policy_step=len(self.policy_steps), reference_frame=frame,
                               contact_count=len(points), classifications=dict(step_counts)))

    def policy_step(self, dones):
        if self.pending != self.env.cfg.decimation:
            raise RuntimeError("Incomplete pre-reset physics contact capture")
        self.done = bool(dones.reshape(-1)[0])
        self.policy_steps.append(dict(first_physics_sample=len(self.steps)-self.pending, physics_samples=self.pending,
            terminal=self.done, failure=bool(self.env.reset_terminated[0]), timeout=bool(self.env.reset_time_outs[0])))
        self.pending = 0

    def save(self, error=None):
        data_path = self.output.with_suffix(".npz")
        np.savez_compressed(data_path, contacts=np.asarray(self.details, dtype=np.float32).reshape(-1, 19))
        valid = (self.done and self.pending == 0 and bool(self.policy_steps) and error is None
                 and not self.counts.get("invalid_geometry", 0))
        self.report.update(capture_complete=valid, error=error, completed=self.done,
            policy_steps=len(self.policy_steps), physics_steps=len(self.steps),
            terminal_physics_samples=sum(p["physics_samples"] for p in self.policy_steps if p["terminal"]),
            detailed_contacts=len(self.details), classifications=dict(self.counts), classifications_by_link=self.by_link,
            peak_pair_normal_force_N=self.peak, max_force_reconstruction_error_N=getattr(self, "peak_error", 0.),
            reference_phase_distribution=self.phase_distribution, samples=self.steps, policy_outcomes=self.policy_steps,
            contact_file=data_path.name, contact_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
            sensor_link_order=self.sensor_links, partner_order=self.partners, classification_order=self.codes,
            contact_columns=["physics_sample", "policy_step", "reference_frame", "sensor", "partner", "force_N",
                "x", "y", "z", "nx", "ny", "nz", "separation_m", "classification", "local_x", "local_y", "local_z",
                "surface_error_m", "normal_agreement"],
            collision_free_certified=False)
        self.output.write_text(json.dumps(self.report, indent=2, allow_nan=False)+"\n")
        print(f"ARTICULATED_CONTACT_AUDIT {self.output} complete={valid}", flush=True)
