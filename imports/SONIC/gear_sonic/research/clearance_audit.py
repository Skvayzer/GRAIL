"""Full-body *diagnostic* observations; not inserted into actor/critic inputs."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import torch

from .body_envelope import URDF, collision_probes, self_pair_mask, world_probe_centers
from .geometry import FIELD_VERSION, sphere_clearances
from .scenes import stair_side_clutter
from .usd_envelope import imported_collision_probes


class ClearanceAudit:
    def __init__(self, env, output):
        if env.config.get("research_clutter") != "stair_side_v1":
            raise ValueError("Audit requires its paired physical clutter configuration")
        self.env = env.env
        self.robot = self.env.scene["robot"]
        from isaaclab.sim.utils.stage import get_current_stage
        self.probes, self.collider_inventory = imported_collision_probes(
            get_current_stage(), "/World/envs/env_0/Robot", self.robot.body_names)
        if set(p.link for p in self.probes) != set(p.link for p in collision_probes()):
            raise ValueError("Imported collider body set differs from pinned URDF; review coverage")
        self.solids = stair_side_clutter()
        self.links = sorted({p.link for p in self.probes})
        self.self_mask = self_pair_mask(self.probes).to(self.env.device)
        self.output = Path(output)
        self.records = []
        self.minimum = {link: float("inf") for link in self.links}
        self.peak_force = {s.name: {link: 0.0 for link in self.links} for s in self.solids}
        self.peak_self_force = {link: {other: 0. for other in self.links if other != link} for link in self.links}
        self.skipped_reset_samples = 0

    def sample(self, actions, dones):
        if not torch.isfinite(actions).all():
            raise ValueError("Nonfinite policy actions")
        # Env.step may reset done environments before returning. Do not label
        # reset poses as terminal collision states; full terminal audit is M1-next.
        active = ~dones.reshape(-1).bool()
        self.skipped_reset_samples += int((~active).sum())
        if not active.any():
            return
        centers_w, radii = world_probe_centers(self.probes, self.robot.body_names,
                                              self.robot.data.body_pos_w, self.robot.data.body_quat_w)
        centers = centers_w-self.env.scene.env_origins[:, None, :]
        for solid in self.solids:
            body = self.env.scene["research_"+solid.name]
            expected_pos = centers.new_tensor(solid.center)+self.env.scene.env_origins
            expected_quat = centers.new_tensor((math.cos(solid.yaw/2), 0., 0., math.sin(solid.yaw/2)))
            if (not torch.allclose(body.data.root_pos_w, expected_pos, atol=1e-4)
                    or not torch.allclose((body.data.root_quat_w*expected_quat).sum(-1).abs(),
                                          torch.ones(self.env.num_envs, device=self.env.device), atol=1e-4)):
                raise ValueError(f"Physical clutter moved away from its field: {solid.name}")
        clearance = sphere_clearances(centers, radii, self.solids)
        if not torch.isfinite(clearance).all():
            raise ValueError("Nonfinite clearance observations")
        distances = torch.cdist(centers, centers)-radii[:, None]-radii[None, :]
        self_min = distances[:, self.self_mask].amin(-1)
        for link in self.links:
            indices = [i for i, p in enumerate(self.probes) if p.link == link]
            value = float(clearance[active][:, indices].amin())
            self.minimum[link] = min(self.minimum[link], value)
            history = self.env.scene["research_self_"+link].data.force_matrix_w_history
            if history is None or history.shape[-2] != len(self.links)-1 or not torch.isfinite(history).all():
                raise ValueError("Missing/nonfinite physical self-contact measurements")
            peak = torch.linalg.vector_norm(history[active], dim=-1).amax(dim=(0, 1, 2))
            for other, force in zip(self.peak_self_force[link], peak.tolist()):
                self.peak_self_force[link][other] = max(self.peak_self_force[link][other], force)
        for solid in self.solids:
            sensor = self.env.scene["research_"+solid.name+"_contacts"]
            history = sensor.data.force_matrix_w_history
            if history is None or history.shape[-2] != len(self.links) or not torch.isfinite(history).all():
                raise ValueError("Missing/nonfinite or incorrectly filtered physical contacts")
            peak = torch.linalg.vector_norm(history[active], dim=-1).amax(dim=(0, 1, 2))
            for link, force in zip(self.links, peak.tolist()):
                self.peak_force[solid.name][link] = max(self.peak_force[solid.name][link], force)
        self.records.append({"sample": len(self.records), "active_envs": int(active.sum()),
                             "min_clutter_clearance_m": float(clearance[active].amin()),
                             "min_self_cover_gap_m": float(self_min[active].amin())})

    def finish(self):
        report = {
            "schema": FIELD_VERSION, "simulation_only": True,
            "policy_changed": False, "policy_observation_changed": False, "rewards_changed": False,
            "geometry_source": "ideal added primitive clutter only; paired terrain SDF not implemented",
            "coverage": "conservative imported USD collision-primitive cover, NOT visual/real-robot mesh certification",
            "self_contact_metric": "sphere-cover gaps and independent pair-filtered PhysX normal forces; not interchangeable",
            "contact_sampling": "filtered normal forces with physics-step history, sampled at policy steps; reset samples skipped",
            "policy_step_s": self.env.step_dt,
            "urdf_sha256": hashlib.sha256(URDF.read_bytes()).hexdigest(),
            "body_names": self.robot.body_names, "joint_names": self.robot.joint_names,
            "probe_count": len(self.probes), "probes": [asdict(p) for p in self.probes],
            "imported_colliders": self.collider_inventory,
            "solids": [asdict(s) for s in self.solids],
            "min_clutter_clearance_by_link_m": self.minimum,
            "peak_clutter_normal_force_by_link_N": self.peak_force,
            "peak_self_normal_force_by_link_N": self.peak_self_force,
            "skipped_reset_samples": self.skipped_reset_samples, "samples": self.records,
            "valid_diagnostic_run": bool(self.records), "avoidance_training_ready": False,
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(f"Research clearance audit: {self.output}", flush=True)
