"""Read-only, environment-local oracle observations for static replicated scenes."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .body_envelope import world_probe_centers
from .obstacle_observation import GuidanceSampler, ObstacleObservation
from .scene_audit import read_snapshot
from .terrain_surface import TerrainSurface
from .learner_state import environment_selection


class RuntimeObservation:
    def __init__(self, wrapper, cat_audit, run, *, challenge=None):
        self.env, self.cat = wrapper.env, cat_audit
        if not 1 <= self.env.num_envs <= 16:
            raise ValueError("Static replicated oracle currently supports 1..16 environments")
        run = Path(run)
        self.run = run
        self.failure_context = {}
        self.invalid_count = 0
        self.guidance_failures_recorded = 0
        vertices, faces, self.terrain = read_snapshot(run)
        if self.env.num_envs > 1 and self.terrain.get("replicas_verified") != list(range(1, self.env.num_envs)):
            raise ValueError("Every replicated terrain must have verified identical local geometry")
        layout_path, grid_path = run/"layout_audit.json", run/"layout_audit.npz"
        layout = json.loads(layout_path.read_text())
        admitted = (challenge is not None and challenge.layout_checked
                    and layout.get("accepted_for_training_challenge")
                    and layout.get("challenge_admission") == challenge.contract)
        if ((not layout["accepted_for_reference_diagnostic"] and not admitted) or layout["grid_file"] != grid_path.name
                or hashlib.sha256(grid_path.read_bytes()).hexdigest() != layout["grid_sha256"]):
            raise ValueError("Runtime oracle requires intact, accepted physical layout evidence")
        with np.load(grid_path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        self.surface = TerrainSurface(vertices, faces, ground_z=self.terrain["ground"]["height"], device=self.env.device)
        self.guidance = GuidanceSampler(layout, arrays, self.env.device)
        self.observer = ObstacleObservation(self.surface, self.cat.mesh, self.cat.placement, self.guidance)
        self.robot = self.env.scene["robot"]
        self.anchor = self.robot.body_names.index(wrapper.motion_command.cfg.anchor_body)
        self.contract = dict(schema="grail-cat-static-runtime-oracle-v1", sensor_realism=False,
            packet=self.observer.spec.manifest(), guidance=self.guidance.attachment.manifest(),
            probe_order=[asdict(p) for p in self.cat.probes], anchor_body=wrapper.motion_command.cfg.anchor_body,
            environment_coordinates="world positions minus each environment origin; shared local static scene",
            layout_sha256=hashlib.sha256(layout_path.read_bytes()).hexdigest(), grid_sha256=layout["grid_sha256"],
            terrain_sha256=self.terrain["sha256"], cat_files=self.cat.fields.meta["files"])

    def guidance_failure(self):
        """Leaving the task's supported route is an episode failure, not a crash.

        Called by Isaac's termination manager BEFORE rewards and pre-reset
        bootstrap. This does not make unknown space valid or waive geometry
        corruption. Other observation failures still abort and save diagnostics.
        """
        root = self.robot.data.body_pos_w[:, self.anchor]-self.env.scene.env_origins
        if not torch.isfinite(root).all():
            raise ValueError("Nonfinite root in guidance termination")
        support, _, known = self.surface.support_below(root, max_drop=2.)
        attachment = self.guidance.attachment.sample(root, support, known)
        failed = ~attachment["valid"]
        if failed.any() and self.guidance_failures_recorded < 8:
            self.guidance_failures_recorded += 1
            record = dict(schema="grail-cat-guidance-task-failure-v1",
                step=int(self.env.common_step_counter), failed=failed.tolist(), root=root.tolist(),
                support=[h if k else None for h, k in zip(support.tolist(), known.tolist())],
                support_known=known.tolist(),
                attachment={k: v.tolist() for k, v in attachment.items()})
            path = self.run/f"guidance_failure_{self.guidance_failures_recorded:06d}.json"
            path.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
            print("GUIDANCE_EPISODE_FAILURE "+json.dumps(record), flush=True)
        return failed

    def sample(self, env_mask=None):
        origin, obj = self.env.scene.env_origins, self.env.scene["object"].data
        expected_pos = origin+origin.new_tensor(self.terrain["live_position"])
        expected_q = origin.new_tensor(self.terrain["live_quaternion_wxyz"])
        if (not torch.allclose(obj.root_pos_w, expected_pos, atol=1e-4, rtol=0)
                or not torch.allclose((obj.root_quat_w*expected_q).sum(-1).abs(),
                                     torch.ones(self.env.num_envs, device=origin.device), atol=1e-4, rtol=0)):
            raise ValueError("Terrain moved relative to its static oracle snapshot")
        selected = environment_selection(env_mask, self.env.num_envs)
        root = self.robot.data.body_pos_w[selected, self.anchor]-origin[selected]
        quaternion = self.robot.data.body_quat_w[selected, self.anchor]
        centers, radii = world_probe_centers(self.cat.probes, self.robot.body_names,
            self.robot.data.body_pos_w[selected], self.robot.data.body_quat_w[selected])
        local_centers = centers-origin[selected, None, :]
        packet = self.observer.sample(root, quaternion, local_centers, radii)
        if not packet["valid"].all():
            self.invalid_count += 1
            stem = self.run/f"invalid_observation_{self.invalid_count:06d}"
            arrays = dict(root=root, quaternion=quaternion, centers=local_centers, radii=radii,
                          **{"valid_"+k: v for k, v in self.observer.last_validity.items()},
                          **{"attachment_"+k: v for k, v in self.guidance.last_attachment.items()})
            np.savez_compressed(stem.with_suffix(".npz"), **{
                k: v.detach().cpu().numpy() for k, v in arrays.items()})
            context = {k: v.detach().cpu().tolist() if isinstance(v, torch.Tensor) else v
                       for k, v in self.failure_context.items()}
            record = dict(schema="grail-cat-invalid-observation-v1", context=context,
                          validity={k: v.detach().cpu().tolist() for k, v in self.observer.last_validity.items()},
                          root=root.detach().cpu().tolist(), snapshot=stem.name+".npz")
            stem.with_suffix(".json").write_text(json.dumps(record, indent=2)+"\n")
            print("INVALID_OBSERVATION "+json.dumps(record), flush=True)
        return packet
