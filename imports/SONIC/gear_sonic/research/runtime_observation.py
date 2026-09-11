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


class RuntimeObservation:
    def __init__(self, wrapper, cat_audit, run):
        self.env, self.cat = wrapper.env, cat_audit
        if not 1 <= self.env.num_envs <= 4:
            raise ValueError("Static replicated oracle currently supports 1..4 environments")
        run = Path(run)
        vertices, faces, self.terrain = read_snapshot(run)
        if self.env.num_envs > 1 and self.terrain.get("replicas_verified") != list(range(1, self.env.num_envs)):
            raise ValueError("Every replicated terrain must have verified identical local geometry")
        layout_path, grid_path = run/"layout_audit.json", run/"layout_audit.npz"
        layout = json.loads(layout_path.read_text())
        if (not layout["accepted_for_reference_diagnostic"] or layout["grid_file"] != grid_path.name
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

    def sample(self):
        origin, obj = self.env.scene.env_origins, self.env.scene["object"].data
        expected_pos = origin+origin.new_tensor(self.terrain["live_position"])
        expected_q = origin.new_tensor(self.terrain["live_quaternion_wxyz"])
        if (not torch.allclose(obj.root_pos_w, expected_pos, atol=1e-4, rtol=0)
                or not torch.allclose((obj.root_quat_w*expected_q).sum(-1).abs(),
                                     torch.ones(self.env.num_envs, device=origin.device), atol=1e-4, rtol=0)):
            raise ValueError("Terrain moved relative to its static oracle snapshot")
        root = self.robot.data.body_pos_w[:, self.anchor]-origin
        quaternion = self.robot.data.body_quat_w[:, self.anchor]
        centers, radii = world_probe_centers(self.cat.probes, self.robot.body_names,
            self.robot.data.body_pos_w, self.robot.data.body_quat_w)
        return self.observer.sample(root, quaternion, centers-origin[:, None, :], radii)
