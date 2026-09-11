"""Pre-action shadow observations and frozen-clone checkpoint parity in Isaac.

Only the untouched actor's action reaches env.step. A separate frozen actor
clone evaluates zero residuals; shadow predictions are recorded, never returned
as actions. One single-episode, headless, static-scene diagnostic only.
"""
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .body_envelope import world_probe_centers
from .obstacle_adapter import ObstacleAdapter
from .obstacle_observation import GuidanceSampler, ObstacleObservation, ObservationSpec
from .scene_audit import read_snapshot
from .terrain_surface import TerrainSurface


def state_hash(module):
    result = hashlib.sha256()
    for key, tensor in sorted(module.state_dict().items()):
        result.update(json.dumps([key, list(tensor.shape), str(tensor.dtype)]).encode())
        result.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


class ObservationShadow:
    def __init__(self, wrapper, cat_audit, policy, output):
        self.env, self.cat, self.policy = wrapper.env, cat_audit, policy
        if self.env.num_envs != 1 or policy.training:
            raise ValueError("Shadow observation audit requires one environment and evaluation actor")
        self.output = Path(output)
        v, f, self.terrain = read_snapshot(self.output.parent)
        self.surface = TerrainSurface(v, f, ground_z=self.terrain["ground"]["height"], device=self.env.device)
        layout_path = self.output.with_name("layout_audit.json")
        layout = json.loads(layout_path.read_text())
        grid_path = self.output.with_name("layout_audit.npz")
        if (not layout["accepted_for_reference_diagnostic"] or layout["grid_file"] != grid_path.name
                or hashlib.sha256(grid_path.read_bytes()).hexdigest() != layout["grid_sha256"]):
            raise ValueError("Shadow requires a passed physical layout preflight with intact grid")
        with np.load(grid_path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        self.guidance = GuidanceSampler(layout, arrays, self.env.device)
        self.observer = ObstacleObservation(self.surface, self.cat.mesh, self.cat.placement, self.guidance)
        self.robot = self.env.scene["robot"]
        self.anchor = self.robot.body_names.index(wrapper.motion_command.cfg.anchor_body)
        self.clone = copy.deepcopy(policy).eval().requires_grad_(False)
        self.backbone_hash = state_hash(policy.actor_module)
        if state_hash(self.clone.actor_module) != self.backbone_hash:
            raise ValueError("Frozen shadow clone differs from restored backbone")
        self.adapter = ObstacleAdapter(len(self.cat.probes), policy.actor_module.token_total_dim).to(self.env.device).eval()
        self.adapter_hash = state_hash(self.adapter)
        self.samples, self.outcomes, self.packets = [], [], []
        self.complete = False
        self.gradient = None
        self.report = dict(schema="grail-cat-observation-shadow-v1", simulation_only=True,
            action_source="original policy.action_mean only; shadow.sample returns None",
            adapter_applied_to_environment=False, original_actor_inputs_changed=False, rewards_changed=False,
            policy_training=False, optimizer_steps=0, packet_spec=self.observer.spec.manifest(),
            guidance_contract=self.guidance.attachment.manifest(),
            probe_order=[asdict(p) for p in self.cat.probes], anchor_body=wrapper.motion_command.cfg.anchor_body,
            latent_dim=self.adapter.token_dim, latent_bound=self.adapter.bound, latent_mode="post_quantization",
            gradient_test="frozen clone only, autograd.grad on adapter head; no optimizer or parameter update",
            base_backbone_sha256=self.backbone_hash, adapter_sha256=self.adapter_hash,
            layout_sha256=hashlib.sha256(layout_path.read_bytes()).hexdigest(), grid_sha256=layout["grid_sha256"],
            terrain_sha256=self.terrain["sha256"], cat_files=self.cat.fields.meta["files"],
            capture_timing="immediately before env.step, including the action preceding terminal reset",
            contact_permissions_granted=False, avoidance_training_ready=False, sensor_realism=False)

    def __enter__(self):
        return self

    def sample(self, buffered_obs, live_actions):
        if self.complete or len(self.samples) != len(self.outcomes) or len(self.samples) >= 500:
            raise ValueError("Shadow sample must pair with exactly one subsequent environment step")
        start = time.perf_counter()
        # Snapshot-relative geometry is invalid if terrain moves or resets.
        obj = self.env.scene["object"].data
        origin = self.env.scene.env_origins
        expected_pos = origin+origin.new_tensor(self.terrain["live_position"])
        expected_q = origin.new_tensor(self.terrain["live_quaternion_wxyz"])
        if (not torch.allclose(obj.root_pos_w, expected_pos, atol=1e-4, rtol=0)
                or not torch.allclose((obj.root_quat_w*expected_q).sum(-1).abs(), torch.ones(1, device=origin.device), atol=1e-4, rtol=0)):
            raise ValueError("Terrain moved relative to shadow oracle snapshot")
        root = self.robot.data.body_pos_w[:, self.anchor]-origin
        quaternion = self.robot.data.body_quat_w[:, self.anchor]
        centers, radii = world_probe_centers(self.cat.probes, self.robot.body_names,
            self.robot.data.body_pos_w, self.robot.data.body_quat_w)
        centers = centers-origin[:, None, :]
        packet = self.observer.sample(root, quaternion, centers, radii)
        # The real actor already ran. Clone its current observation buffer, do
        # not call rollout again, sample a distribution, or append history.
        inputs = {k: v.detach().clone() for k, v in buffered_obs.items()}
        pristine = {k: v.clone() for k, v in inputs.items()}
        cpu_rng = torch.get_rng_state().clone()
        gpu_rng = torch.cuda.get_rng_state(self.env.device).clone() if root.is_cuda else None
        residual = self.adapter(packet)
        if torch.count_nonzero(residual):
            raise ValueError("Shadow prototype must remain zero-initialized; no trained residual allowed")
        baseline = self.clone.forward(inputs)[:, -1]
        candidate = self.clone.forward(inputs, latent_residual=residual,
                                       latent_residual_mode="post_quantization")[:, -1]
        if not all(torch.isfinite(x).all() for x in (baseline, candidate, live_actions)):
            raise ValueError("Nonfinite checkpoint parity inputs/outputs")
        if not torch.equal(baseline, candidate) or not torch.equal(baseline, live_actions):
            raise ValueError("Zero residual / frozen clone changed the released actor action mean")
        if self.gradient is None and packet["valid"].all():
            with torch.enable_grad():
                value = self.adapter(packet)
                action = self.clone.forward(inputs, latent_residual=value, latent_residual_mode="post_quantization")
                gradient = torch.autograd.grad(action.square().sum(), self.adapter.head.weight)[0]
            if not torch.isfinite(gradient).all() or torch.linalg.vector_norm(gradient) <= 0:
                raise ValueError("Latent residual has no finite gradient through the frozen decoder")
            self.gradient = dict(step=len(self.samples), head_gradient_norm=float(gradient.norm()),
                                 frozen_actor_parameter_gradients=any(p.grad is not None for p in self.clone.parameters()))
            if self.gradient["frozen_actor_parameter_gradients"]:
                raise ValueError("Frozen actor received parameter gradients")
        if (not torch.equal(cpu_rng, torch.get_rng_state()) or
                (gpu_rng is not None and not torch.equal(gpu_rng, torch.cuda.get_rng_state(self.env.device)))):
            raise ValueError("Shadow forward consumed the rollout RNG stream")
        if any(not torch.equal(pristine[k], v) or not torch.equal(v, buffered_obs[k]) for k, v in inputs.items()):
            raise ValueError("Shadow forward mutated actor observations")
        self.packets.append({**{k: v.detach().cpu().numpy().copy()[0] for k, v in packet.items()},
            "root": root.detach().cpu().numpy().copy()[0], "quaternion": quaternion.detach().cpu().numpy().copy()[0],
            "probe_centers": centers.detach().cpu().numpy().copy()[0],
            "baseline_action": baseline.detach().cpu().numpy().copy()[0],
            "zero_residual_action": candidate.detach().cpu().numpy().copy()[0],
            "live_action": live_actions.detach().cpu().numpy().copy()[0],
            "residual": residual.detach().cpu().numpy().copy()[0]})
        self.samples.append(dict(step=len(self.samples), valid=bool(packet["valid"][0]),
            guidance_valid=bool(packet["guidance"][0, 7]),
            guidance_target_cell=self.guidance.last_attachment["target_cell"][0].tolist(),
            guidance_connector_xy_m=float(self.guidance.last_attachment["connector_xy"][0]),
            guidance_covered_cells=int(self.guidance.last_attachment["covered_cells"][0]),
            volume_geometry_valid_fraction=float(packet["volume"][:, 2:].mean()),
            parity_max_abs_error=float((candidate-live_actions).abs().max()),
            processing_ms=(time.perf_counter()-start)*1000.))
        # Intentionally no action/packet return path to the simulator.

    def outcome(self, dones):
        if len(self.samples) != len(self.outcomes)+1:
            raise ValueError("Shadow outcome lacks a corresponding pre-action sample")
        done = bool(dones.reshape(-1)[0])
        self.outcomes.append(dict(step=len(self.outcomes), terminal=done,
                                 failure=bool(self.env.reset_terminated.reshape(-1)[0])))
        self.complete = done

    def __exit__(self, kind, error, traceback):
        backbone_same = state_hash(self.policy.actor_module) == self.backbone_hash == state_hash(self.clone.actor_module)
        adapter_same = state_hash(self.adapter) == self.adapter_hash
        valid = bool(self.complete and self.samples and len(self.samples) == len(self.outcomes)
                     and self.gradient and backbone_same and adapter_same and error is None
                     and not any(o["failure"] for o in self.outcomes))
        if self.packets:
            arrays = {k: np.stack([packet[k] for packet in self.packets]) for k in self.packets[0]}
            np.savez_compressed(self.output.with_suffix(".npz"), **arrays)
            self.report.update(packet_file=self.output.with_suffix(".npz").name,
                packet_sha256=hashlib.sha256(self.output.with_suffix(".npz").read_bytes()).hexdigest())
        self.report.update(complete=valid, error=str(error) if error else None,
            samples=self.samples, outcomes=self.outcomes, frames=len(self.samples),
            valid_frames=sum(s["valid"] for s in self.samples), gradient_check=self.gradient,
            base_backbone_unchanged=backbone_same, adapter_unchanged=adapter_same,
            physics_steps_added=0, actor_action_parity_exact=bool(self.samples) and
            all(s["parity_max_abs_error"] == 0. for s in self.samples))
        self.output.write_text(json.dumps(self.report, indent=2, allow_nan=False)+"\n")
        print(f"OBSTACLE_OBSERVATION_SHADOW {self.output} complete={valid}", flush=True)
        if error is None and not valid:
            raise ValueError("Incomplete obstacle observation/parity audit")
        return False
