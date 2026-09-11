"""Reference-conditioned posture-avoidance signals, distinct from terrain support.

Only CAT clutter is a forbidden signed solid here. Open terrain is NOT treated
as solid/free space. Foot support remains governed by the reference-tracking
task; no unvalidated contact-phase labels grant permissions. No policy actions,
optimizer, robot API, or simulation startup lives in this module.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .body_envelope import world_probe_centers
from .scene_audit import read_reference


FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
LOWER = ("left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
         "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link")


@dataclass(frozen=True)
class AvoidanceSpec:
    clearance_m: float = .08
    contact_penalty_threshold_n: float = 1.
    contact_failure_threshold_n: float = 20.
    foot_world_failure_m: float = .20

    def __post_init__(self):
        if (not all(math.isfinite(v) and v > 0 for v in asdict(self).values())
                or self.contact_failure_threshold_n <= self.contact_penalty_threshold_n):
            raise ValueError("Positive finite avoidance thresholds and ordered contact limits required")


def avoidance_signals(gaps, probe_links, contact_peaks, previous_root, root, goal, dt, spec=AvoidanceSpec()):
    """Per-link clearance avoids weighting links by number of cover spheres.

Progress is reduction in named-root XY distance to a fixed episode goal, not
hand motion. It telescopes over an episode, so moving back cancels moving out.
Returns rates; Isaac multiplies reward terms by the control timestep.
"""
    n = len(root)
    if (gaps.ndim != 2 or gaps.shape != (n, len(probe_links)) or not probe_links
            or root.shape != (n, 3) or previous_root.shape != root.shape or goal.shape != root.shape
            or contact_peaks.shape != (n, len(set(probe_links)))
            or not math.isfinite(dt) or dt <= 0
            or any(not torch.isfinite(v).all() for v in (gaps, contact_peaks, root, previous_root, goal))
            or (contact_peaks < 0).any()):
        raise ValueError("Invalid avoidance signal geometry/contact/step inputs")
    names = sorted(set(probe_links))
    link_gaps = torch.stack([gaps[:, [i for i, link in enumerate(probe_links) if link == name]].amin(1)
                             for name in names], 1)
    # Capped at four only for deeply penetrating states; physical-contact failure
    # is independent of this differentiable shaping term.
    penalty = ((spec.clearance_m-link_gaps).clamp_min(0)/spec.clearance_m).square().clamp_max(4).mean(1)
    contact = (contact_peaks > spec.contact_penalty_threshold_n).float().mean(1)
    failure = (contact_peaks > spec.contact_failure_threshold_n).any(1)
    before = torch.linalg.vector_norm(previous_root[:, :2]-goal[:, :2], dim=1)
    after = torch.linalg.vector_norm(root[:, :2]-goal[:, :2], dim=1)
    return dict(clearance_penalty=penalty, contact_penalty=contact, contact_failure=failure,
                root_progress=(before-after)/dt, goal_distance=after,
                link_gaps=link_gaps, min_gap=link_gaps.amin(1), contact_peak=contact_peaks.amax(1))


def foot_world_error(reference, actual):
    if (reference.ndim != 3 or reference.shape[1:] != (2, 3) or actual.shape != reference.shape
            or not torch.isfinite(reference).all() or not torch.isfinite(actual).all()):
        raise ValueError("Two finite world-frame foot poses required")
    return torch.linalg.vector_norm(reference-actual, dim=-1).amax(1)


def failure_event_rate(terminated, dt):
    if terminated.ndim != 1 or terminated.dtype != torch.bool or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Boolean per-environment termination and positive timestep required")
    return terminated.float()/dt


class AvoidanceTask:
    """Explicit begin-step lifecycle; shared read-only reward/termination cache."""
    def __init__(self, wrapper, cat_audit, run, spec=AvoidanceSpec()):
        self.env, self.motion, self.cat, self.spec = wrapper.env, wrapper.motion_command, cat_audit, spec
        self.run = Path(run)
        self.robot = self.env.scene["robot"]
        self.anchor = self.robot.body_names.index(self.motion.cfg.anchor_body)
        self.foot_indices = [list(self.motion.cfg.body_names).index(n) for n in FEET]
        self.probe_links = [p.link for p in self.cat.probes]
        self.links = sorted(set(self.probe_links))
        anchors, _, _, _ = read_reference(self.run)
        index = np.linalg.norm(anchors[:, :2]-anchors[0, :2], axis=1).argmax()
        self.goal = torch.as_tensor(anchors[index], device=self.env.device).expand(self.env.num_envs, -1).clone()
        self.rows, self.outcomes, self.expected_counter, self.cache = [], [], None, None
        self.report = dict(schema="grail-cat-posture-avoidance-task-v1", simulation_only=True,
            spec=asdict(spec), collision_links=self.links, probe_links=self.probe_links,
            goals_env_local=self.goal.tolist(), support_mode="paired reference footholds, not a free-detour controller",
            terrain_contact_permission="no new phase/terrain exemptions; CAT prohibited for every collision-bearing link",
            reward_units="rates integrated by Isaac step_dt; failure penalty is one event cost",
            success_claim=False, oracle_fields=True, thresholds_validated_for_hardware=False)

    def root(self):
        return self.robot.data.body_pos_w[:, self.anchor]-self.env.scene.env_origins

    def before_step(self):
        if len(self.rows) != len(self.outcomes):
            raise ValueError("Previous avoidance step has no recorded outcome")
        self.previous_root = self.root().detach().clone()
        self.expected_counter = int(self.env.common_step_counter)+1
        self.cache = None

    def measure(self):
        if self.expected_counter is None or int(self.env.common_step_counter) != self.expected_counter:
            raise ValueError("Avoidance terms require an explicit pre-action begin-step")
        if self.cache is not None:
            return self.cache
        centers, radii = world_probe_centers(self.cat.probes, self.robot.body_names,
            self.robot.data.body_pos_w, self.robot.data.body_quat_w)
        gaps = self.cat.gaps(centers-self.env.scene.env_origins[:, None], radii)
        peaks = []
        for name in self.links:
            history = self.env.scene["research_cat_contact_"+name].data.force_matrix_w_history
            if (history is None or history.ndim != 5 or history.shape[0] != self.env.num_envs
                    or history.shape[1] < self.env.cfg.decimation or history.shape[2:] != (1, 1, 3)
                    or not torch.isfinite(history).all()):
                raise ValueError("Missing full-interval, named link-to-CAT contact history")
            peaks.append(torch.linalg.vector_norm(history[:, :self.env.cfg.decimation], dim=-1).flatten(1).amax(1))
        values = avoidance_signals(gaps, self.probe_links, torch.stack(peaks, 1), self.previous_root,
            self.root(), self.goal, self.env.step_dt, self.spec)
        values["foot_world_error"] = foot_world_error(self.motion.body_pos_w[:, self.foot_indices],
                                                      self.motion.robot_body_pos_w[:, self.foot_indices])
        self.cache = values
        self.rows.append({k: v.detach().cpu().numpy().copy() for k, v in values.items()})
        return values

    def outcome(self, dones):
        if self.cache is None or len(self.rows) != len(self.outcomes)+1:
            raise ValueError("Avoidance terms were not measured before reset")
        self.outcomes.append(dict(dones=dones.reshape(-1).tolist(),
            terminated=self.env.reset_terminated.tolist(), truncated=self.env.reset_time_outs.tolist()))
        self.expected_counter = None

    def save(self):
        if not self.rows or len(self.rows) != len(self.outcomes):
            raise ValueError("Incomplete avoidance reward/termination capture")
        path = self.run/"avoidance_task.npz"
        np.savez_compressed(path, **{k: np.stack([row[k] for row in self.rows]) for k in self.rows[0]})
        self.report.update(steps=len(self.rows), outcomes=self.outcomes,
            data_file=path.name, data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            min_clutter_gap_m=min(float(row["min_gap"].min()) for row in self.rows),
            peak_cat_contact_n=max(float(row["contact_peak"].max()) for row in self.rows),
            max_foot_world_error_m=max(float(row["foot_world_error"].max()) for row in self.rows))
        (self.run/"avoidance_task.json").write_text(json.dumps(self.report, indent=2, allow_nan=False)+"\n")
