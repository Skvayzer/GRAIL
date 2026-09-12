"""Simulation-only CAT-to-GRAIL decoder distillation; no simulator starts here.

Flat CAT mode learns continuous motor tokens. Retention mode takes original
GRAIL reference tokens. The frozen decoder always outputs all 29 joint actions.
Upper-body nominal posture is an explicit prior, never a fabricated CAT label.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/"imports/SONIC"))
from gear_sonic.research.observation_shadow import state_hash
from gear_sonic.research.cat_teacher import JOINTS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_grail(context):
    from gear_sonic.trl.utils.common import custom_instantiate
    context = Path(context)
    meta = json.loads((context/"cat_teacher_shadow.json").read_text())
    path = context/meta["policy_config_file"]
    if sha(path) != meta["policy_config_sha256"]:
        raise ValueError("Changed GRAIL policy configuration")
    cfg = json.loads(path.read_text())
    env, algo = (OmegaConf.create(cfg[k]) for k in ("env_config", "algo_config"))
    actor = custom_instantiate(algo.actor, env_config=env, algo_config=algo,
        module_dim_dict=algo.get("module_dim", {}), backbone_kwargs={}, _resolve=False)
    checkpoint = context/"checkpoint/last.pt"
    provenance = json.loads((ROOT/"provenance.json").read_text())
    if sha(checkpoint) != provenance["checkpoint_sha256"]:
        raise ValueError("GRAIL checkpoint differs from pinned release")
    weights = torch.load(checkpoint, map_location="cpu", weights_only=False)
    parameters = weights.get("actor_model_state_dict", weights.get("policy_state_dict"))
    if parameters is None:
        raise ValueError("Missing actor in pinned checkpoint")
    actor.load_state_dict(parameters, strict=True)
    actor.eval().requires_grad_(False)
    if actor.running_mean_std is not None or state_hash(actor.actor_module) != meta["base_backbone_sha256"]:
        raise ValueError("Unexpected GRAIL normalization/backbone")
    with np.load(context/meta["packet_file"], allow_pickle=False) as packet:
        scale = packet["action_scale"][0, 0].copy()
        offset = packet["action_offset"][0, 0].copy()
    if sha(context/meta["packet_file"]) != meta["packet_sha256"]:
        raise ValueError("Changed source action contract")
    contract = dict(joints=meta["action_joint_names"], scale=scale.tolist(), offset=offset.tolist(),
        clip=meta["wrapper_action_clip"], backbone_sha256=meta["base_backbone_sha256"],
        context=str(context.resolve()), context_packet_sha256=meta["packet_sha256"])
    return actor, contract


class DistillStudent(nn.Module):
    def __init__(self, actor, contract, teacher):
        super().__init__()
        self.decoder = copy.deepcopy(actor.actor_module.decoders["g1_dyn"]).eval().requires_grad_(False)
        self.contract = contract
        self.legs = [contract["joints"].index(n) for n in JOINTS]
        self.upper = [i for i in range(29) if i not in self.legs]
        self.register_buffer("scale", torch.tensor(contract["scale"]))
        self.register_buffer("offset", torch.tensor(contract["offset"]))
        # Reuse pretrained CAT hidden features, not a new randomly initialized
        # avoidance feature extractor. Both teacher and copied trunk stay frozen.
        self.features = nn.Sequential(*sum(([copy.deepcopy(layer), nn.SiLU()]
            for layer in teacher.layers[:-1]), [])).eval().requires_grad_(False)
        self.adapter = nn.Sequential(nn.Linear(64+1029+64+1, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 64))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.register_buffer("mean", torch.zeros(1029))
        self.register_buffer("std", torch.ones(1029))

    def latent(self, obs, cat, base, mode):
        if obs.shape[-1] != 1029 or cat.shape[-1] != 162 or base.shape[-1] != 64:
            raise ValueError("Unexpected GRAIL/CAT/token contract")
        feature = self.features(cat)
        x = torch.cat((feature, ((obs-self.mean)/self.std).clamp(-10, 10), base, mode[:, None]), -1)
        # Continuous post-quantization correction, bounded for the first pilot.
        return base+2*torch.tanh(self.adapter(x))

    def forward(self, obs, cat, base, mode):
        latent = self.latent(obs, cat, base, mode)
        # g1_dyn BaseModule consumes concatenated [token_flattened, proprioception].
        raw = self.decoder(torch.cat((latent, obs), -1))
        limit = self.contract["clip"]
        raw = raw.clamp(-limit, limit)
        return raw*self.scale+self.offset

    def trainable(self):
        return self.adapter.parameters()

    def frozen_hash(self):
        return state_hash(self.decoder), state_hash(self.features)


def objective(model, batch):
    predicted = model(batch["obs"], batch["cat"], batch["base"], batch["mode"])
    flat = batch["mode"] > .5
    retention = ~flat
    zero = predicted.sum()*0
    legs = ((predicted[flat][:, model.legs]-batch["target"][flat][:, model.legs])**2).mean() if flat.any() else zero
    posture = ((predicted[flat][:, model.upper]-batch["target"][flat][:, model.upper])**2).mean() if flat.any() else zero
    retain = ((predicted[retention]-batch["target"][retention])**2).mean() if retention.any() else zero
    total = legs+.25*posture+retain
    return total, dict(leg_mse=float(legs.detach()), posture_mse=float(posture.detach()),
                       retention_mse=float(retain.detach()))


class ProprioHistory:
    """Pinned GRAIL policy term order, oldest-to-newest, 10 samples per term.

Flat scene uses a static virtual object at world origin: zero future position
delta, identity future orientation delta, current body-relative pose. This is
not the obstacle representation; CAT fields provide obstacle information.
"""
    def __init__(self, contract, native_joints):
        self.ids = [native_joints.index(n) for n in contract["joints"]]
        self.offset = np.array(contract["offset"])
        self.scale = np.array(contract["scale"])
        self.history = None

    def sample(self, qpos, qvel, previous_targets):
        from scipy.spatial.transform import Rotation
        r = Rotation.from_quat(qpos[3:7], scalar_first=True).as_matrix()
        values = (qvel[3:6], qpos[7:][self.ids]-self.offset, qvel[6:][self.ids],
                  (previous_targets[self.ids]-self.offset)/self.scale, -r[2])
        if self.history is None:
            self.history = [np.repeat(v[None], 10, axis=0) for v in values]
        else:
            self.history = [np.concatenate((old[1:], v[None])) for old, v in zip(self.history, values)]
        identity6 = np.eye(3)[:, :2].ravel()
        # PolicyCfg declaration order (not YAML defaults order): delta terms
        # precede object_pos_b/object_ori_b_6d.
        objects = np.r_[np.zeros(30), np.tile(identity6, 10), -r.T@qpos[:3], r.T[:, :2].ravel()]
        result = np.concatenate([h.ravel() for h in self.history]+[objects]).astype(np.float32)
        if result.shape != (1029,) or not np.isfinite(result).all():
            raise ValueError("Invalid reconstructed flat GRAIL observations")
        return result


def flat_field_admission(points, origin, shape, resolution, foot_contacts):
    """Native flat-floor boundary extension, NOT general out-of-map acceptance.

CAT's foot sites can penetrate its compliant z=0 floor by millimeters. Permit
native lower-Z clamping only for either foot with verified floor contact, at
most 2 cm below this KNOWN plane. XY/upper-Z/other sites remain strict.
    """
    points, origin = np.asarray(points), np.asarray(origin)
    if points.shape != (11, 3) or not np.isfinite(points).all() or abs(origin[2]) > 1e-9:
        raise ValueError("Expected 11 sites on the native known z=0 flat task")
    index = (points-origin)/resolution
    valid_axes = (index >= 0) & (index <= np.array(shape[:3])-1)
    exceptions = 0
    for row, contact in zip((3, 4), foot_contacts):
        if not valid_axes[row, 2] and -.02 <= points[row, 2] < 0 and contact:
            valid_axes[row, 2] = True
            exceptions += 1
    return bool(valid_axes.all()), exceptions
