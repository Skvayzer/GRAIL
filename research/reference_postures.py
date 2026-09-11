"""Reconstruct the pinned reference with upstream FK; no actor or simulation.

Every reconstructed imported-collider centre must match the saved Isaac
reference sweep before modified postures may be evaluated. This is geometry,
not proof of dynamic balance or tracking by the released controller.
"""
from dataclasses import asdict
import json
from pathlib import Path

import joblib
import numpy as np
from omegaconf import OmegaConf
import torch

from artifacts import ROOT, verify_file, safe_path
from cat_scenes import sha256
from gear_sonic.research.body_envelope import Probe, world_probe_centers
from gear_sonic.research.scene_audit import read_reference


def reconstruct(reference_run, family="stair_p1"):
    from gear_sonic.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch
    reference_run = Path(reference_run).resolve()
    anchors, saved_centers, saved_radii, links = read_reference(reference_run)
    metadata = json.loads((reference_run/"residual_preflight.json").read_text())
    if not metadata["complete"] or not metadata["base_backbone_unchanged"]:
        raise ValueError("Reference posture reconstruction requires a completed frozen runtime preflight")
    probes = [Probe(**item) for item in metadata["oracle"]["probe_order"]]
    if [p.link for p in probes] != list(links) or not np.allclose([p.radius for p in probes], saved_radii, atol=1e-7, rtol=0):
        raise ValueError("Imported probe metadata differs from reference sweep")
    manifest = json.loads((ROOT/"data_manifest.json").read_text())
    candidates = [item for item in manifest["files"] if item["path"].startswith(f"data/{family}/robot/")]
    if len(candidates) != 1:
        raise ValueError("Exactly one pinned paired reference required")
    item = candidates[0]
    source = ROOT/"artifacts"/safe_path(item["path"])
    verify_file(source, item)
    checkpoint_cfg = ROOT/"artifacts/checkpoint/SONIC/models/terrain_release/config.yaml"
    config_item = next(item for item in manifest["files"] if item["path"].endswith("terrain_release/config.yaml"))
    verify_file(checkpoint_cfg, config_item)
    cfg = OmegaConf.load(checkpoint_cfg).manager_env.commands.motion.motion_lib_cfg
    cfg.asset.assetRoot = str(ROOT.parent/"imports/SONIC"/cfg.asset.assetRoot)
    model = Humanoid_Batch(cfg, device=torch.device("cpu"))
    # Same pinned artifact and loader used by the reproduced upstream baseline.
    data = joblib.load(source)
    if len(data) != 1:
        raise ValueError("Ambiguous motion artifact")
    raw = next(iter(data.values()))
    pose, trans = torch.as_tensor(raw["pose_aa"]).float(), torch.as_tensor(raw["root_trans_offset"]).float()
    result = model.fk_batch(pose[None], trans[None], return_full=True,
        fps=float(raw["fps"]), target_fps=50, interpolate_data=True)
    positions = result.global_translation[0]
    quaternions = result.global_rotation[0][..., [3, 0, 1, 2]]
    centers, radii = world_probe_centers(probes, model.body_names, positions, quaternions)
    if centers.shape != saved_centers.shape or not np.allclose(centers.numpy(), saved_centers, atol=2e-5, rtol=0):
        error = float(np.max(np.abs(centers.numpy()-saved_centers))) if centers.shape == saved_centers.shape else None
        raise ValueError(f"Upstream FK does not reproduce captured reference: {error}")
    if not np.allclose(positions[:, model.body_names.index("pelvis")].numpy(), anchors, atol=2e-5, rtol=0):
        raise ValueError("Reference root/frame differs from terrain-local recording")
    inventory = json.loads((reference_run/"cat_audit.json").read_text())["imported_colliders"]
    if any(item["type"] not in ("Sphere", "Capsule") for item in inventory):
        raise ValueError("Inscribed-sphere witness requires actual spheres/capsules, not arbitrary mesh approximations")
    inner_radii = torch.tensor([inventory[p.collision_index]["radius"] for p in probes])
    provenance = dict(schema="grail-cat-reference-posture-reconstruction-v1", source_motion=str(source),
        source_motion_sha256=sha256(source), source_config_sha256=sha256(checkpoint_cfg),
        source_mjcf_sha256=sha256(model.mjcf_file), reference_run=str(reference_run),
        reference_sweep_sha256=sha256(reference_run/"reference_sweep.npz"),
        imported_probe_order=[asdict(p) for p in probes], body_names=list(model.body_names),
        max_reference_probe_error_m=float(np.max(np.abs(centers.numpy()-saved_centers))),
        frames=len(centers), fps=50, dynamic_feasibility_verified=False)
    return dict(model=model, raw=raw, fk=result, probes=probes, centers=centers, radii=radii,
                inner_radii=inner_radii, inventory=inventory, anchors=anchors, provenance=provenance)
