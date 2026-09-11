#!/usr/bin/env python3
"""Inspect only the hash-pinned official checkpoint; no simulation or control."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from artifacts import ROOT, PROVENANCE, verify_file


def main():
    import torch
    import yaml
    manifest = json.loads((ROOT / "data_manifest.json").read_text())
    item = next(x for x in manifest["files"] if x["path"] == PROVENANCE["checkpoint_path"])
    if item["hash"] != PROVENANCE["checkpoint_sha256"] or item["hash_kind"] != "sha256":
        raise ValueError("Not the pinned official checkpoint")
    path = ROOT / "artifacts" / item["path"]
    verify_file(path, item)
    # The upstream checkpoint includes a serialized TrainerState, not just tensors.
    # Never expose this loader as a general arbitrary-file inspection endpoint.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    actor = checkpoint.get("actor_model_state_dict", checkpoint.get("policy_state_dict"))
    if not actor:
        raise ValueError("Checkpoint has no actor state")
    shapes = {}
    for key, tensor in actor.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Non-tensor actor state: {key}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite actor tensor: {key}")
        shapes[key] = list(tensor.shape)
    config = yaml.safe_load(path.with_name("config.yaml").read_text())
    urdf = ROOT.parent / "imports/SONIC/gear_sonic/data/assets/robot_description/urdf/g1/main_nodex.urdf"
    joints = [j.attrib["name"] for j in ET.parse(urdf).getroot().findall("joint") if j.attrib["type"] != "fixed"]
    report = {
        "simulation_started": False, "robot_actuation": False,
        "checkpoint_sha256": item["hash"], "checkpoint_keys": list(checkpoint),
        "actor_tensor_count": len(shapes), "actor_parameter_elements": sum(t.numel() for t in actor.values()),
        "actor_tensors_finite": True, "tensor_shapes": shapes,
        "urdf_movable_joint_count": len(joints), "urdf_joint_names_not_runtime_order": joints,
        "robot_config": config["manager_env"]["config"]["robot"],
        "proprioception_history": config.get("actor_prop_history_length"),
        "simulation_dt": config["manager_env"]["config"].get("sim_dt"),
        "decimation": config["manager_env"]["config"].get("decimation"),
        "checkpoint_restoration_in_simulator": "not_tested_by_this_audit",
    }
    output = ROOT / "runs/checkpoint_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"tensor_shapes", "urdf_joint_names_not_runtime_order"}}, indent=2))
    print(f"Complete report: {output}")


if __name__ == "__main__":
    main()
