"""Read-only CAT scene fields and exact-mesh diagnostic queries.

The voxel field is unknown outside its sampled domain. Mesh distance is a
separate query against the complete exported clutter, not a claim about terrain
or unknown sensor space. All coordinates are metres, Z up, WXYZ rotations.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .geometry import sample_xyz_grid


def verify_scene_files(directory):
    directory = Path(directory)
    data = json.loads((directory/"scene.json").read_text())
    if data["schema"] != "cat-isaac-scene-v1":
        raise ValueError("Unknown CAT scene schema")
    expected_files = {"obs.npy", "sdf.npy", "bf.npy", "gf.npy", "travel.npy", "scene.usda"}
    if "role_provenance" in data:
        if (data["scene"] != "random" or data["role_provenance"].get("schema") not in ("cat-random-role-trace-v1", "cat-random-role-trace-v2")
                or data["role_provenance"].get("file") != "role_trace.npz"):
            raise ValueError("Unvalidated random role provenance")
        expected_files.add("role_trace.npz")
    if set(data["files"]) != expected_files:
        raise ValueError("Unexpected scene files")
    for name, expected in data["files"].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Scene checksum mismatch: {name}")
    return data


@dataclass(frozen=True)
class Placement:
    translation: tuple = (0., 0., 0.)
    yaw: float = 0.

    def __post_init__(self):
        if len(self.translation) != 3 or not all(math.isfinite(v) for v in (*self.translation, self.yaw)):
            raise ValueError("Finite XYZ translation and yaw in radians required")

    @property
    def quaternion(self):
        return (math.cos(self.yaw/2), 0., 0., math.sin(self.yaw/2))

    def vectors_to_world(self, vectors):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return torch.stack((c*vectors[..., 0]-s*vectors[..., 1],
                            s*vectors[..., 0]+c*vectors[..., 1], vectors[..., 2]), -1)

    def to_world(self, points):
        return self.vectors_to_world(points)+points.new_tensor(self.translation)

    def to_local(self, points):
        p = points-points.new_tensor(self.translation)
        return Placement(yaw=-self.yaw).vectors_to_world(p)


class CatFields:
    def __init__(self, directory, placement=Placement(), device="cpu"):
        self.directory = Path(directory)
        self.meta = verify_scene_files(self.directory)
        self.placement = placement
        m = self.meta
        if m["axis_order"] != "xyz" or m["units"] != "m" or m["resolution"] <= 0:
            raise ValueError("Unvalidated CAT coordinates")
        shape = tuple(m["shape"])
        if len(shape) != 3 or min(shape) < 2:
            raise ValueError("Invalid CAT grid shape")
        if not np.allclose(np.array(m["origin_corner"])+m["resolution"]/2, m["sample_origin"]):
            raise ValueError("Inconsistent voxel-centre origin")
        self.fields = {}
        for name, channels in (("sdf", 1), ("bf", 3), ("gf", 3)):
            array = np.load(self.directory/(name+".npy"), allow_pickle=False)
            expected = shape if channels == 1 else (*shape, channels)
            if array.shape != expected or not np.isfinite(array).all():
                raise ValueError(f"Invalid CAT array: {name}")
            self.fields[name] = torch.as_tensor(array.reshape(*shape, channels), device=device, dtype=torch.float32)

    def sample(self, points_world):
        local = self.placement.to_local(points_world)
        out = {}
        for name, array in self.fields.items():
            value, valid = sample_xyz_grid(array, local, self.meta["sample_origin"], self.meta["resolution"])
            out[name] = self.placement.vectors_to_world(value) if name in ("bf", "gf") else value[..., 0]
        out["valid"] = valid  # arrays have identical domains and are validated finite
        return out


def cat_mesh_arrays(directory):
    """Use the exact checksummed USD triangles. Import USD only at call time."""
    from pxr import Usd, UsdGeom, UsdPhysics
    directory = Path(directory)
    verify_scene_files(directory)
    stage = Usd.Stage.Open(str(directory/"scene.usda"))
    if UsdGeom.GetStageUpAxis(stage) != "Z" or UsdGeom.GetStageMetersPerUnit(stage) != 1.:
        raise ValueError("Unvalidated USD units or up axis")
    prim = stage.GetPrimAtPath("/CAT/Obstacles")
    mesh = UsdGeom.Mesh(prim)
    if (not mesh or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            or UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() != "none"
            or prim.HasAPI(UsdPhysics.RigidBodyAPI)):
        raise ValueError("Expected static non-convex physical CAT mesh")
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    if not np.allclose(np.array(matrix), np.eye(4)):
        raise ValueError("CAT source mesh must be untransformed; use an explicit Placement")
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    if not (counts == 3).all():
        raise ValueError("Only the exported triangle topology is supported")
    return (np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32),
            np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3))
