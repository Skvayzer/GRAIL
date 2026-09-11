"""Conservative sphere covers of the loaded G1 URDF COLLISION primitives.

Not a certified cover of visual meshes or a physical G1. Missing/new geometry
fails loudly. Multiple spheres cover each finite cylinder including flat caps.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import torch

URDF = Path(__file__).resolve().parents[1] / "data/assets/robot_description/urdf/g1/main_nodex.urdf"


@dataclass(frozen=True)
class Probe:
    link: str
    offset: tuple[float, float, float]
    radius: float
    collision_index: int


def collision_probes(path=URDF, spacing=0.06):
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("Positive sphere spacing required")
    root = ET.parse(path).getroot()
    probes = []
    for link in root.findall("link"):
        for i, collision in enumerate(link.findall("collision")):
            origin = collision.find("origin")
            attrs = origin.attrib if origin is not None else {}
            xyz = tuple(map(float, attrs.get("xyz", "0 0 0").split()))
            roll, pitch, yaw = map(float, attrs.get("rpy", "0 0 0").split())
            # Local cylinder axis = Rz(yaw) Ry(pitch) Rx(roll) [0,0,1].
            axis = (math.cos(yaw)*math.sin(pitch)*math.cos(roll)+math.sin(yaw)*math.sin(roll),
                    math.sin(yaw)*math.sin(pitch)*math.cos(roll)-math.cos(yaw)*math.sin(roll),
                    math.cos(pitch)*math.cos(roll))
            shape = collision.find("geometry")[0]
            if shape.tag == "sphere":
                probes.append(Probe(link.attrib["name"], xyz, float(shape.attrib["radius"]), i))
            elif shape.tag == "cylinder":
                length, radius = float(shape.attrib["length"]), float(shape.attrib["radius"])
                count = max(1, math.ceil(length/spacing))
                # Each sphere covers a closed axial slice, including its cap rim.
                cover_radius = math.hypot(radius, length/(2*count))
                for j in range(count):
                    along = -length/2+(j+.5)*length/count
                    offset = tuple(x+a*along for x, a in zip(xyz, axis))
                    probes.append(Probe(link.attrib["name"], offset, cover_radius, i))
            else:
                raise ValueError(f"Unvalidated collision geometry: {link.attrib['name']} {shape.tag}")
    if not probes:
        raise ValueError("No collision primitives found")
    return probes


def world_probe_centers(probes, body_names, body_pos, body_quat):
    """Resolve by exact names, never assume URDF/PhysX array ordering. WXYZ."""
    missing = set(p.link for p in probes)-set(body_names)
    if missing:
        raise ValueError(f"Collision links absent from simulator: {sorted(missing)}")
    indices = [body_names.index(p.link) for p in probes]
    offsets = body_pos.new_tensor([p.offset for p in probes])
    q = body_quat[..., indices, :]
    if not torch.isfinite(q).all() or not torch.allclose(torch.linalg.vector_norm(q, dim=-1),
                                                       torch.ones_like(q[..., 0]), atol=1e-3):
        raise ValueError("Expected finite normalized WXYZ body rotations")
    v = offsets.expand_as(q[..., 1:])
    uv = torch.linalg.cross(q[..., 1:], v)
    rotated = v+2*(q[..., :1]*uv+torch.linalg.cross(q[..., 1:], uv))
    return body_pos[..., indices, :]+rotated, body_pos.new_tensor([p.radius for p in probes])


def self_pair_mask(probes, path=URDF):
    """Diagnostic only: exclude same-link/direct-joint pairs, retain cross-body.

    Covers can overlap even when collision meshes do not. Negative values are
    broad-phase warnings, not proof of actual self contact or a reward term.
    """
    root = ET.parse(path).getroot()
    adjacent = {frozenset((j.find("parent").attrib["link"], j.find("child").attrib["link"]))
                for j in root.findall("joint")}
    return torch.tensor([[i < j and a.link != b.link and frozenset((a.link, b.link)) not in adjacent
                          for j, b in enumerate(probes)] for i, a in enumerate(probes)])
