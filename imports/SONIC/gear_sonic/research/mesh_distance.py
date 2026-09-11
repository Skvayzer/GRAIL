"""Warp distance to closed, outward-wound triangle clutter; not a physics solver."""
import math

import numpy as np
import torch
import trimesh
import warp as wp


@wp.kernel
def _distance(mesh: wp.uint64, points: wp.array(dtype=wp.vec3), max_distance: float,
              distances: wp.array(dtype=float), normals: wp.array(dtype=wp.vec3),
              valid: wp.array(dtype=wp.int32)):
    i = wp.tid()
    q = wp.mesh_query_point_sign_winding_number(mesh, points[i], max_distance)
    if q.result:
        closest = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        delta = points[i]-closest
        length = wp.length(delta)
        distances[i] = length*q.sign
        if length > 1.e-7:
            normals[i] = wp.normalize(delta)*q.sign
        else:
            a = wp.mesh_eval_position(mesh, q.face, 1., 0.)
            b = wp.mesh_eval_position(mesh, q.face, 0., 1.)
            c = wp.mesh_eval_position(mesh, q.face, 0., 0.)
            normals[i] = wp.normalize(wp.cross(b-a, c-a))
        valid[i] = 1


class ClosedMeshDistance:
    def __init__(self, vertices, faces, device="cpu"):
        vertices, faces = np.asarray(vertices), np.asarray(faces)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
                or faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer)
                or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("Invalid finite triangle mesh")
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        # Check each disconnected component; one inverted component must not be
        # hidden by another component's positive volume. Do not silently repair.
        if (not mesh.is_watertight or not mesh.is_winding_consistent
                or any(part.volume <= 0 for part in mesh.split(only_watertight=False))
                or (mesh.area_faces <= 1e-12).any()):
            raise ValueError("Signed queries require closed, nondegenerate, outward-wound components")
        wp.init()
        self.device = torch.device(device)
        self.vertices = torch.as_tensor(vertices.copy(), dtype=torch.float32, device=self.device).contiguous()
        self.faces = torch.as_tensor(faces.copy().ravel(), dtype=torch.int32, device=self.device).contiguous()
        self.mesh = wp.Mesh(points=wp.from_torch(self.vertices, dtype=wp.vec3),
                            indices=wp.from_torch(self.faces), support_winding_number=True)

    def query(self, points, max_distance=100.):
        if points.shape[-1] != 3 or not math.isfinite(max_distance) or max_distance <= 0:
            raise ValueError("Expected XYZ points and positive finite query radius")
        if points.device != self.device or points.dtype != torch.float32:
            raise ValueError("Query must share mesh device and float32 type")
        shape = points.shape[:-1]
        finite = torch.isfinite(points).all(-1)
        safe = torch.nan_to_num(points).reshape(-1, 3).contiguous()
        d = torch.full((len(safe),), float("nan"), device=self.device)
        n = torch.full_like(safe, float("nan"))
        valid = torch.zeros(len(safe), dtype=torch.int32, device=self.device)
        if len(safe):
            wp.launch(_distance, len(safe), inputs=[self.mesh.id, wp.from_torch(safe, dtype=wp.vec3), max_distance,
                      wp.from_torch(d), wp.from_torch(n, dtype=wp.vec3), wp.from_torch(valid)],
                      device=str(self.device), stream=wp.stream_from_torch() if self.device.type == "cuda" else None)
        valid = valid.reshape(shape).bool() & finite
        return (torch.where(valid, d.reshape(shape), float("nan")),
                torch.where(valid[..., None], n.reshape(*shape, 3), float("nan")), valid)


def reference_clearance_summary(clearance, links, margin=.03):
    """Conservative *sampled* reference gate; negative sphere gaps are warnings.

    Positive gaps clear the collider cover at the queried frames only. This is
    neither articulated reachability nor continuous-time collision certification.
    """
    if clearance.ndim != 2 or clearance.shape[1] != len(links) or not clearance.shape[0]:
        raise ValueError("Expected nonempty frame-by-probe clearance array")
    if not math.isfinite(margin) or margin < 0 or not torch.isfinite(clearance).all():
        raise ValueError("Invalid reference clearance/margin")
    minima = {link: float(clearance[:, [i for i, v in enumerate(links) if v == link]].amin()) for link in sorted(set(links))}
    return dict(sampled_frames=len(clearance), minimum_clearance_m=float(clearance.amin()),
                minimum_by_link_m=minima, required_margin_m=margin,
                flagged_frames=int((clearance < margin).any(-1).sum()),
                sampled_reference_clear=bool((clearance >= margin).all()),
                continuous_articulated_path_verified=False)
