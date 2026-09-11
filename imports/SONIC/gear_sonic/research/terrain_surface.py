"""Exact triangle-surface rays, including open terrain meshes. Not a signed SDF.

Terrain tops are *candidate* support; CAT surfaces never become support here.
All queries use metres/Z-up and retain normals/miss masks explicitly.
"""
import math

import numpy as np
import torch
import warp as wp


@wp.kernel
def _rays(mesh: wp.uint64, origins: wp.array(dtype=wp.vec3), directions: wp.array(dtype=wp.vec3),
          limit: float, distances: wp.array(dtype=float), normals: wp.array(dtype=wp.vec3),
          valid: wp.array(dtype=wp.int32)):
    i = wp.tid()
    hit = wp.mesh_query_ray(mesh, origins[i], directions[i], limit)
    if hit.result:
        distances[i] = hit.t
        # Preserve authored winding, not the ray-facing normal: upside-down
        # geometry must not silently turn into upward support.
        a = wp.mesh_eval_position(mesh, hit.face, 1., 0.)
        b = wp.mesh_eval_position(mesh, hit.face, 0., 1.)
        c = wp.mesh_eval_position(mesh, hit.face, 0., 0.)
        normals[i] = wp.normalize(wp.cross(b-a, c-a))
        valid[i] = 1


class TerrainSurface:
    def __init__(self, vertices, faces, *, ground_z=None, device="cpu"):
        vertices, faces = np.asarray(vertices), np.asarray(faces)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
                or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                or not np.issubdtype(faces.dtype, np.integer) or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("Expected finite terrain triangles")
        triangles = vertices[faces]
        if (np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1) <= 1e-12).any():
            raise ValueError("Degenerate terrain triangles")
        if ground_z is not None and not math.isfinite(ground_z):
            raise ValueError("Invalid explicit ground plane")
        wp.init()
        self.device, self.ground_z = torch.device(device), ground_z
        self.vertices = torch.as_tensor(vertices.copy(), dtype=torch.float32, device=self.device).contiguous()
        self.faces = torch.as_tensor(faces.copy().ravel(), dtype=torch.int32, device=self.device).contiguous()
        self.mesh = wp.Mesh(points=wp.from_torch(self.vertices, dtype=wp.vec3), indices=wp.from_torch(self.faces))

    def raycast(self, origins, directions, limit=100.):
        if (origins.shape != directions.shape or origins.shape[-1] != 3
                or origins.dtype != torch.float32 or directions.dtype != torch.float32
                or origins.device != self.device or directions.device != self.device
                or not math.isfinite(limit) or limit <= 0):
            raise ValueError("Rays need matching float32 XYZ arrays/device and positive limit")
        shape = origins.shape[:-1]
        finite = torch.isfinite(origins).all(-1) & torch.isfinite(directions).all(-1)
        finite &= (torch.linalg.vector_norm(directions, dim=-1)-1.).abs() < 1e-4
        ro = torch.nan_to_num(origins).reshape(-1, 3).contiguous()
        rd = torch.nan_to_num(directions).reshape(-1, 3).contiguous()
        distances = torch.full((len(ro),), float("nan"), device=self.device)
        normals = torch.full_like(ro, float("nan"))
        valid = torch.zeros(len(ro), dtype=torch.int32, device=self.device)
        if len(ro):
            wp.launch(_rays, len(ro), inputs=[self.mesh.id, wp.from_torch(ro, dtype=wp.vec3),
                wp.from_torch(rd, dtype=wp.vec3), limit, wp.from_torch(distances),
                wp.from_torch(normals, dtype=wp.vec3), wp.from_torch(valid)], device=str(self.device),
                stream=wp.stream_from_torch() if self.device.type == "cuda" else None)
        valid = valid.reshape(shape).bool() & finite
        return (torch.where(valid, distances.reshape(shape), float("nan")),
                torch.where(valid[..., None], normals.reshape(*shape, 3), float("nan")), valid)

    def support_below(self, points, max_drop=10.):
        direction = torch.zeros_like(points)
        direction[..., 2] = -1.
        d, normal, valid = self.raycast(points, direction, max_drop)
        height = points[..., 2]-d
        if self.ground_z is not None:
            ground_valid = torch.isfinite(points).all(-1) & (points[..., 2] >= self.ground_z)
            ground_valid &= points[..., 2]-self.ground_z <= max_drop
            use_ground = ground_valid & (~valid | (height < self.ground_z))
            height = torch.where(use_ground, self.ground_z, height)
            normal = torch.where(use_ground[..., None], normal.new_tensor([0., 0., 1.]), normal)
            valid |= ground_valid
        return height, normal, valid
