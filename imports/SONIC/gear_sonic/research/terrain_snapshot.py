"""Read actual USD collision triangles at the LIVE PhysX rigid-body pose.

Never use visual-only triangles or assume source USD's local axes equal world.
Import simulator modules only inside capture(); pure USD routines are testable.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics


def rotation_wxyz(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1., atol=1e-4):
        raise ValueError("Expected normalized WXYZ quaternion")
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def remap_rays(origins, directions, from_position, from_quaternion, to_position, to_quaternion):
    """Rigid change of query frame; ray distances are invariant (no rescaling)."""
    origins, directions = np.asarray(origins), np.asarray(directions)
    positions = np.asarray([from_position, to_position])
    if (origins.shape != directions.shape or origins.shape[-1] != 3 or positions.shape != (2, 3)
            or not np.isfinite(origins).all() or not np.isfinite(directions).all() or not np.isfinite(positions).all()
            or not np.allclose(np.linalg.norm(directions, axis=-1), 1., atol=1e-4)):
        raise ValueError("Invalid frame/rays")
    a, b = rotation_wxyz(from_quaternion), rotation_wxyz(to_quaternion)
    rotation = a @ b.T  # row-vector from-world -> body -> to-world
    return (origins-positions[0]) @ rotation + positions[1], directions @ rotation


def rigid_collision_mesh(stage, root_path, position, quaternion):
    """Live pose replaces authored translation/rotation exactly once, retains scale."""
    if UsdGeom.GetStageUpAxis(stage) != "Z" or not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.):
        raise ValueError("Expected metre/Z-up stage")
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid() or not root.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError("Missing physical terrain rigid body")
    position = np.asarray(position, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("Invalid live position")
    rotation = rotation_wxyz(quaternion)
    cache = UsdGeom.XformCache()
    root_matrix = np.asarray(cache.GetLocalToWorldTransform(root))
    # USD uses row-vector matrices. Strip the authored rigid pose only, retaining
    # positive uniform scale. Mirroring/shear/nonuniform root scale is rejected.
    linear = root_matrix[:3, :3]
    scale = np.linalg.norm(linear, axis=1)
    if not (scale > 0).all() or not np.allclose(scale, scale[0], atol=1e-5):
        raise ValueError("Unvalidated nonuniform terrain root scale")
    authored_rotation = linear / scale[:, None]
    if not np.allclose(authored_rotation @ authored_rotation.T, np.eye(3), atol=1e-5) or np.linalg.det(authored_rotation) < .999:
        raise ValueError("Mirrored/sheared terrain root")
    vertices, faces, inventory = [], [], []
    offset = 0
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        if not prim.IsA(UsdGeom.Mesh) or UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() != "none":
            raise ValueError(f"Unvalidated terrain collider/approximation: {prim.GetPath()}")
        mesh = UsdGeom.Mesh(prim)
        if not (np.asarray(mesh.GetFaceVertexCountsAttr().Get()) == 3).all():
            raise ValueError("Terrain collider is not triangular")
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
        if np.linalg.det(matrix[:3, :3]) <= 0:
            raise ValueError("Mirrored terrain collider")
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        authored_world = points @ matrix[:3, :3] + matrix[3, :3]
        local = (authored_world-root_matrix[3, :3]) @ authored_rotation.T
        world = local @ rotation.T + position
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
        if mesh.GetOrientationAttr().Get() == "leftHanded":
            indices = indices[:, ::-1].copy()
        faces.append(indices + offset)
        vertices.append(world)
        inventory.append(dict(prim=str(prim.GetPath()), triangles=len(indices), approximation="none"))
        offset += len(points)
    if not vertices:
        raise ValueError("No physical terrain meshes")
    return np.concatenate(vertices).astype(np.float32), np.concatenate(faces), inventory


def ground_plane(stage, root_path="/World/ground"):
    root = stage.GetPrimAtPath(root_path)
    planes = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        if prim.GetTypeName() != "Plane":
            raise ValueError("Ground is not the validated infinite plane")
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        axis = prim.GetAttribute("axis").Get()
        normal = matrix.TransformDir(Gf.Vec3d(*{"X": (1., 0., 0.), "Y": (0., 1., 0.), "Z": (0., 0., 1.)}[axis]))
        normal = np.asarray(normal) / normal.GetLength()
        if not np.allclose(normal, [0., 0., 1.], atol=1e-5):
            raise ValueError("Ground plane is not upward and horizontal")
        planes.append(dict(prim=str(prim.GetPath()), height=float(matrix.ExtractTranslation()[2])))
    if len(planes) != 1:
        raise ValueError("Expected exactly one physical ground plane")
    return planes[0]


def capture(wrapper, run, *, allow_replicated=False):
    from isaaclab.sim.utils.stage import get_current_stage
    import torch
    stage, env = get_current_stage(), wrapper.env
    if env.num_envs != 1 and not (allow_replicated and 1 <= env.num_envs <= 4):
        raise ValueError("Terrain layout audit currently requires one environment")
    obj = env.scene["object"]
    path = obj.root_physx_view.prim_paths[0]
    origin = env.scene.env_origins[0].cpu().numpy()
    position = obj.data.root_pos_w[0].cpu().numpy()
    quaternion = obj.data.root_quat_w[0].cpu().numpy()
    vertices, faces, inventory = rigid_collision_mesh(stage, path, position, quaternion)
    vertices -= origin
    replicas = []
    for i in range(1, env.num_envs):
        other_origin = env.scene.env_origins[i].cpu().numpy()
        other_position = obj.data.root_pos_w[i].cpu().numpy()
        other_quaternion = obj.data.root_quat_w[i].cpu().numpy()
        other_vertices, other_faces, _ = rigid_collision_mesh(stage, obj.root_physx_view.prim_paths[i],
                                                             other_position, other_quaternion)
        if (not np.allclose(other_position-other_origin, position-origin, atol=2e-5, rtol=0)
                or not np.isclose(abs(np.dot(other_quaternion, quaternion)), 1., atol=1e-5)
                or other_vertices.shape != vertices.shape or not np.array_equal(other_faces, faces)
                or not np.allclose(other_vertices-other_origin, vertices, atol=2e-5, rtol=0)
                or abs(other_origin[2]-origin[2]) > 1e-5):
            raise ValueError("Replicated environments do not share identical local terrain")
        replicas.append(i)
    plane = ground_plane(stage)
    plane["height"] -= float(origin[2])
    motion = wrapper.motion_command
    ids = motion.motion_ids.unique()
    if len(ids) != 1:
        raise ValueError("Shared terrain snapshot requires one identical paired motion")
    total = int(motion.motion_lib.get_time_step_total(ids)[0])
    steps = torch.arange(total, device=env.device)
    positions = motion.motion_lib.get_object_root_pos(ids.expand(total), steps)[:, 0]
    quaternions = motion.motion_lib.get_object_root_quat(ids.expand(total), steps)[:, 0]
    # Static terrain only; dynamic obstacles need time-indexed collision queries.
    if not torch.allclose(positions, positions[:1].expand_as(positions), atol=1e-5) or not torch.allclose(
            (quaternions*quaternions[:1]).sum(-1).abs(), torch.ones(total, device=env.device), atol=1e-5):
        raise ValueError("Moving terrain reference is unsupported by static layout audit")
    if not torch.isclose((obj.data.root_quat_w[0]*motion.object_root_quat[0, 0]).sum().abs(),
                         torch.tensor(1., device=env.device), atol=1e-4) or not torch.allclose(
            obj.data.root_pos_w[0], motion.object_root_pos[0, 0], atol=1e-4):
        raise ValueError("Live terrain does not match reference placement")
    run = Path(run)
    mesh_path = run / "terrain_snapshot.npz"
    np.savez_compressed(mesh_path, vertices=vertices, faces=faces)
    meta = dict(schema="grail-cat-terrain-snapshot-v1", units="m", up_axis="Z",
        rigid_body_prim=path,
        coordinate_frame="environment_zero_local", live_position=(position-origin).tolist(),
        live_quaternion_wxyz=quaternion.tolist(), ground=plane, colliders=inventory,
        source_file=mesh_path.name, sha256=hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        mesh_contract="exact collision triangles at live pose; open meshes allowed; NOT signed volume")
    if allow_replicated:
        meta["replicas_verified"] = replicas
    (run / "terrain_snapshot.json").write_text(json.dumps(meta, indent=2)+"\n")
    return vertices, faces, meta
