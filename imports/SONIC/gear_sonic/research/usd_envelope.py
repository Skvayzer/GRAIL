"""Cover actual imported collider primitives, including URDF cylinder->capsule conversion.

Import after Kit startup. Does not alter assets. Mesh coverage remains a future
gate; unknown geometry/scaling fails rather than dropping a robot body.
"""
import math

from pxr import Gf, UsdGeom, UsdPhysics

from .body_envelope import Probe


def imported_collision_probes(stage, root_path, body_names, spacing=0.06):
    if spacing <= 0 or not math.isfinite(spacing):
        raise ValueError("Invalid probe spacing")
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise ValueError(f"Missing robot prim {root_path}")
    from pxr import Usd
    cache = UsdGeom.XformCache()
    probes, inventory = [], []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        body = prim
        while body.IsValid() and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if not body.IsValid() or body.GetName() not in body_names:
            raise ValueError(f"Collider has no named rigid body: {prim.GetPath()}")
        transform, _ = cache.ComputeRelativeTransform(prim, body)
        basis = [transform.TransformDir(Gf.Vec3d(*(1. if j == i else 0. for j in range(3)))) for i in range(3)]
        if any(abs(v.GetLength()-1) > 1e-5 for v in basis) or any(
                abs(Gf.Dot(basis[i], basis[j])) > 1e-5 for i in range(3) for j in range(i)):
            raise ValueError(f"Unvalidated collider scaling/shear: {prim.GetPath()}")
        kind = prim.GetTypeName()
        if kind == "Sphere":
            radius = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
            local_centers, cover_radius = [(0., 0., 0.)], radius
        elif kind in ("Capsule", "Cylinder"):
            geom = UsdGeom.Capsule(prim) if kind == "Capsule" else UsdGeom.Cylinder(prim)
            height, radius = geom.GetHeightAttr().Get(), geom.GetRadiusAttr().Get()
            axis = {"X": 0, "Y": 1, "Z": 2}[geom.GetAxisAttr().Get()]
            count = max(1, math.ceil(height/spacing))
            if kind == "Capsule":
                along = [-height/2+j*height/count for j in range(count+1)]
            else:
                along = [-height/2+(j+.5)*height/count for j in range(count)]
            cover_radius = math.hypot(radius, height/(2*count))
            local_centers = [tuple(a if k == axis else 0. for k in range(3)) for a in along]
        else:
            raise ValueError(f"Unsupported actual collider {kind}: {prim.GetPath()}")
        if radius <= 0 or not math.isfinite(cover_radius):
            raise ValueError(f"Invalid collider dimensions: {prim.GetPath()}")
        index = len(inventory)
        for point in local_centers:
            offset = tuple(transform.Transform(Gf.Vec3d(*point)))
            probes.append(Probe(body.GetName(), offset, cover_radius, index))
        inventory.append({"prim": str(prim.GetPath()), "body": body.GetName(), "type": kind,
                          "radius": radius, "cover_radius": cover_radius, "probes": len(local_centers)})
    if not probes:
        raise ValueError("No imported collision geometry found")
    return probes, inventory
