import unittest

from pxr import Gf, Usd, UsdGeom, UsdPhysics
import torch

from gear_sonic.research.usd_envelope import imported_collision_probes


class ImportedEnvelopeTests(unittest.TestCase):
    def test_actual_capsule_caps_are_covered(self):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/Robot")
        body = UsdGeom.Xform.Define(stage, "/Robot/forearm")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        capsule = UsdGeom.Capsule.Define(stage, "/Robot/forearm/collider")
        capsule.CreateRadiusAttr(.1)
        capsule.CreateHeightAttr(.2)
        capsule.CreateAxisAttr("X")
        capsule.AddTranslateOp().Set(Gf.Vec3d(.3, 0., 0.))
        UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
        probes, inventory = imported_collision_probes(stage, "/Robot", ["forearm"])
        self.assertEqual(inventory[0]["type"], "Capsule")
        centers = torch.tensor([p.offset for p in probes])
        radii = torch.tensor([p.radius for p in probes])
        # Axial tips + cylindrical surface, including translation and X axis.
        surface = torch.tensor([[.1, 0., 0.], [.5, 0., 0.], [.3, .1, 0.], [.4, .1, 0.]])
        self.assertLessEqual(float((torch.cdist(surface, centers)-radii).amin(-1).amax()), 1e-6)

    def test_unsupported_collision_is_not_silently_dropped(self):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/Robot")
        body = UsdGeom.Xform.Define(stage, "/Robot/hand")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        cube = UsdGeom.Cube.Define(stage, "/Robot/hand/collider")
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        with self.assertRaisesRegex(ValueError, "Unsupported actual collider"):
            imported_collision_probes(stage, "/Robot", ["hand"])


if __name__ == "__main__":
    unittest.main()
