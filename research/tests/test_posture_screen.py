from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
import xml.etree.ElementTree as ET

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from screen_posture_clutter import classify_reference
from posture_witness import smooth_blend, joint_dof_order, arm_pose
from gear_sonic.research.body_envelope import Probe
from gear_sonic.research.capsule_screen import segment_distance, nonlocal_arm_pairs, capsule_gaps


class ReferenceConflictTests(unittest.TestCase):
    def fixture(self):
        outer = torch.full((100, 3), .1)
        inner = torch.full((100, 3), .11)
        outer[60:72, 1] = -.03
        inner[60:72, 1] = -.02
        return outer, inner, ["pelvis", "left_elbow_link", "left_ankle_roll_link"]

    def test_actual_arm_intersection_with_clear_protected_body(self):
        result = classify_reference(*self.fixture())
        self.assertTrue(result["candidate"])
        self.assertEqual(result["arm_intersection_frames"], 12)
        self.assertEqual(result["first_intersection_frame"], 60)
        self.assertFalse(result["training_ready"])
        self.assertFalse(result["posture_feasibility_verified"])

    def test_outer_cover_warning_alone_is_not_real_intersection(self):
        outer, inner, links = self.fixture()
        inner[:] = .005
        self.assertFalse(classify_reference(outer, inner, links)["candidate"])

    def test_spawn_and_leg_conflicts_rejected(self):
        outer, inner, links = self.fixture()
        outer[75, 2] = .01
        self.assertEqual(classify_reference(outer, inner, links)["state"], "REJECTED_PROTECTED_BODY_CONFLICT")
        outer, inner, links = self.fixture()
        outer[0, 1] = -.01
        self.assertEqual(classify_reference(outer, inner, links)["state"], "REJECTED_SPAWN_CONFLICT")

    def test_bad_shapes_and_unknown_values_fail(self):
        outer, inner, links = self.fixture()
        inner[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            classify_reference(outer, inner, links)


class CapsuleTests(unittest.TestCase):
    def test_crossing_parallel_skew_and_degenerate_segments(self):
        a = torch.tensor([[0., 0., 0.]]).repeat(5, 1)
        b = torch.tensor([[1., 0., 0.]]).repeat(5, 1)
        c = torch.tensor([[.5, -1., 0.], [0., .2, 0.], [.5, -1., .3], [2., 0., 0.], [.2, .3, .4]])
        d = torch.tensor([[.5, 1., 0.], [1., .2, 0.], [.5, 1., .3], [3., 0., 0.], [.2, .3, .4]])
        b[4] = a[4]
        expected = torch.tensor([0., .2, .3, 1., (.2**2+.3**2+.4**2)**.5])
        torch.testing.assert_close(segment_distance(a, b, c, d), expected, atol=1e-6, rtol=0)
        torch.testing.assert_close(segment_distance(c, d, a, b), expected, atol=1e-6, rtol=0)

    def test_nearly_parallel_endpoint_minimum(self):
        a, b = torch.tensor([[0., 0., 0.]]), torch.tensor([[1., 0., 0.]])
        c, d = torch.tensor([[0., .1, 0.]]), torch.tensor([[1., .10000001, 0.]])
        torch.testing.assert_close(segment_distance(a, b, c, d), torch.tensor([.1]), atol=1e-6, rtol=0)

    def test_multiple_primitives_and_adjacent_colliders(self):
        model = NS(body_names=["torso", "shoulder", "empty", "elbow", "wrist"], _parents=[-1, 0, 1, 2, 3])
        names = ["torso", "shoulder", "shoulder", "elbow", "wrist"]
        inventory = [dict(body=name) for name in names]
        pairs = nonlocal_arm_pairs(model, inventory)
        self.assertNotIn((1, 2), pairs)  # same-body primitives
        self.assertNotIn((1, 3), pairs)  # adjacent through collider-free joint
        self.assertIn((1, 4), pairs)     # not the whole ipsilateral arm chain
        self.assertIn((0, 4), pairs)

    def test_sphere_capsule_radii_applied_once(self):
        centers = torch.tensor([[[0., 0., 0.], [0., 0., 1.], [1., 0., .5]]])
        probes = [Probe("arm", (0., 0., 0.), .11, 0), Probe("arm", (0., 0., 1.), .11, 0),
                  Probe("torso", (1., 0., .5), .2, 1)]
        inventory = [dict(type="Capsule", radius=.1), dict(type="Sphere", radius=.2)]
        torch.testing.assert_close(capsule_gaps(centers, probes, inventory, [(0, 1)]), torch.tensor([[.7]]), atol=1e-6, rtol=0)


class BlendTests(unittest.TestCase):
    def fixture(self):
        suffixes = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
        names = [f"{side}_{suffix}_joint" for side in ("left", "right") for suffix in suffixes]
        names += [f"protected_{i}_joint" for i in range(15)]
        world = ET.Element("worldbody")
        root = ET.SubElement(world, "body", name="root")
        ET.SubElement(root, "joint", name="free_root", type="free")
        for name in names:
            body = ET.SubElement(root, "body", name=name+"_body")
            ET.SubElement(body, "joint", name=name)
        xml = ET.Element("mujoco")
        xml.append(world)
        model = NS(tree=ET.ElementTree(xml), num_dof=29, joints_range=torch.tensor([[-3., 3.]]).repeat(29, 1),
                   dof_axis=torch.tensor([[1., 0., 0.]]).repeat(29, 1),
                   body_names=["root"]+[n+"_body" for n in names],
                   mjcf_data={"body_to_joint": {n+"_body": n for n in names}})
        model.fk_batch = lambda pose, trans, **kwargs: NS(pose=pose.clone(), trans=trans.clone())
        raw = dict(pose_aa=torch.full((5, 30, 3), .1), root_trans_offset=torch.zeros(5, 3), fps=25)
        return dict(model=model, raw=raw)

    def test_constant_posture_corrects_initial_arms_but_never_root_waist_or_legs(self):
        data = self.fixture()
        source = data["raw"]["pose_aa"].clone()
        pose, fk, targets = arm_pose(data, -.5, .15, .7, mode="constant")
        torch.testing.assert_close(data["raw"]["pose_aa"], source, atol=0, rtol=0)
        torch.testing.assert_close(pose[:, [0]+list(range(15, 30))], source[:, [0]+list(range(15, 30))], atol=0, rtol=0)
        self.assertFalse(torch.equal(pose[0, 1:15], source[0, 1:15]))
        self.assertEqual(len(targets), 14)
        for index, value in enumerate(targets.values(), start=1):
            torch.testing.assert_close(pose[:, index], torch.tensor([[value, 0., 0.]]).repeat(5, 1), atol=0, rtol=0)
        torch.testing.assert_close(fk.trans[0], data["raw"]["root_trans_offset"], atol=0, rtol=0)

    def test_blend_remains_default_and_bad_mode_or_joint_limit_rejects(self):
        data = self.fixture()
        pose, _, _ = arm_pose(data, -.5, .15, .7)
        torch.testing.assert_close(pose, data["raw"]["pose_aa"], atol=0, rtol=0)  # All fixture frames before blend starts.
        with self.assertRaises(ValueError):
            arm_pose(data, 0., .1, .7, mode="unknown")
        with self.assertRaises(ValueError):
            arm_pose(data, 10., .1, .7, mode="constant")

    def test_scalar_dof_order_excludes_free_root(self):
        tree = ET.ElementTree(ET.fromstring('<mujoco><worldbody><body><joint name="root" type="free"/>'
            '<body><joint name="left_elbow_joint"/><body><joint name="left_wrist_yaw_joint"/></body>'
            '</body></body></worldbody></mujoco>'))
        model = NS(tree=tree, num_dof=2, joints_range=torch.zeros(2, 2), dof_axis=torch.zeros(2, 3),
                   actuated_joints_idx=[0, 1, 2])
        self.assertEqual(joint_dof_order(model), ["left_elbow_joint", "left_wrist_yaw_joint"])
        model.dof_axis = torch.zeros(3, 3)
        with self.assertRaises(ValueError):
            joint_dof_order(model)

    def test_quintic_blend_preserves_spawn_and_is_bounded(self):
        t = torch.linspace(0, 5, 1001)
        b = smooth_blend(t)
        self.assertTrue((b[t <= 1] == 0).all())
        self.assertTrue((b[t >= 3] == 1).all())
        self.assertTrue((b[1:] >= b[:-1]-1e-6).all())
        self.assertTrue(((b >= 0) & (b <= 1)).all())


if __name__ == "__main__":
    unittest.main()
