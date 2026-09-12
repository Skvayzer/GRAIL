"""Batched observation/action mapping and imitation wiring; CPU, no robot."""
import math
from types import SimpleNamespace
import unittest

import torch

from gear_sonic.research.cat_bridge import (AppliedTargetHistory, CatObservationBridge,
    GROUP_SIZES, JOINTS, OBS_JOINTS, SITES, normalize_fields, teacher_leg_loss)

ALL_JOINTS = OBS_JOINTS + tuple(f"{side}_wrist_{axis}_joint" for side in ("left", "right")
                              for axis in ("roll", "pitch", "yaw"))


def contract():
    return dict(schema="cat-observation-action-contract-v1", action_joints=JOINTS,
        observation_joints=OBS_JOINTS, default_joint_positions={n: .01*i for i, n in enumerate(ALL_JOINTS)},
        action_scale=.5, soft_lower=[-1.]*12, soft_upper=[1.]*12,
        sites=[dict(site=s, body="pelvis", position=[1., 0., 0.]) for s in SITES])


def inputs(names=ALL_JOINTS, b=2):
    c = contract()
    return dict(joint_pos=torch.tensor([[c["default_joint_positions"][n] for n in names]]).repeat(b, 1),
        joint_vel=torch.zeros(b, len(names)), pelvis_rotation=torch.eye(3).repeat(b, 1, 1),
        gyro=torch.zeros(b, 3), gravity=torch.tensor([[0., 0., -1.]]).repeat(b, 1),
        last_action=torch.zeros(b, 12), previous_targets=torch.zeros(b, 12),
        command_world=torch.tensor([[1., .3, .2, 1.]]).repeat(b, 1), foot_height=torch.full((b, 1), .07),
        phase=torch.tensor([[0., math.pi]]).repeat(b, 1), gf=torch.arange(b*33.).reshape(b, 11, 3),
        bf=torch.ones(b, 11, 3), sdf=torch.linspace(-2, 1, 11).repeat(b, 1)[..., None])


class BridgeTests(unittest.TestCase):
    def test_packet_exact_group_layout_and_default_subtraction(self):
        x = inputs()
        packet = CatObservationBridge(contract(), ALL_JOINTS).pack(**x)
        self.assertEqual(packet.shape, (2, 162))
        torch.testing.assert_close(packet[:, 6:29], torch.zeros(2, 23))
        torch.testing.assert_close(packet[:, 76:80], torch.tensor([[1., .3, .2, 0.]]).repeat(2, 1))
        torch.testing.assert_close(packet[:, 81:85], torch.cat((x["phase"].cos(), x["phase"].sin()), -1))
        point, start = 0, 85
        for count in GROUP_SIZES:
            for field in (x["gf"], x["bf"]*(x["sdf"] < .5), x["sdf"].clamp(-1, .5)):
                expected = field[:, point:point+count].reshape(2, -1)
                torch.testing.assert_close(packet[:, start:start+expected.shape[1]], expected)
                start += expected.shape[1]
            point += count
        self.assertEqual(start, 162)

    def test_joint_order_is_named_and_supports_large_batches(self):
        names = tuple(reversed(ALL_JOINTS))
        bridge = CatObservationBridge(contract(), names)
        x = inputs(names, 512)
        x["joint_pos"][:, names.index(JOINTS[0])] += .13
        packet = bridge.pack(**x)
        torch.testing.assert_close(packet[:, 6], torch.full((512,), .13))
        self.assertEqual(packet.shape, (512, 162))
        with self.assertRaises(ValueError):
            CatObservationBridge(contract(), (ALL_JOINTS[0],)*29)

    def test_heading_rotation_is_not_full_pelvis_rotation(self):
        x = inputs(b=1)
        # Yaw +90deg: world +Y is heading +X. Roll does not rotate PF into floor.
        x["pelvis_rotation"] = torch.tensor([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]])
        p = CatObservationBridge(contract(), ALL_JOINTS).pack(**x)
        torch.testing.assert_close(p[:, 77:80], torch.tensor([[.2, -.3, 0.]]))
        torch.testing.assert_close(p[:, 85:88], torch.tensor([[1., 0., 2.]]))

    def test_incremental_targets_use_previous_not_default_and_clip(self):
        bridge = CatObservationBridge(contract(), ALL_JOINTS)
        previous = torch.full((2, 12), .7)
        first = bridge.leg_targets(torch.full_like(previous, .4), previous)
        second = bridge.leg_targets(torch.full_like(previous, .4), first)
        torch.testing.assert_close(first, torch.full_like(first, .9))
        torch.testing.assert_close(second, torch.ones_like(second))
        torch.testing.assert_close(previous, torch.full_like(previous, .7))

    def test_history_uses_executed_targets_and_resets_only_selected_rows(self):
        history = AppliedTargetHistory(torch.zeros(2, 12))
        teacher_counterfactual = torch.ones(2, 12)
        actual = torch.full((2, 12), .1)
        history.commit(actual)
        torch.testing.assert_close(history.last_action, torch.full((2, 12), .2))
        actual[0] = .3
        history.commit(actual, torch.tensor([True, False]))
        torch.testing.assert_close(history.last_action, torch.zeros(2, 12))
        self.assertFalse(torch.equal(history.targets, teacher_counterfactual))
        actual.zero_()
        self.assertEqual(float(history.targets[0, 0]), float(torch.tensor(.3)))

    def test_site_offsets_and_gyro_are_link_local_read_only(self):
        bridge = CatObservationBridge(contract(), ALL_JOINTS, ["pelvis"])
        q = torch.tensor([[[math.sqrt(.5), 0., 0., math.sqrt(.5)]]])
        data = SimpleNamespace(joint_pos=torch.zeros(1, 29), joint_vel=torch.zeros(1, 29),
            body_pos_w=torch.tensor([[[10., 20., 1.]]]), body_quat_w=q,
            body_ang_vel_w=torch.tensor([[[0., 1., 0.]]]))
        before = {k: v.clone() for k, v in vars(data).items()}
        values = bridge.read_articulation(data, torch.tensor([[10., 20., 0.]]))
        torch.testing.assert_close(values["sites"], torch.tensor([[[0., 1., 1.]]]).repeat(1, 11, 1))
        torch.testing.assert_close(values["gyro"], torch.tensor([[1., 0., 0.]]), atol=2e-7, rtol=1e-6)
        for key in before:
            torch.testing.assert_close(before[key], getattr(data, key))

    def test_unnormalized_and_invalid_inputs_do_not_hide_errors(self):
        x = inputs()
        x["gravity"][0, 1] = float("nan")
        with self.assertRaises(ValueError):
            CatObservationBridge(contract(), ALL_JOINTS).pack(**x)
        gf, bf = normalize_fields(torch.ones(2, 11, 3), torch.ones(2, 11, 3), torch.tensor([1., 0.]))
        self.assertEqual(int(gf[1].count_nonzero()), 0)
        torch.testing.assert_close(bf.norm(dim=-1), torch.ones(2, 11), atol=1e-6, rtol=1e-6)

    def test_imitation_updates_legs_not_fake_arm_labels(self):
        names = tuple(reversed(ALL_JOINTS))
        targets = torch.nn.Parameter(torch.zeros(3, 29))
        label = torch.ones(3, 12, requires_grad=True)
        opt = torch.optim.SGD([targets], lr=1.)
        before = teacher_leg_loss(targets, names, label).item()
        for _ in range(10):
            opt.zero_grad()
            teacher_leg_loss(targets, names, label).backward()
            opt.step()
        self.assertLess(teacher_leg_loss(targets, names, label).item(), before)
        self.assertIsNone(label.grad)
        self.assertEqual(int(targets[:, [names.index(n) for n in names if n not in JOINTS]].count_nonzero()), 0)


if __name__ == "__main__":
    unittest.main()
