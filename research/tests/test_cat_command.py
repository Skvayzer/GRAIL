"""Teacher command/gait/delay mechanics; no Isaac or robot."""
import math
import json
from pathlib import Path
import unittest

import torch

from gear_sonic.research.cat_command import delayed_sites, field_command, gait_step
from gear_sonic.research.cat_teacher_shadow import config_json


class CommandTests(unittest.TestCase):
    def test_resolved_policy_path_serialization(self):
        self.assertEqual(json.loads(config_json(dict(path=Path("/tmp/run")))), dict(path="/tmp/run"))
        with self.assertRaises(TypeError):
            config_json(dict(unexpected=object()))

    def test_empty_command_and_pelvis_forward_projection(self):
        gf = torch.zeros(3, 11, 3)
        bf = torch.zeros_like(gf)
        gf[1, 1, 0] = 1
        gf[2, 1, 1] = 1
        torch.testing.assert_close(field_command(gf, bf), torch.tensor([
            [0., 0., 0., 0.], [.75, .525, 0., 0.], [.75, 0., .525, 0.]]))

    def test_native_stop_countdown_and_phase(self):
        command = torch.tensor([[0., 0., 0., 0.], [.75, .5, 0., 0.], [.75, .5, 0., 0.]])
        last = torch.tensor([[1., .5, 0., 0.]]).repeat(3, 1)
        phase = torch.tensor([[0., math.pi]]).repeat(3, 1)
        out, new_phase, stop = gait_step(command, last, phase, torch.tensor([100, 20, 0]), .1)
        torch.testing.assert_close(stop, torch.tensor([50, 19, 0]))
        torch.testing.assert_close(out, torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., 0.]]))
        torch.testing.assert_close(new_phase[:2, 0], torch.tensor([.1, .1]))
        torch.testing.assert_close(new_phase[2], torch.zeros(2))
        torch.testing.assert_close(command[0], torch.zeros(4))

    def test_delayed_root_keeps_articulation_not_old_body_positions(self):
        sites = torch.tensor([[[2., 1., .7], [2., 2., .8]]])
        root = torch.tensor([[2., 0., .5]])
        identity = torch.eye(3)[None]
        oldroot = torch.tensor([[1., 0., .5]])
        yaw90 = torch.tensor([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]])
        result = delayed_sites(sites, root, identity, oldroot, yaw90)
        torch.testing.assert_close(result, torch.tensor([[[0., 0., .7], [-1., 0., .8]]]))


if __name__ == "__main__":
    unittest.main()
