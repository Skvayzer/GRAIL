"""Small CPU tests for real decoder distillation plumbing; no Isaac starts."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import tempfile

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cat_distill_model import DistillStudent, JOINTS, ProprioHistory, objective, flat_field_admission
from cat_distill_model import sha
from cat_distill_aggregate import aggregate


def fixture():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    names = list(JOINTS)+[f"upper_{i}" for i in range(17)]
    contract = dict(joints=names, scale=[.5]*29, offset=[0.]*29, clip=20)
    actor = SimpleNamespace(actor_module=SimpleNamespace(decoders={"g1_dyn": nn.Linear(1093, 29)}))
    expert = SimpleNamespace(layers=nn.ModuleList([nn.Linear(162, 64), nn.Linear(64, 24)]))
    model = DistillStudent(actor, contract, expert)
    batch = dict(obs=torch.randn(8, 1029)*.05, cat=torch.randn(8, 162)*.05,
                 base=torch.zeros(8, 64), mode=torch.tensor([1.]*4+[0.]*4))
    with torch.no_grad():
        target = model(**batch)
    batch["target"] = target.clone()
    batch["target"][:4, :12] += .1
    return model, batch, contract


class DistillationTests(unittest.TestCase):
    def test_dagger_aggregation_preserves_heldout_and_rejects_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); parent = root/"parent"; source = root/"source"
            parent.mkdir(); source.mkdir()
            arrays = dict(obs=np.zeros((2, 1029), np.float32), cat=np.zeros((2, 162), np.float32),
                base=np.zeros((2, 64), np.float32), mode=np.ones(2, np.float32),
                target=np.zeros((2, 29), np.float32), episode=np.array([0, 6]), validation=np.array([False, True]))
            np.savez_compressed(parent/"dataset.npz", **arrays)
            digest = sha(parent/"dataset.npz")
            (parent/"dataset.json").write_text(json.dumps(dict(dataset_sha256=digest)))
            rows = {k: v[:1].copy() for k, v in arrays.items()}
            np.savez_compressed(source/"dagger.npz", **rows)
            meta = dict(sha256=sha(source/"dagger.npz"), source_dataset_sha256=digest,
                        training_seeds=[0], robot_actuation=False)
            (source/"dagger.json").write_text(json.dumps(meta))
            aggregate(parent, [source], root/"good")
            with np.load(root/"good/dataset.npz") as a:
                self.assertEqual(a["validation"].sum(), 1)
                np.testing.assert_array_equal(a["obs"][a["validation"]], arrays["obs"][1:])
            meta["training_seeds"] = [6]
            (source/"dagger.json").write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                aggregate(parent, [source], root/"bad")

    @unittest.skipUnless((Path(__file__).resolve().parents[1]/"runs/20260912T095135_246506Z_stair_p1_cat_audit/cat_teacher_shadow.npz").exists(),
                         "Optional pinned live GRAIL packet not installed")
    def test_history_layout_against_actual_live_grail_packet(self):
        from scipy.spatial.transform import Rotation
        root = Path(__file__).resolve().parents[1]/"runs/20260912T095135_246506Z_stair_p1_cat_audit"
        meta = json.loads((root/"cat_teacher_shadow.json").read_text())
        self.assertEqual(sha(root/meta["packet_file"]), meta["packet_sha256"])
        with np.load(root/meta["packet_file"], allow_pickle=False) as a:
            ids = [meta["joint_names"].index(n) for n in meta["action_joint_names"]]
            obs = a["grail_obs__actor_obs"][:, 0, 0]
            gravity = -Rotation.from_quat(a["body_quat"][:, 0, meta["body_names"].index("pelvis")], scalar_first=True).as_matrix()[:, 2]
            for start, width, raw in ((30, 29, a["joint_pos"][:, 0, ids]-a["action_offset"][:, 0]),
                                      (320, 29, a["joint_vel"][:, 0, ids]), (900, 3, gravity)):
                expected = np.stack([raw[t-9:t+1].ravel() for t in range(9, len(raw))])
                np.testing.assert_allclose(obs[9:, start:start+10*width], expected, atol=1e-6, rtol=0)
            np.testing.assert_array_equal(obs[1:, 610+9*29:900], a["grail_actions"][:-1, 0])

    def test_support_boundary_only_for_contacting_feet_at_known_flat_floor(self):
        p = np.tile([.5, .5, .5], (11, 1)); p[3, 2] = -.01
        self.assertEqual(flat_field_admission(p, [0, 0, 0], (50, 50, 50), .04, [True, False]), (True, 1))
        self.assertFalse(flat_field_admission(p, [0, 0, 0], (50, 50, 50), .04, [False, False])[0])
        p[0, 2] = -.01
        self.assertFalse(flat_field_admission(p, [0, 0, 0], (50, 50, 50), .04, [True, True])[0])
        p[0, 2] = .5; p[3, 0] = -.01
        self.assertFalse(flat_field_admission(p, [0, 0, 0], (50, 50, 50), .04, [True, True])[0])
        p[3, 0] = .5; p[3, 2] = -.021
        self.assertFalse(flat_field_admission(p, [0, 0, 0], (50, 50, 50), .04, [True, True])[0])

    def test_zero_adapter_exactly_preserves_grail_decoder(self):
        model, batch, _ = fixture()
        predicted = model(**{k: batch[k] for k in ("obs", "cat", "base", "mode")})
        expected = model.decoder(torch.cat((batch["base"], batch["obs"]), -1))*.5
        torch.testing.assert_close(predicted, expected, atol=0, rtol=0)

    def test_actual_updates_reduce_leg_loss_without_updating_decoder_or_teacher(self):
        model, batch, _ = fixture()
        frozen = model.frozen_hash()
        before = objective(model, batch)[1]["leg_mse"]
        opt = torch.optim.Adam(model.trainable(), lr=.001)
        for _ in range(15):
            opt.zero_grad(); loss, _ = objective(model, batch); loss.backward(); opt.step()
        self.assertLess(objective(model, batch)[1]["leg_mse"], before)
        self.assertEqual(frozen, model.frozen_hash())
        self.assertTrue(all(p.grad is None for p in model.decoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.features.parameters()))

    def test_retention_does_not_read_cat_leg_targets(self):
        model, batch, _ = fixture()
        batch["mode"][:] = 0
        _, result = objective(model, batch)
        self.assertEqual(result["leg_mse"], 0)
        self.assertEqual(result["posture_mse"], 0)
        self.assertGreater(result["retention_mse"], 0)

    def test_optimizer_resume_exact(self):
        model, batch, _ = fixture()
        opt = torch.optim.Adam(model.trainable())
        for _ in range(2):
            opt.zero_grad(); objective(model, batch)[0].backward(); opt.step()
        saved_model, saved_opt = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())
        results = []
        for _ in range(2):
            opt.zero_grad(); objective(model, batch)[0].backward(); opt.step()
            results.append(torch.cat([p.detach().flatten().clone() for p in model.trainable()]))
            model.load_state_dict(saved_model); opt.load_state_dict(copy.deepcopy(saved_opt))
        torch.testing.assert_close(*results, atol=0, rtol=0)

    def test_history_joint_order_reset_and_no_fake_zero_rotation(self):
        _, _, contract = fixture()
        names = contract["joints"][::-1]
        history = ProprioHistory(contract, names)
        q = np.r_[0, 0, .8, 1, 0, 0, 0, np.arange(29)*.01]
        v = np.zeros(35); previous = np.arange(29)*.02
        obs = history.sample(q, v, previous)
        self.assertEqual(obs.shape, (1029,))
        np.testing.assert_allclose(obs[30:59], q[7:][::-1])
        np.testing.assert_allclose(obs[900:930], np.tile([0, 0, -1], 10))
        np.testing.assert_allclose(obs[960:966], [1, 0, 0, 1, 0, 0])
        q[7] += .2
        new = history.sample(q, v, previous)
        np.testing.assert_allclose(new[30:59], obs[30:59])
        self.assertAlmostEqual(new[30+9*29+28]-obs[30+9*29+28], .2, places=6)


if __name__ == "__main__":
    unittest.main()
