"""Synthetic tensor storage tests: no optimizer steps or robot/simulator API."""
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gear_sonic.research.residual_learning import ResidualActorCritic
from gear_sonic.research.residual_rollout import ResidualRollout, minibatches
from gear_sonic.research.learning_checkpoint import save_checkpoint, load_checkpoint
from test_residual_learning import packet


class StorageTests(unittest.TestCase):
    def test_collection_clones_sources_and_preserves_time_environment_order(self):
        model, p, state = ResidualActorCritic(2, 7), packet(2), torch.zeros(2, 7)
        gen = torch.Generator().manual_seed(41)
        storage = ResidualRollout(3, 2)
        for t in range(3):
            state[:, 0] = torch.tensor([t*2, t*2+1])
            action = model.sample(p, state, gen)
            storage.append(p, state, action, torch.ones(2), torch.zeros(2, dtype=bool),
                           torch.zeros(2, dtype=bool), torch.zeros(2))
        state.fill_(100)
        data = storage.finish()
        torch.testing.assert_close(data["state"][:, 0], torch.arange(6.).float())
        self.assertFalse(any(t.requires_grad for t in data.values()))
        batches = list(minibatches(data, 4, gen))
        self.assertEqual([len(b["returns"]) for b in batches], [4, 2])
        self.assertEqual(sorted(torch.cat([b["state"][:, 0] for b in batches]).tolist()), list(range(6)))
        with self.assertRaises(ValueError):
            storage.finish()

    def test_incomplete_or_invalid_rollout_is_rejected(self):
        storage = ResidualRollout(3, 2)
        with self.assertRaises(ValueError):
            storage.finish()
        with self.assertRaises(ValueError):
            ResidualRollout(1024, 16)
        p = packet(2)
        p["valid"][0] = False
        with self.assertRaises(ValueError):
            storage.append(p, torch.zeros(2, 7), {}, torch.zeros(2), torch.zeros(2, dtype=bool),
                           torch.zeros(2, dtype=bool), torch.zeros(2))


class CheckpointTests(unittest.TestCase):
    contract = dict(backbone_sha256="1"*64, observation={"state_order": ["fixture"]}, algorithm={"name": "fixture"})

    def make(self, seed=42, bound=.1):
        net = ResidualActorCritic(2, 7, seed=seed, bound=bound)
        opt = torch.optim.Adam(net.parameters(), lr=3e-4)
        gen = torch.Generator().manual_seed(seed)
        return net, opt, gen

    def test_roundtrip_is_learner_only_does_not_update_or_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"step000000.pt"
            original = self.make()
            digest = save_checkpoint(path, *original, self.contract, updates=0)
            restored = self.make(seed=100)
            updates = load_checkpoint(path, *restored, self.contract)
            self.assertEqual(updates, 0)
            self.assertEqual(len(digest), 64)
            for k, v in original[0].state_dict().items():
                torch.testing.assert_close(v, restored[0].state_dict()[k], atol=0, rtol=0)
            self.assertTrue(torch.equal(original[2].get_state(), restored[2].get_state()))
            self.assertEqual(original[1].state_dict()["state"], {})  # Never stepped.
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, *restored, self.contract, updates=1)
            self.assertEqual(before, path.read_bytes())
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_wrong_base_contract_and_non_tensor_hyperparameter_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"initial.pt"
            original = self.make()
            save_checkpoint(path, *original, self.contract, updates=0)
            for bad in ({**self.contract, "backbone_sha256": "2"*64},
                        {**self.contract, "observation": {"state_order": ["different"]}}):
                with self.assertRaisesRegex(ValueError, "contract"):
                    load_checkpoint(path, *self.make(), bad)
            with self.assertRaisesRegex(ValueError, "contract"):
                load_checkpoint(path, *self.make(bound=.05), self.contract)
            original[1].param_groups[0]["lr"] = float("nan")
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                save_checkpoint(Path(directory)/"bad.pt", *original, self.contract, updates=0)

    def test_optimizer_cannot_include_a_backbone_parameter(self):
        net, _, gen = self.make()
        extra = torch.nn.Parameter(torch.ones(1))
        wrong = torch.optim.Adam([*net.parameters(), extra])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "frozen backbone"):
                save_checkpoint(Path(directory)/"bad.pt", net, wrong, gen, self.contract, updates=0)
