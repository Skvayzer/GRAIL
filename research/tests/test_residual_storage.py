"""Synthetic tensor storage tests: no optimizer steps or robot/simulator API."""
from pathlib import Path
import copy
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

    def populated_fixture(self):
        model, optimizer, gen = self.make()
        # Construct NONEMPTY moment buffers directly. This checks serialization
        # and validation, not a claim of a completed optimizer/training update.
        for parameter in model.parameters():
            optimizer.state[parameter] = dict(step=torch.tensor(2.),
                exp_avg=torch.full_like(parameter, .125), exp_avg_sq=torch.full_like(parameter, .03125))
        return model, optimizer, gen

    def test_nonempty_moments_roundtrip_without_an_optimizer_step(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"synthetic_moments.pt"
            original = self.populated_fixture()
            save_checkpoint(path, *original, self.contract, updates=2)
            restored = self.make(seed=80)
            self.assertEqual(load_checkpoint(path, *restored, self.contract), 2)
            a, b = original[1].state_dict(), restored[1].state_dict()
            self.assertEqual(a["param_groups"], b["param_groups"])
            for index, moment in a["state"].items():
                for key, value in moment.items():
                    torch.testing.assert_close(value, b["state"][index][key], atol=0, rtol=0)

    def test_invalid_moments_fail_before_any_live_state_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"moments.pt"
            save_checkpoint(path, *self.populated_fixture(), self.contract, updates=2)
            original = torch.load(path, weights_only=True)
            def bad_shape(payload):
                payload["optimizer"]["state"][0]["exp_avg"] = torch.zeros(1)
            def bad_counter(payload):
                payload["optimizer"]["state"][0]["step"] = torch.tensor(-1.)
            def bad_variance(payload):
                payload["optimizer"]["state"][0]["exp_avg_sq"].fill_(-.1)
            def bad_hyperparameter(payload):
                payload["optimizer"]["param_groups"][0]["lr"] *= 100
            def bad_name_order(payload):
                payload["optimizer_parameter_names"][0].reverse()
            def missing_moment(payload):
                del payload["optimizer"]["state"][0]
            def wrong_update_count(payload):
                payload["updates"] = 3
            for change in (bad_shape, bad_counter, bad_variance, bad_hyperparameter, bad_name_order, missing_moment, wrong_update_count):
                payload = copy.deepcopy(original)
                change(payload)
                bad = Path(directory)/(change.__name__+".pt")
                torch.save(payload, bad)
                model, optimizer, gen = self.make(seed=90)
                before, rng = copy.deepcopy(model.state_dict()), gen.get_state().clone()
                with self.subTest(change=change.__name__), self.assertRaises(ValueError):
                    load_checkpoint(bad, model, optimizer, gen, self.contract)
                self.assertEqual(optimizer.state_dict()["state"], {})
                self.assertTrue(torch.equal(gen.get_state(), rng))
                for key, value in before.items():
                    torch.testing.assert_close(value, model.state_dict()[key], atol=0, rtol=0)

    def test_different_optimizer_type_and_duplicate_parameter_ownership_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"moments.pt"
            save_checkpoint(path, *self.populated_fixture(), self.contract, updates=2)
            model, optimizer, gen = self.make()
            with self.assertRaisesRegex(ValueError, "contract"):
                load_checkpoint(path, model, torch.optim.AdamW(model.parameters()), gen, self.contract)
            optimizer.param_groups[0]["params"].append(optimizer.param_groups[0]["params"][0])
            with self.assertRaisesRegex(ValueError, "exactly"):
                save_checkpoint(Path(directory)/"duplicate.pt", model, optimizer, gen, self.contract, updates=0)
