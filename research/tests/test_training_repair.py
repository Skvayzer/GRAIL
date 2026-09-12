"""CPU-only regressions for the failed overnight pilot repair."""
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gear_sonic.research.residual_learning import ResidualActorCritic
from gear_sonic.research.residual_engine import PPOUpdater, UpdateConfig
from test_residual_engine import collect
from test_residual_learning import packet
from cat_curriculum import recipes, SPLITS, STAGES


class CriticIsolationTests(unittest.TestCase):
    def test_value_gradient_cannot_change_actor(self):
        model = ResidualActorCritic(2, 7, critic_mode="independent")
        _, _, value = model(packet(3), torch.randn(3, 7))
        value.square().mean().backward()
        groups = model.gradient_groups()
        self.assertTrue(all(p.grad is None for p in groups["actor"]))
        self.assertTrue(all(p.grad is not None for p in groups["critic"]))
        self.assertFalse({id(p) for p in groups["actor"]} & {id(p) for p in groups["critic"]})
        self.assertEqual(sum(map(len, groups.values())), len(list(model.parameters())))

    def test_full_objective_connects_all_parameters_and_preserves_zero_mean(self):
        model = ResidualActorCritic(2, 7, critic_mode="independent")
        residual, _ = model.deterministic(packet(3), torch.ones(3, 7))
        self.assertEqual(int(torch.count_nonzero(residual)), 0)
        cfg = UpdateConfig(horizon=4, epochs=2, minibatch_size=4)
        gen = torch.Generator().manual_seed(2)
        data = collect(model, gen, cfg)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        report = PPOUpdater(model, opt, gen, cfg).run(data, optimize=False)
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertTrue(all("actor_gradient_norm" in r and "critic_gradient_norm" in r for r in report["batches"]))
        self.assertEqual(model.manifest()["schema"], "grail-cat-residual-actor-critic-v2")

    def test_legacy_model_manifest_and_parameter_names_remain_compatible(self):
        model = ResidualActorCritic(2, 7)
        self.assertEqual(model.manifest()["schema"], "grail-cat-residual-actor-critic-v1")
        self.assertNotIn("critic_mode", model.manifest())
        self.assertEqual(set(model.gradient_groups()), {"joint"})


class CurriculumTests(unittest.TestCase):
    def test_seed_splits_are_disjoint_and_recipes_do_not_repeat(self):
        sets = [set(s) for s in SPLITS.values()]
        self.assertTrue(all(not a & b for i, a in enumerate(sets) for b in sets[i+1:]))
        for stage in STAGES:
            for split in SPLITS:
                rows = recipes(stage, split)
                keys = [tuple(sorted(r.items())) for r in rows]
                self.assertEqual(len(keys), len(set(keys)))
                self.assertEqual(rows, recipes(stage, split))

    def test_small_batch_has_seed_diversity_and_lateral_starts_without_high_low(self):
        rows = recipes("lateral", "development")[:4]
        self.assertEqual(len({r["seed"] for r in rows}), 4)
        self.assertTrue(all(r["n_floor"] == r["n_ceiling"] == 0 for r in rows))
        self.assertEqual({r["difficulty"] for r in rows}, {.2, .4})


if __name__ == "__main__":
    unittest.main()
