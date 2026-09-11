"""Synthetic on-policy collection/gradients ONLY; no real optimizer steps."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gear_sonic.research.residual_engine import UpdateConfig, ResidualCollector, PPOUpdater, bootstrap_value
from gear_sonic.research.residual_learning import ResidualActorCritic
from test_residual_learning import packet


def setup():
    model = ResidualActorCritic(2, 7)
    config = UpdateConfig(horizon=4, epochs=2, minibatch_size=4)
    generator = torch.Generator().manual_seed(456)
    return model, generator, config


def collect(model, generator, config):
    collector = ResidualCollector(model, generator, 2, config)
    state, p = torch.zeros(2, 7), packet(2)
    for step in range(config.horizon):
        residual = collector.sample(p, state)
        final = state+.05
        terminal = torch.tensor([step == 1, False])
        timeout = torch.tensor([False, step == 2])
        collector.outcome(residual, torch.tensor([.1, .3]), terminal, timeout, p, final)
        state = final.clone()
        state[terminal | timeout] = 0
    return collector.finish()


class CollectionTests(unittest.TestCase):
    def test_stochastic_sample_likelihood_and_per_environment_resets(self):
        model, generator, config = setup()
        data = collect(model, generator, config)
        logp, _, _ = model.evaluate_action({k: data["packet/"+k] for k in packet()},
                                          data["state"], data["pre_tanh"])
        torch.testing.assert_close(logp, data["old_log_prob"])
        self.assertEqual(data["terminated"].tolist(), [False, False, True, False, False, False, False, False])
        self.assertEqual(data["truncated"].tolist(), [False, False, False, False, False, True, False, False])
        self.assertEqual(float(data["next_value"][2]), 0.)
        self.assertFalse(any(value.requires_grad for value in data.values()))

    def test_substituted_action_fails_entire_horizon(self):
        model, generator, config = setup()
        c = ResidualCollector(model, generator, 2, config)
        residual = c.sample(packet(2), torch.zeros(2, 7))
        with self.assertRaisesRegex(ValueError, "Executed latent"):
            c.outcome(torch.zeros_like(residual), torch.zeros(2), torch.zeros(2, dtype=bool),
                      torch.zeros(2, dtype=bool), packet(2), torch.zeros(2, 7))
        with self.assertRaisesRegex(ValueError, "Failed collection"):
            c.finish()

    def test_terminal_invalid_geometry_is_not_evaluated_but_timeout_is_required(self):
        model, _, _ = setup()
        p, state = packet(2), torch.zeros(2, 7)
        p["valid"][0] = False
        p["volume"][0] = float("nan")
        state[0] = float("nan")
        terminal = torch.tensor([True, False])
        result = bootstrap_value(model, p, state, terminal)
        self.assertEqual(float(result[0]), 0.)
        self.assertTrue(torch.isfinite(result).all())
        # With no true termination, the very same state is an invalid timeout
        # or ongoing final state, not permission to invent a bootstrap.
        with self.assertRaisesRegex(ValueError, "Invalid nonterminal/timeout"):
            bootstrap_value(model, p, state, torch.zeros(2, dtype=bool))

    def test_invalid_current_observation_is_not_skipped_or_reused(self):
        model, generator, config = setup()
        c = ResidualCollector(model, generator, 2, config)
        p = packet(2)
        p["valid"][1] = False
        with self.assertRaises(ValueError):
            c.sample(p, torch.zeros(2, 7))
        with self.assertRaisesRegex(ValueError, "Failed collection"):
            c.sample(packet(2), torch.zeros(2, 7))

    def test_mid_rollout_parameter_change_is_rejected(self):
        model, generator, config = setup()
        c = ResidualCollector(model, generator, 2, config)
        with torch.no_grad():
            model.obstacle.head.bias.add_(.1)  # Fixture mutation, not optimization.
        with self.assertRaisesRegex(ValueError, "changed during"):
            c.sample(packet(2), torch.zeros(2, 7))


class UpdateTests(unittest.TestCase):
    def test_full_gradient_loop_default_never_steps_or_changes_weights_rng(self):
        model, generator, config = setup()
        data = collect(model, generator, config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        updater = PPOUpdater(model, optimizer, generator, config)
        weights = copy.deepcopy(model.state_dict())
        rng, global_rng = generator.get_state().clone(), torch.get_rng_state().clone()
        with patch.object(optimizer, "step", side_effect=AssertionError("No optimizer steps authorized")):
            report = updater.run(data)
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertEqual(len(report["batches"]), 4)
        self.assertTrue(all(row["gradient_norm"] > 0 for row in report["batches"]))
        self.assertEqual(optimizer.state_dict()["state"], {})
        self.assertTrue(torch.equal(rng, generator.get_state()))
        self.assertTrue(torch.equal(global_rng, torch.get_rng_state()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        for key, value in weights.items():
            torch.testing.assert_close(value, model.state_dict()[key], atol=0, rtol=0)

    def test_stale_behavior_rejected_before_gradients_or_step(self):
        model, generator, config = setup()
        data = collect(model, generator, config)
        data["old_log_prob"][0] += 1
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        with patch.object(optimizer, "step", side_effect=AssertionError("No step")):
            with self.assertRaisesRegex(ValueError, "Stale or inconsistent"):
                PPOUpdater(model, optimizer, generator, config).run(data)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_optimize_dispatch_count_uses_mock_never_real_parameter_update(self):
        model, generator, config = setup()
        data = collect(model, generator, config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        updater = PPOUpdater(model, optimizer, generator, config)
        # Exercise the explicit branch with a NO-OP spy, never Adam.step.
        with patch.object(optimizer, "step", return_value=None) as spy:
            report = updater.run(data, optimize=True)
        self.assertEqual(spy.call_count, 4)
        self.assertEqual(report["optimizer_steps"], spy.call_count)
        self.assertEqual(optimizer.state_dict()["state"], {})

    def test_kl_stops_before_gradient_and_cleanup_preserves_mode(self):
        model, generator, config = setup()
        data = collect(model, generator, config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        updater = PPOUpdater(model, optimizer, generator, config)
        evaluate = model.evaluate_action
        calls = 0
        def over_kl(*args):
            nonlocal calls
            calls += 1
            logp, entropy, value = evaluate(*args)
            # First two calls validate original behavior. Third is update batch.
            return logp+(1. if calls > 2 else 0.), entropy, value
        model.eval()
        with patch.object(model, "evaluate_action", side_effect=over_kl), \
                patch.object(optimizer, "step", side_effect=AssertionError("No step")):
            report = updater.run(data)
        self.assertTrue(report["kl_early_stop"])
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertEqual(len(report["batches"]), 1)
        self.assertFalse(model.training)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_nonfinite_gradients_abort_and_are_cleared(self):
        model, generator, config = setup()
        data = collect(model, generator, config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        hook = model.value_head.weight.register_hook(lambda g: g*float("nan"))
        try:
            with patch.object(optimizer, "step", side_effect=AssertionError("No step")):
                with self.assertRaisesRegex(ValueError, "Nonfinite"):
                    PPOUpdater(model, optimizer, generator, config).run(data)
        finally:
            hook.remove()
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_update_bounds(self):
        for kw in (dict(horizon=True), dict(horizon=10000), dict(epochs=0), dict(minibatch_size=1),
                   dict(learning_rate=.1), dict(target_kl=float("nan")), dict(max_gradient_norm=0)):
            with self.assertRaises(ValueError):
                UpdateConfig(**kw)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_full_teacher_observation_dimensions_gradient_only(self):
        device = torch.device("cuda:0")
        model = ResidualActorCritic(104, 2221).to(device)
        config = UpdateConfig(horizon=4, epochs=1, minibatch_size=8)
        generator = torch.Generator(device=device).manual_seed(321)
        c = ResidualCollector(model, generator, 4, config)
        p = dict(volume=torch.zeros(4, 4, 13, 13, 11, device=device),
                 probes=torch.zeros(4, 104, 8, device=device), guidance=torch.zeros(4, 9, device=device),
                 valid=torch.ones(4, dtype=bool, device=device))
        p["volume"][:, 2:] = 1
        p["probes"][..., 6:] = 1
        p["guidance"][:, 7] = 1
        state = torch.zeros(4, 2221, device=device)
        for _ in range(4):
            latent = c.sample(p, state)
            c.outcome(latent, torch.arange(4., device=device), torch.zeros(4, dtype=bool, device=device),
                      torch.zeros(4, dtype=bool, device=device), p, state)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        with patch.object(optimizer, "step", side_effect=AssertionError("No real optimizer step")):
            result = PPOUpdater(model, optimizer, generator, config).run(c.finish())
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertEqual(result["samples"], 16)
        self.assertEqual(len(result["batches"]), 2)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
