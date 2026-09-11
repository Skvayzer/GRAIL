"""Pure tensor checks. No task training, GRAIL checkpoint, Isaac or optimizer."""
import copy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gear_sonic.research.obstacle_adapter import ObstacleAdapter
from gear_sonic.research.residual_learning import (
    ResidualActorCritic, LossConfig, generalized_advantage, normalized_advantages, clipped_ppo_loss,
)


def packet(n=3):
    volume = torch.zeros((n, 4, 3, 3, 3))
    volume[:, 2:] = 1
    probes = torch.zeros((n, 2, 8))
    probes[..., 6:] = 1
    guidance = torch.zeros((n, 9))
    guidance[:, 7] = 1
    return dict(volume=volume, probes=probes, guidance=guidance, valid=torch.ones(n, dtype=bool))


class LatentPolicyTests(unittest.TestCase):
    def test_zero_mean_no_rng_change_and_nonzero_exploration(self):
        before = torch.get_rng_state().clone()
        net = ResidualActorCritic(2, 7)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        p, state = packet(), torch.ones(3, 7)
        residual, value = net.deterministic(p, state)
        self.assertEqual(int(torch.count_nonzero(residual)), 0)
        generator = torch.Generator().manual_seed(78)
        sample = net.sample(p, state, generator)
        self.assertGreater(float(sample["residual"].abs().sum()), 0.)
        self.assertLess(float(sample["residual"].abs().max()), net.bound)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(value.shape, (3,))

    def test_sample_likelihood_and_rng_state_roundtrip(self):
        net, p, state = ResidualActorCritic(2, 7), packet(), torch.ones(3, 7)
        gen = torch.Generator().manual_seed(18)
        rng = gen.get_state().clone()
        sample = net.sample(p, state, gen)
        likelihood, entropy, value = net.evaluate_action(p, state, sample["pre_tanh"])
        torch.testing.assert_close(likelihood, sample["log_prob"], atol=0, rtol=0)
        torch.testing.assert_close(value, sample["value"], atol=0, rtol=0)
        gen.set_state(rng)
        repeated = net.sample(p, state, gen)
        for key in sample:
            torch.testing.assert_close(sample[key], repeated[key], atol=0, rtol=0)
        self.assertTrue(torch.isfinite(entropy).all())
        wrong_likelihood, _, _ = net.evaluate_action(p, state, sample["residual"])
        self.assertFalse(torch.allclose(likelihood, wrong_likelihood))

    def test_invalid_packets_never_explore_or_enter_loss(self):
        net, p, state = ResidualActorCritic(2, 7), packet(), torch.ones(3, 7)
        p["valid"][1] = False
        with torch.no_grad():
            net.obstacle.head.bias.fill_(1)
        residual, _ = net.deterministic(p, state)
        self.assertEqual(int(torch.count_nonzero(residual[1])), 0)
        with self.assertRaisesRegex(ValueError, "valid observations"):
            net.sample(p, state, torch.Generator())
        with self.assertRaisesRegex(ValueError, "Invalid obstacle"):
            net.evaluate_action(p, state, torch.zeros(3, 64))
        with self.assertRaisesRegex(ValueError, "explicit RNG"):
            net.sample(p, state, None)

    def test_gradients_reach_actor_head_and_critic_without_parameter_update(self):
        net, p, state = ResidualActorCritic(2, 7), packet(), torch.arange(21.).reshape(3, 7)/21
        before = copy.deepcopy(net.state_dict())
        sample = net.sample(p, state, torch.Generator().manual_seed(3))
        log_prob, entropy, values = net.evaluate_action(p, state, sample["pre_tanh"])
        loss, metrics = clipped_ppo_loss(log_prob, sample["log_prob"], entropy, values,
            sample["value"], torch.ones(3), normalized_advantages(torch.tensor([-1., 0., 2.])))
        actor_grad, value_grad = torch.autograd.grad(loss, (net.obstacle.head.weight, net.value_head.weight))
        self.assertGreater(float(actor_grad.norm()), 0.)
        self.assertGreater(float(value_grad.norm()), 0.)
        self.assertEqual(metrics["approx_kl"], 0.)
        self.assertTrue(all(parameter.grad is None for parameter in net.parameters()))
        for key, value in before.items():
            torch.testing.assert_close(value, net.state_dict()[key], atol=0, rtol=0)

    def test_shared_encoder_refactor_preserves_old_shadow_result_and_keys(self):
        net, p = ObstacleAdapter(2, 64), packet()
        old = copy.deepcopy(net.state_dict())
        with torch.no_grad():
            # Exercise nonzero weights too: output expression is exactly the same.
            net.head.weight.fill_(.02)
            fused = net.fusion(torch.cat((net.volume_net(p["volume"]), net.probe_net(p["probes"]), p["guidance"]), -1))
            expected = net.bound*torch.tanh(net.head(fused))
            torch.testing.assert_close(net(p), expected, atol=0, rtol=0)
        self.assertEqual(set(old), set(net.state_dict()))

    def test_bad_state_and_contract_rejected(self):
        for kwargs in (dict(state_dim=0), dict(state_dim=20000), dict(state_dim=7, initial_std=0)):
            with self.assertRaises(ValueError):
                ResidualActorCritic(2, **kwargs)
        net = ResidualActorCritic(2, 7)
        for bad in (torch.zeros(3, 8), torch.full((3, 7), float("nan"))):
            with self.assertRaises(ValueError):
                net(packet(), bad)


class AdvantageTests(unittest.TestCase):
    def test_timeout_bootstraps_final_state_but_not_next_episode(self):
        reward, value = torch.tensor([[1.], [100.]]), torch.tensor([[2.], [20.]])
        next_value = torch.tensor([[5.], [7.]])
        term = torch.tensor([[False], [True]])
        timeout = torch.tensor([[True], [False]])
        adv, returns = generalized_advantage(reward, value, next_value, term, timeout, gamma=.9, lam=1)
        torch.testing.assert_close(adv, torch.tensor([[3.5], [80.]]))
        torch.testing.assert_close(returns, torch.tensor([[5.5], [100.]]))

    def test_true_terminal_wins_and_horizon_bootstraps(self):
        rewards = torch.ones(2, 2)
        values = torch.zeros(2, 2)
        next_values = torch.full((2, 2), 10.)
        terminated = torch.tensor([[True, False], [True, False]])
        truncated = torch.tensor([[True, False], [False, False]])
        adv, _ = generalized_advantage(rewards, values, next_values, terminated, truncated, gamma=.5, lam=.8)
        torch.testing.assert_close(adv, torch.tensor([[1., 8.4], [1., 6.]]))

    def test_invalid_flags_or_values_rejected(self):
        v = torch.zeros(3, 1)
        with self.assertRaises(ValueError):
            generalized_advantage(v, v, v, v, v)
        with self.assertRaises(ValueError):
            generalized_advantage(v, v, v+float("nan"), v.bool(), v.bool())
        with self.assertRaises(ValueError):
            normalized_advantages(torch.ones(1))
        torch.testing.assert_close(normalized_advantages(torch.ones(3)), torch.zeros(3))


class ObjectiveTests(unittest.TestCase):
    def test_policy_clipping_signs_and_value_clipping(self):
        ratio = torch.tensor([1.5, .5], requires_grad=True)
        values = torch.tensor([1., 0.], requires_grad=True)
        zero = torch.zeros(2)
        loss, metrics = clipped_ppo_loss(ratio.log(), zero, zero, values, zero,
            torch.tensor([1., 1.]), torch.tensor([1., -1.]), LossConfig(value_weight=1))
        # min(1.5,1.2)=1.2; min(-.5,-.8)=-.8 -> policy loss=-.2.
        # max(0,.8^2), max(1,1) -> .5 * mean(.64,1)=.41.
        self.assertAlmostEqual(float(loss.detach()), .21, places=6)
        self.assertAlmostEqual(metrics["clip_fraction"], 1.)
        self.assertGreater(metrics["approx_kl"], 0.)

    def test_no_gradient_through_saved_targets(self):
        zero = torch.zeros(3)
        with self.assertRaisesRegex(ValueError, "detached"):
            clipped_ppo_loss(zero, zero.requires_grad_(), zero, zero, zero, zero, zero)
        with self.assertRaises(ValueError):
            LossConfig(clip_ratio=2.)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
