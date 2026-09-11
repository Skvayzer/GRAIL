"""Exercise the simulator bridge using an in-memory fake, never Isaac/actuation."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gear_sonic.research.learner_state import environment_selection
from gear_sonic.research.residual_engine import ResidualCollector, UpdateConfig
from gear_sonic.research.residual_learning import ResidualActorCritic
from gear_sonic.research.residual_simulator import SimulatorCollector
from test_residual_learning import packet


class FakePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_module = torch.nn.Linear(7, 29)
        self.eval()
        self.obs_dict_buffer = {}
    def init_rollout(self):
        self.obs_dict_buffer = {}
    def act_inference(self, inputs, *, latent_residual, latent_residual_mode):
        assert latent_residual_mode == "post_quantization"
        self.obs_dict_buffer = {key: value[:, None].clone() for key, value in inputs.items()}
        return self.actor_module(inputs["fixture"])+latent_residual[:, :29]


class FakeTask:
    def __init__(self):
        self.begins, self.ends = 0, 0
    def before_step(self):
        self.begins += 1
    def outcome(self, dones):
        self.ends += 1


class FakeWrapper:
    def __init__(self):
        class Rewards:
            def compute(self, dt):
                return torch.tensor([.1, .2])
        self.env = NS(num_envs=2, position=torch.zeros(2, 7), reward_manager=Rewards(),
                      reset_terminated=torch.zeros(2, dtype=bool), reset_time_outs=torch.zeros(2, dtype=bool),
                      research_avoidance=FakeTask())
        self.count, self.actions = 0, []
        self.wrong_wrapper_reward = False
    def obs(self):
        return {"fixture": self.env.position.clone()}
    def step(self, action):
        self.count += 1
        self.actions.append(action["actions"].clone())
        self.env.position += .1
        self.env.reset_terminated = torch.tensor([self.count == 2, False])
        self.env.reset_time_outs = torch.tensor([False, self.count == 3])
        self.env.position[self.env.reset_terminated] = float("nan")
        # The tap is invoked now, BEFORE these rows are reset.
        reward = self.env.reward_manager.compute(.02)
        done = self.env.reset_terminated | self.env.reset_time_outs
        self.env.position[done] = 0.
        if self.wrong_wrapper_reward:
            reward = reward+10
        return self.obs(), reward, done, {}


class FakeOracle:
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.selections = []
        self.bad_timeout = False
    def sample(self, env_mask=None):
        env = self.wrapper.env
        index = environment_selection(env_mask, 2)
        self.selections.append(None if env_mask is None else env_mask.clone())
        position = env.position[index]
        if not torch.isfinite(position).all():
            raise ValueError("Invalid physical row reached geometry")
        p = packet(len(position))
        if self.bad_timeout and self.wrapper.count == 3:
            p["valid"][-1] = False
        return p


class FakeState:
    def __init__(self, wrapper):
        self.wrapper, self.commits = wrapper, []
        self.history = NS(commit=self.commit)
    def physical(self, env_mask=None):
        return self.wrapper.env.position[environment_selection(env_mask, 2)].clone()
    def state(self, *, phase_offset=0, next_physical=None, env_mask=None):
        return self.physical(env_mask) if next_physical is None else next_physical.clone()
    def commit(self, current, reset):
        self.commits.append((current.clone(), reset.clone()))


def fixture():
    wrapper, actor = FakeWrapper(), FakePolicy()
    oracle, state = FakeOracle(wrapper), FakeState(wrapper)
    model = ResidualActorCritic(2, 7)
    collector = ResidualCollector(model, torch.Generator().manual_seed(52), 2,
                                  UpdateConfig(horizon=4, minibatch_size=4))
    return wrapper, actor, oracle, state, collector


class SimulatorBridgeTests(unittest.TestCase):
    def test_default_never_enables_exploration(self):
        w, actor, oracle, state, collector = fixture()
        with self.assertRaisesRegex(ValueError, "opt-in"):
            SimulatorCollector(w, actor, oracle, state, collector)
        self.assertEqual(w.count, 0)

    def test_pre_reset_bootstrap_and_asynchronous_failed_rows(self):
        w, actor, oracle, state, collector = fixture()
        original = {key: value.clone() for key, value in actor.state_dict().items()}
        with SimulatorCollector(w, actor, oracle, state, collector, allow_simulation_exploration=True) as runtime:
            obs = w.obs()
            for _ in range(4):
                obs, _, _, _ = runtime.step(obs)
        data = collector.finish()
        self.assertEqual(runtime.steps, 4)
        self.assertEqual((runtime.terminations, runtime.timeouts), (1, 1))
        self.assertEqual((w.env.research_avoidance.begins, w.env.research_avoidance.ends), (4, 4))
        self.assertEqual(w.count, 4)  # No hidden extra physics steps.
        self.assertEqual(float(data["next_value"][2]), 0.)
        expected = collector.model.deterministic(packet(1), torch.full((1, 7), .3))[1]
        torch.testing.assert_close(data["next_value"][5:6], expected)
        self.assertTrue(any(mask is not None and torch.equal(mask, torch.tensor([False, True])) for mask in oracle.selections))
        self.assertEqual(state.commits[1][1].tolist(), [True, False])
        self.assertEqual(state.commits[2][1].tolist(), [False, True])
        self.assertNotIn("compute", vars(w.env.reward_manager))
        self.assertEqual(actor.obs_dict_buffer, {})  # Original teacher remains untouched.
        for key, value in original.items():
            torch.testing.assert_close(actor.state_dict()[key], value, atol=0, rtol=0)
        self.assertTrue(all(p.grad is None for p in actor.parameters()))

    def test_wrapper_recovery_cannot_silently_change_training_data(self):
        w, actor, oracle, state, collector = fixture()
        w.wrong_wrapper_reward = True
        with self.assertRaisesRegex(ValueError, "Wrapper recovery"):
            with SimulatorCollector(w, actor, oracle, state, collector, allow_simulation_exploration=True) as runtime:
                runtime.step(w.obs())
        self.assertTrue(collector.failed)
        self.assertNotIn("compute", vars(w.env.reward_manager))
        with self.assertRaisesRegex(ValueError, "Failed collection"):
            collector.finish()

    def test_invalid_timeout_discards_rollout_and_restores_capture(self):
        w, actor, oracle, state, collector = fixture()
        oracle.bad_timeout = True
        with self.assertRaisesRegex(ValueError, "Invalid nonterminal/timeout"):
            with SimulatorCollector(w, actor, oracle, state, collector, allow_simulation_exploration=True) as runtime:
                obs = w.obs()
                for _ in range(4):
                    obs, _, _, _ = runtime.step(obs)
        self.assertEqual(w.count, 3)
        self.assertTrue(collector.failed)
        self.assertNotIn("compute", vars(w.env.reward_manager))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
