import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from residual_results import audit_runtime
from gear_sonic.research.learner_state import PhysicalHistory
from gear_sonic.research.residual_learning import generalized_advantage


def fixture():
    history = PhysicalHistory(torch.tensor([[1., 2.], [3., 4.]]), 2)
    state = history.data.flatten(1).clone()
    rows = []
    for i in range(4):
        done = torch.tensor([i == 1, i == 3])
        physical = history.data[:, -1]+1
        final = history.candidate(physical).flatten(1)
        physical[done] = 0.
        history.commit(physical, done)
        post = history.data.flatten(1)
        rows.append(dict(state=state, next_state=final, post_step_state=post,
            value=state.sum(1)*.1, next_value=final.sum(1)*.1, post_step_value=post.sum(1)*.1,
            reward=torch.ones(2), terminated=torch.zeros(2, dtype=torch.bool), truncated=done, dones=done,
            reference_step=torch.full((2,), i)))
        state = post.clone()
    arrays = {k: torch.stack([r[k] for r in rows]) for k in rows[0]}
    adv, returns = generalized_advantage(arrays["reward"], arrays["value"], arrays["next_value"],
                                        arrays["terminated"], arrays["truncated"])
    arrays.update(advantages=adv, returns=returns)
    return {k: v.numpy() for k, v in arrays.items()}


def write_fixture(directory, arrays):
    run = Path(directory)
    path = run/"residual_preflight.npz"
    np.savez_compressed(path, **arrays)
    report = dict(schema="grail-cat-residual-runtime-preflight-v1", complete=True,
        data_file=path.name, data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        base_backbone_unchanged=True, learner_unchanged=True, optimizer_steps=0, exploration_enabled=False,
        training_data=False, actor_action_parity_exact=True, steps=4, num_envs=2, capture_calls=4,
        state=dict(state_dim=4, history_length=2, physical_dim=2))
    (run/"residual_preflight.json").write_text(json.dumps(report))


class ResultsTests(unittest.TestCase):
    def test_asynchronous_timeout_history_and_gae(self):
        with tempfile.TemporaryDirectory() as directory:
            write_fixture(directory, fixture())
            result = audit_runtime(directory)
            self.assertTrue(result["passed"])
            self.assertEqual(result["timeouts"], 2)

    def test_wrong_reset_bootstrap_not_hidden_by_valid_checksum(self):
        arrays = fixture()
        arrays["next_value"][arrays["dones"]] = arrays["post_step_value"][arrays["dones"]]
        with tempfile.TemporaryDirectory() as directory:
            write_fixture(directory, arrays)
            with self.assertRaisesRegex(ValueError, "GAE"):
                audit_runtime(directory)

    def test_nonreset_final_state_mismatch_detected(self):
        arrays = fixture()
        arrays["next_state"][0, 0, 0] += 1
        with tempfile.TemporaryDirectory() as directory:
            write_fixture(directory, arrays)
            with self.assertRaisesRegex(ValueError, "parity"):
                audit_runtime(directory)

    def test_missing_field_and_hash_changes_rejected(self):
        arrays = fixture()
        del arrays["next_state"]
        with tempfile.TemporaryDirectory() as directory:
            write_fixture(directory, arrays)
            with self.assertRaisesRegex(ValueError, "fields"):
                audit_runtime(directory)
            write_fixture(directory, fixture())
            (Path(directory)/"residual_preflight.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "mutated"):
                audit_runtime(directory)


if __name__ == "__main__":
    unittest.main()
