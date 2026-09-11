"""Offline operations checks: no real process signals, networking or updates."""
import copy
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2_job import process_identity, live
from m2_overnight import stop_owned, supervision_reason
from m2_results import audit
from gear_sonic.research.learning_checkpoint import save_checkpoint
from gear_sonic.research.observation_shadow import state_hash
from gear_sonic.research.residual_learning import ResidualActorCritic
from gear_sonic.research.training_admission import digest


class JobTests(unittest.TestCase):
    def test_current_identity_and_pid_reuse_rejection(self):
        identity = process_identity(os.getpid())
        self.assertIsNotNone(identity)
        self.assertTrue(live({"identity": identity}))
        changed = dict(identity, start_ticks="different")
        self.assertFalse(live({"identity": changed}))
        self.assertFalse(live({"identity": None}))

    def test_stop_signals_only_owned_live_launcher(self):
        process = Mock()
        process.poll.return_value = None
        stop_owned(process)
        process.send_signal.assert_called_once_with(signal.SIGINT)
        process.wait.assert_called_once_with(timeout=45)
        dead = Mock()
        dead.poll.return_value = 0
        stop_owned(dead)
        dead.send_signal.assert_not_called()

    def test_supervisor_budgets_and_nonfinite_values(self):
        self.assertIsNone(supervision_reason(100, 30, 2000, 28800))
        self.assertIn("disk", supervision_reason(19, 30, 2000, 28800))
        self.assertIn("progress", supervision_reason(100, 601, 2000, 28800))
        self.assertIn("budget", supervision_reason(100, 30, 28801, 28800))
        self.assertIn("Nonfinite", supervision_reason(float("nan"), 30, 2000, 28800))


class ResultTests(unittest.TestCase):
    def fixture(self, root):
        root = Path(root)
        model = ResidualActorCritic(2, 7)
        optimizer = torch.optim.Adam(model.parameters())
        generator = torch.Generator().manual_seed(3)
        # Tuple metadata deliberately exercises torch-save versus JSON types.
        contract = dict(backbone_sha256="a"*64, observation={"shape": (3, 3, 3)}, algorithm={})
        checkpoint = root/"learner_final.pt"
        saved_hash = save_checkpoint(checkpoint, model, optimizer, generator, contract, 0)
        plan = dict(mode="collect", num_envs=1, iterations=2, update={"horizon": 4})
        (root/"training_plan.json").write_text(json.dumps(plan))
        rows = [dict(iteration=i, environment_steps=i*4, optimizer_steps=0,
                     minimum_cat_gap_m=.1, maximum_contact_n=0., maximum_foot_error_m=.01) for i in (1, 2)]
        (root/"metrics.jsonl").write_text("\n".join(json.dumps(x) for x in rows)+"\n")
        (root/"run.json").write_text(json.dumps(dict(exit_code=0, outputs_valid=True)))
        result = dict(plan=plan, plan_sha256=digest(root/"training_plan.json"), complete=True,
            metrics=rows, backbone_unchanged=True, checkpoint_roundtrip_verified=True,
            initial_optimizer_steps=0, optimizer_steps=0, total_optimizer_steps=0,
            initial_learner_sha256=state_hash(model), learner_sha256=state_hash(model),
            learner_changed=False, contract=contract, simulator_steps=8, failures=0, timeouts=0,
            checkpoint_sha256=saved_hash)
        (root/"training_result.json").write_text(json.dumps(result))
        return root, result

    def test_actual_untouched_checkpoint_audits_without_simulator(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.fixture(directory)
            result = audit(root)
            self.assertTrue(result["passed"])
            self.assertEqual(result["optimizer_steps"], 0)
            self.assertEqual(result["transitions"], 8)

    def test_changed_checkpoint_and_false_update_claim_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, result = self.fixture(directory)
            changed = copy.deepcopy(result)
            changed["total_optimizer_steps"] = 1
            (root/"training_result.json").write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                audit(root)
            (root/"training_result.json").write_text(json.dumps(result))
            with (root/"learner_final.pt").open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum"):
                audit(root)


if __name__ == "__main__":
    unittest.main()
