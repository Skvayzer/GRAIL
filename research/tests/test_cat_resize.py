from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cat_parallel_train import resume_phase_progress,interval_crossed


class ResizeTests(unittest.TestCase):
    def test_checkpoint_cadence_tracks_transitions_not_old_batch_count(self):
        self.assertFalse(interval_crossed(0,536576,1638400))
        self.assertTrue(interval_crossed(1597440,2134016,1638400))
        self.assertFalse(interval_crossed(2134016,2670592,1638400))
        self.assertTrue(interval_crossed(16000000,16536576,16384000))

    def config(self):
        return dict(args=dict(num_envs=2048),native_recipe=dict(policy_config=dict(unroll_length=32)),
                    phases=[("lateral","transfer",64),("lateral","ppo",256)])

    def test_legacy_stage_boundary_preserves_completed_transfer(self):
        c=dict(phase=0,next_update=64,total_steps=4194304)
        progress,budgets=resume_phase_progress(c,self.config(),[("lateral","transfer"),("lateral","ppo")])
        self.assertEqual(progress,[4194304,0])
        self.assertEqual(budgets,[4194304,16777216])

    def test_legacy_mid_stage_is_in_transitions_not_old_update_index(self):
        c=dict(phase=1,next_update=3,total_steps=4194304+3*65536)
        progress,_=resume_phase_progress(c,self.config(),[("lateral","transfer"),("lateral","ppo")])
        self.assertEqual(progress,[4194304,196608])

    def test_resized_checkpoint_keeps_partial_batch_progress(self):
        c=dict(phase_transitions=[4194304,196608+524288],
               phase_budget_transitions=[4194304,16777216],total_steps=4915200)
        progress,_=resume_phase_progress(c,self.config(),[("lateral","transfer"),("lateral","ppo")])
        self.assertEqual(progress,c["phase_transitions"])
        c["total_steps"]+=1
        with self.assertRaisesRegex(ValueError,"accounting"):
            resume_phase_progress(c,self.config(),[("lateral","transfer"),("lateral","ppo")])

    def test_different_stage_order_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"sequence"):
            resume_phase_progress({},self.config(),[("lateral","ppo"),("lateral","transfer")])
