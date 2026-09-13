import itertools
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cat_parallel_train import should_evaluate


class EvaluationGateTests(unittest.TestCase):
    def test_disabled_blocks_periodic_and_phase_end_even_together(self):
        for periodic,end in itertools.product((False,True),repeat=2):
            self.assertFalse(should_evaluate(enabled=False,smoke_updates=0,
                stop_requested=False,periodic_due=periodic,phase_complete=end))

    def test_enabled_keeps_existing_schedule(self):
        for periodic,end in itertools.product((False,True),repeat=2):
            self.assertEqual(should_evaluate(enabled=True,smoke_updates=0,
                stop_requested=False,periodic_due=periodic,phase_complete=end),periodic or end)

    def test_stop_and_smoke_still_skip_evaluation(self):
        for smoke,stop in ((1,False),(0,True),(1,True)):
            self.assertFalse(should_evaluate(enabled=True,smoke_updates=smoke,
                stop_requested=stop,periodic_due=True,phase_complete=True))
