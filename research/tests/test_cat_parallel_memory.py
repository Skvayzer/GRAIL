from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cat_parallel_train import memory_fraction


class MemoryBudgetTests(unittest.TestCase):
    def test_torch_cap_leaves_room_for_non_torch_allocators(self):
        self.assertEqual(memory_fraction(14.,32*2**30),14/32)

    def test_invalid_limits_fail_closed(self):
        for value in (0.,-1.,32.,33.,float("inf"),float("nan")):
            with self.subTest(limit=value), self.assertRaises(ValueError):
                memory_fraction(value,32*2**30)
