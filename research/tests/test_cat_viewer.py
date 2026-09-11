import ast
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cat_viewer import gallery_layout


class CatViewerTests(unittest.TestCase):
    def test_gallery_only_translates_and_leaves_a_one_metre_gap(self):
        a = dict(effective_size=[3., 2., 1.52], origin_corner=[-.5, -1., 0.])
        self.assertEqual(gallery_layout([a]), [[0., 0., 0.]])
        self.assertEqual(gallery_layout([a, a]), [[0., -1.5, 0.], [0., 1.5, 0.]])
        with self.assertRaises(ValueError):
            gallery_layout([])
        with self.assertRaises(ValueError):
            gallery_layout([a]*5)

    def test_viewer_import_does_not_start_simulator(self):
        # The dry-run and supervisor must remain usable before any Kit import.
        path = Path(__file__).resolve().parents[1]/"cat_viewer.py"
        module = ast.parse(path.read_text())
        for node in module.body:
            if isinstance(node, ast.Import):
                self.assertFalse(any(n.name.startswith(("isaac", "omni", "pxr")) for n in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertFalse(node.module.startswith(("isaac", "omni", "pxr")))


if __name__ == "__main__":
    unittest.main()
