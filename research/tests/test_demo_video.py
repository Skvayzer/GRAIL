from pathlib import Path
import sys
import tempfile
import unittest

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_video import inspect_video


class VideoTests(unittest.TestCase):
    def test_decodes_and_counts_moving_video(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"test.mp4"
            with imageio.get_writer(path, fps=25, codec="libx264", pixelformat="yuv420p") as writer:
                for i in range(30):
                    frame = np.zeros((64, 96, 3), np.uint8)
                    frame[:, :, 0] = i*8
                    writer.append_data(frame)
            report = inspect_video(path)
            self.assertEqual(report["frames"], 30)
            self.assertEqual((report["width"], report["height"]), (96, 64))
            self.assertEqual(report["duration_s"], 1.2)
            self.assertGreater(report["mean_frame_change"], 0.)

    def test_empty_stale_renderer_is_not_a_passed_video(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"blank.mp4"
            with imageio.get_writer(path, fps=25, codec="libx264", pixelformat="yuv420p") as writer:
                for _ in range(30):
                    writer.append_data(np.zeros((64, 96, 3), np.uint8))
            with self.assertRaisesRegex(ValueError, "static"):
                inspect_video(path)
