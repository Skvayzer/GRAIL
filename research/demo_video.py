"""Small MP4 writer/validator for labeled research demos, not benchmark evidence."""
from pathlib import Path
import hashlib

import imageio.v2 as imageio
import numpy as np


def inspect_video(path):
    """Decode every frame; reject empty/static/invalid captures, retain checksum."""
    path = Path(path)
    count, variation, previous, shape = 0, 0., None, None
    with imageio.get_reader(path, format="ffmpeg") as reader:
        meta = reader.get_meta_data()
        fps = float(meta["fps"])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid video framerate")
        for frame in reader:
            if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                raise ValueError("Expected decoded RGB video")
            if shape is not None and frame.shape != shape:
                raise ValueError("Video frame dimensions changed")
            shape = frame.shape
            reduced = frame[::16, ::16].astype(np.float32)
            if previous is not None:
                variation += float(np.abs(reduced-previous).mean())
            previous = reduced
            count += 1
    if count < 25 or variation < .1:
        raise ValueError("Video is too short or visually static; inspect renderer")
    return dict(file=str(path), frames=count, fps=fps, duration_s=count/fps,
                width=shape[1], height=shape[0], decoded=True,
                mean_frame_change=variation/max(1, count-1),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
