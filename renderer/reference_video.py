"""Timestamp-based 24fps reference sampling; short or truncated inputs are rejected."""
from pathlib import Path

import av
import numpy as np


def read_reference(path, frames=72, fps=24, width=832, height=480):
    if frames <= 0 or fps != 24:
        raise ValueError("Installed H3 video reference requires positive frame count and 24 fps")
    images, timestamps = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        native_fps = float(stream.average_rate or 0)
        if native_fps <= 0:
            raise ValueError("Reference FPS unavailable")
        for index, frame in enumerate(container.decode(stream)):
            timestamp = float(frame.time) if frame.time is not None else index/native_fps
            timestamps.append(timestamp)
            images.append(frame.to_image().resize((width, height)))
            if timestamp-timestamps[0] >= frames/fps:
                break
    if not images:
        raise ValueError("Reference has no decodable video")
    times = np.asarray(timestamps)-timestamps[0]
    if np.any(np.diff(times) <= 0):
        raise ValueError("Reference timestamps are non-monotonic")
    available = float(times[-1]+1/native_fps)
    if available+1e-6 < frames/fps:
        raise ValueError(f"Reference too short: {available:.3f}s < {frames/fps:.3f}s; no padding allowed")
    targets = np.arange(frames)/fps
    indices = np.searchsorted(times, targets, side="right")-1
    selected = [images[int(index)] for index in indices]
    return selected, {"path": str(Path(path).resolve()), "native_fps": native_fps,
                      "target_fps": fps, "target_frames": frames,
                      "selected_source_indices": indices.tolist(),
                      "selected_source_times": times[indices].tolist(),
                      "resampling": "timestamp previous-frame hold, no loop or end padding",
                      "audio_used": False}
