from pathlib import Path

import av
import numpy as np


def sample_frames(video_path, output_dir, interval=5.0, extra_times=()):
    if interval <= 0:
        raise ValueError("frame_interval_sec must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 24)
        decoded = [(i, float(frame.time) if frame.time is not None else i/fps,
                    frame.to_image()) for i, frame in enumerate(container.decode(stream))]
    if not decoded:
        raise ValueError("Video has no decodable frames")
    times = np.asarray([x[1] for x in decoded])
    times -= times[0]
    duration = times[-1] + 1/fps
    targets = [0, times[-1]/2, times[-1]] if duration <= interval + 1/fps else [
        *np.arange(0, duration, interval), times[-1]]
    indices = sorted({int(np.argmin(abs(times-t))) for t in [*targets,*extra_times]})
    frames = []
    for index in indices:
        frame_id, _, image = decoded[index]
        path = output_dir / f"{frame_id:06d}.png"
        image.save(path)
        frames.append({"frame_id": frame_id, "timestamp": float(times[index]),
                       "path": str(path), "width": image.width, "height": image.height})
    return frames
