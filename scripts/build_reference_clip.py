"""Build a disclosed sample-and-hold clip from original sampled RGB, not inverse renders."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import av
import numpy as np
from PIL import Image
from common import save_json


def build(manifest_path, output, duration=3.0):
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = json.loads(manifest_path.read_text())
    entries = manifest["frames"]
    times = np.asarray([entry["time"] for entry in entries], float)
    if len(times) < 2 or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Need strictly ordered sampled frames beginning at t=0")
    if times[-1] < duration:
        raise ValueError("Require sample coverage through the requested end; no end padding")
    indices = np.searchsorted(times, np.arange(round(duration*24))/24, side="right")-1
    output.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(output), "w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = 832, 480, "yuv420p"
        for index in indices:
            image = Image.open(manifest_path.parent / entries[index]["file"]).convert("RGB").resize((832, 480))
            frame = av.VideoFrame.from_image(image)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    metadata = {"source_manifest": str(manifest_path), "source_video_as_recorded": manifest["source"],
                "source_kind": "original sampled RGB supplied with inverse reconstruction",
                "not_original_full_fps_video": True, "not_ground_truth_geometry": True,
                "source_sampling": manifest["sampling"], "resampling": "timestamp previous-frame hold",
                "output_frames": len(indices), "output_fps": 24,
                "source_indices": indices.tolist(), "source_times": times[indices].tolist()}
    save_json(output.with_suffix(".metadata.json"), metadata)
    print({"output": str(output), "frames": len(indices), "unique_source_frames": len(set(indices.tolist()))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("output")
    args = parser.parse_args()
    build(args.manifest, args.output)
