import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from state.models import WorldObject, ObservationObject
from perception.visualize import draw_frame

parser = argparse.ArgumentParser()
parser.add_argument("episode")
args = parser.parse_args()
root = Path(args.episode)
world = json.loads((root / "world_before.json").read_text())
objects = [WorldObject(**o) for o in world["objects"]]
for attempt in sorted(root.glob("attempt_*")):
    frames = json.loads((attempt / "observations.json").read_text())
    matches = json.loads((attempt / "matching.json").read_text())
    for entry, matching in zip(frames, matches):
        frame = entry["frame"]
        draw_frame(frame, [ObservationObject(**o) for o in entry["objects"]],
                   matching["matching"], objects, world["camera"],
                   attempt / "sampled_frames" / f"debug_{frame['frame_id']:06d}.png")
print("Redrew debug overlays without label collisions")
