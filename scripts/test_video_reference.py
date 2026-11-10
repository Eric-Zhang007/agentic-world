import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_config
from renderer.minimax_h3 import H3Renderer
from renderer.reference_video import read_reference

parser = argparse.ArgumentParser()
parser.add_argument("video")
parser.add_argument("--output", required=True)
parser.add_argument("--steps", type=int, default=3)
parser.add_argument("--inspect-only", action="store_true")
args = parser.parse_args()
config = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")
config["renderer"]["steps"] = args.steps
if args.inspect_only:
    frames, metadata = read_reference(args.video)
    print(metadata)
else:
    print(H3Renderer(config["renderer"]).render(
        "Video 1 is the source reference. Keep its object identities, positions, camera motion "
        "and geometry. Produce a faithful continuous reconstruction of the reference scene. "
        "Do not add or remove objects.", reference_video=args.video, output_dir=args.output))
