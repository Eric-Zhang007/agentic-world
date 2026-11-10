import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_config
from compiler.compile import compile_world
from renderer.minimax_h3 import H3Renderer
from state.models import WorldObject


def scene():
    objects = [
        WorldObject("road_01", "road", [0, 2, 16], bbox_3d=[18, 0.1, 25], provenance="AGENT_CREATED",
                    state={"matchable": False}),
        WorldObject("car_01", "car", [-2.5, 0.7, 14], bbox_3d=[4, 1.6, 1.8], provenance="AGENT_CREATED"),
        WorldObject("person_01", "person", [1, 0.5, 12], bbox_3d=[0.6, 1.8, 0.5], provenance="AGENT_CREATED"),
        WorldObject("tree_01", "tree", [4, -0.5, 17], bbox_3d=[2.5, 5, 2.5], provenance="AGENT_CREATED"),
    ]
    camera = {"intrinsics": [[600, 0, 416], [0, 600, 240], [0, 0, 1]],
              "world_to_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
              "width": 832, "height": 480}
    return objects, camera


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.steps:
        config["renderer"]["steps"] = args.steps
    objects, camera = scene()
    compiled = compile_world(objects, camera, Path(args.output) / "proxy", "Keep the scene stable.")
    print(H3Renderer(config["renderer"]).render(
        compiled["prompt"], compiled["first_frame"], output_dir=args.output), flush=True)
