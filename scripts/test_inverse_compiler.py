from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from common import save_json
from compiler.compile import compile_world
from state.adapters import inverse_scene

parser = argparse.ArgumentParser()
parser.add_argument("--scene", required=True)
parser.add_argument("--source", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
objects, camera = inverse_scene(args.scene, 1.0)
camera["whitebox_source_root"] = str(Path(args.source).resolve())
result = compile_world(objects, camera, args.output, "Maintain the inferred geometry.")
image = np.asarray(Image.open(result["first_frame"]))
assert image.std() > 1 and image.shape == (480, 832, 3)
assert all(o.provenance == "INFERRED" for o in objects)
save_json(Path(args.output) / "world_snapshot.json",
          {"objects": [o.to_dict() for o in objects], "camera": camera})
print({"objects": len(objects), "nonblank_std": float(image.std()), "result": result["first_frame"]})
