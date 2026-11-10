"""Export portable coarse assets from the dataset's frozen renderer source."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def export(source: Path, output: Path):
    sys.path.insert(0, str(source))
    from wm_engine.proxy.geometry import FACES
    from wm_engine.proxy.humanoid import PARTS, local_part
    from wm_engine.proxy.vehicle import vehicle_vertices

    def convert(points, floor):
        points = np.asarray(points)
        return np.column_stack((-points[:, 0], points[:, 2] - floor, points[:, 1])).tolist()

    human = [{"name": name, "vertices": convert(local_part(name, lo, hi), -.95)}
             for name, lo, hi in PARTS]
    vehicle = vehicle_vertices(np.zeros(3), [-1, -2, 0], [1, 2, 1.5])
    catalog = {
        "source": str(source),
        "coordinates": "Y up, +Z forward; foot/root at Y=0; units metres",
        "height_factors": {"standing": 1.0, "crouched": .65, "low_guard": .4},
        "faces": [list(face) for face in FACES],
        "assets": {
            "gta_human": {"size": [.65, 1.8, .4], "parts": human},
            "gta_vehicle": {"size": [2, 1.5, 4], "parts": [
                {"name": "body", "vertices": convert(vehicle, 0)}]},
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(human)} rigid human parts and vehicle body to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("assets/gta/catalog.json"))
    args = parser.parse_args()
    export(args.source, args.output)
