"""Reuse original classified collision surfaces; never turn building bounds solid."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--position", type=float, nargs=3, required=True)
    parser.add_argument("--radius", type=float, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    from wm_engine.world.static_store import StaticWorldStore
    from wm_engine.world.collision import CollisionStore
    from wm_engine.proxy.scene import scene_geometry
    from wm_engine.proxy.shading import surface_colors

    vertices, colors, details = scene_geometry(
        StaticWorldStore(args.world), CollisionStore(args.world / "collision"),
        args.position, args.radius)
    colors = surface_colors(vertices, colors, "relief")
    local = vertices - np.asarray(args.position)
    local = local[..., [0, 2, 1]]
    local[..., 2] *= -1
    packed = np.concatenate((local, np.repeat(colors[:, None, :] / 255., 3, axis=1)), axis=2)
    args.output.mkdir(parents=True, exist_ok=True)
    packed.astype("<f4").tofile(args.output / "surfaces.bin")
    physical, mesh_ids = CollisionStore(args.world / "collision").region(args.position,args.radius,include_procedural=False)
    collision = physical['vertices'] - np.asarray(args.position)
    collision = collision[..., [0,2,1]]
    collision[...,2] *= -1
    collision.astype('<f4').tofile(args.output / 'collision.bin')
    metadata = {"origin_gta": args.position, "radius": args.radius,
                "triangles": len(vertices), "format": "little-endian float32, XYZ RGB per vertex",
                "world_to_local": "[x-origin_x, z-origin_z, -(y-origin_y)]",
                "source_world": str(args.world), "details": details,
                "collision_triangles": len(collision), "collision_mesh_ids": mesh_ids,
                "physics": "original non-procedural collision triangles; XYZ float32, separate from derived visual vegetation and water"}
    (args.output / "region.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"triangles": len(vertices), "output": str(args.output)}))


if __name__ == "__main__":
    main()
