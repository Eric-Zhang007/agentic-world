"""Prepare source-geometry review scenes without editing an accepted world."""
import json
from pathlib import Path


def main():
    out = Path("outputs/gta_asset_alignment")
    out.mkdir(parents=True, exist_ok=True)
    base = {"schema_version": 1, "terrain": {"seed": 0, "size": [30, 30],
            "resolution": 16, "height_amplitude": 0, "flatten_radius": 5},
            "camera": {"position": [7, 5, 10], "target": [0, 1, 0], "fov": 50},
            "entities": []}
    for name, kind, pos, size, color in [
        ("gta_human", "actor", [-2, 0, 0], [.65, 1.8, .4], [0, 0, 1]),
        ("gta_vehicle", "box", [2, .75, 0], [2, 1.5, 4], [1, .75, .8])]:
        base["entities"].append({"id": name, "asset_id": name, "kind": kind,
                                 "position": pos, "size": size, "color": color})
    (out / "catalog_scene.json").write_text(json.dumps(base, indent=2), encoding="utf-8")
    base["entities"] = []
    base["static_geometry"] = {"path": "urban_region/surfaces.bin"}
    base["camera"] = {"position": [0, 0, 0], "target": [.60229676, -.07397245, -.79483752], "fov": 60}
    base["review_note"] = "Recorded camera position/forward; provisional FOV, not a pixel alignment claim. Static geometry review only."
    (out / "region_scene.json").write_text(json.dumps(base, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
