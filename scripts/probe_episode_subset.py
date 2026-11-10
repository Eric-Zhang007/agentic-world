"""Validate extracted Parquet payloads, the existing adapter, and video decoding."""
import argparse
import json
from pathlib import Path
import sys

import av
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from state.adapters import gta_world


def main(root, source_root):
    result = {"parquet": {}, "videos": {}}
    for path in sorted(root.rglob("*.parquet")):
        table = pq.read_table(path)
        result["parquet"][str(path.relative_to(root))] = {
            "rows": table.num_rows, "schema": str(table.schema),
            "first_row": table.slice(0, 1).to_pylist(),
        }
    world = pq.read_table(root / "server/world_state.parquet")
    first = world.slice(0, 1).to_pylist()[0]
    payload = json.loads(first["payload_json"])
    objects = gta_world(payload, source_root)
    result["adapter"] = [{"object_id": o.object_id, "category": o.category,
                          "position": o.position, "provenance": o.provenance}
                         for o in objects]
    for path in sorted(root.rglob("*.mkv")):
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            times = []
            shape = None
            for frame in container.decode(stream):
                if shape is None:
                    frame.to_image().save(path.with_suffix(".first.png"))
                    shape = [frame.width, frame.height]
                times.append(float(frame.time) if frame.time is not None else None)
            result["videos"][str(path.relative_to(root))] = {
                "decoded_frames": len(times), "dimensions": shape,
                "average_rate": str(stream.average_rate), "first_time": times[0],
                "last_time": times[-1],
                "monotonic_pts": all(a is not None and b is not None and b > a
                                     for a, b in zip(times, times[1:])),
            }
    (root / "probe.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    args = parser.parse_args()
    main(args.root, args.source_root)
