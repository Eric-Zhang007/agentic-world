"""Import one video_to_whitebox scene time point into a JSON snapshot.

This adapter intentionally preserves uncertainty. Generic scene geometry is
kept as category ``unknown`` and no confidence/depth/embedding is invented.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_world.state.models import WorldObject


@dataclass
class Camera:
    projection: str
    position: list[float]
    target: list[float]
    fov_degrees: float | None
    ortho_scale: float | None
    width: int | None
    height: int | None
    world_to_camera: list[list[float]] | None
    provenance: str = "INFERRED"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _lerp_value(left: Any, right: Any, alpha: float) -> Any:
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError("keyframe array sizes differ")
        return [_lerp_value(a, b, alpha) for a, b in zip(left, right)]
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left * (1.0 - alpha) + right * alpha
    return left if alpha < 1.0 else right


def interpolate_keyframes(keyframes: list[dict[str, Any]], timestamp: float) -> dict[str, Any]:
    if not keyframes:
        raise ValueError("keyframes must not be empty")
    ordered = sorted(keyframes, key=lambda item: float(item["time"]))
    times = [float(item["time"]) for item in ordered]
    if not math.isfinite(timestamp) or not all(math.isfinite(t) for t in times):
        raise ValueError("timestamps must be finite")
    if len(set(times)) != len(times):
        raise ValueError("duplicate keyframe timestamps")
    if len(ordered) == 1:
        return {key: value for key, value in ordered[0].items() if key != "time"}
    if timestamp < ordered[0]["time"] or timestamp > ordered[-1]["time"]:
        raise ValueError(
            f"timestamp {timestamp} is outside [{ordered[0]['time']}, "
            f"{ordered[-1]['time']}]"
        )
    if timestamp == ordered[-1]["time"]:
        return {key: value for key, value in ordered[-1].items() if key != "time"}
    for left, right in zip(ordered, ordered[1:]):
        if left["time"] <= timestamp <= right["time"]:
            span = float(right["time"]) - float(left["time"])
            alpha = 0.0 if span == 0 else (timestamp - left["time"]) / span
            return {
                key: (
                    (np.asarray(left[key]) + alpha *
                     ((np.asarray(right[key]) - np.asarray(left[key]) + 180) % 360 - 180)).tolist()
                    if key == "rotation" else _lerp_value(left[key], right[key], alpha)
                )
                for key in left
                if key != "time"
            }
    raise AssertionError("unreachable keyframe interval")


def _world_to_camera(position: np.ndarray, target: np.ndarray) -> list[list[float]]:
    forward = target - position
    norm = np.linalg.norm(forward)
    if norm == 0:
        raise ValueError("camera position and target must differ")
    forward = forward / norm
    up = np.array([0.0, 1.0, 0.0] if abs(forward[2]) > .99 else [0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm == 0:
        raise ValueError("camera forward vector is parallel to world up")
    right = right / right_norm
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ position
    return transform.tolist()


def _object(source: dict[str, Any], key: dict[str, Any], scene_path: Path,
            assumptions: list[str], timestamp: float) -> WorldObject:
    return WorldObject(
        object_id=str(source["id"]),
        category="unknown",
        position=key.get("position"),
        rotation=[math.radians(x) for x in key["rotation"]],
        scale=key.get("scale"),
        bbox_3d=None,
        appearance_embedding=None,
        attributes={
            "source": str(scene_path),
            "source_group": "objects",
            "shape": source.get("shape"),
            "scene_version": source.get("_scene_version"),
            "assumptions": assumptions,
            "semantic_category_missing": True,
            "source_rotation_degrees": key["rotation"],
            "rotation_convention": "xyz_radians_Rz_Ry_Rx",
            "source_metadata": {k: v for k, v in source.items()
                                if k not in ("keyframes", "_scene_version")},
        },
        state={"time": timestamp, "matchable": False},
        confidence=None,
        provenance="INFERRED",
    )


def _character(source: dict[str, Any], key: dict[str, Any], scene_path: Path,
               assumptions: list[str], timestamp: float) -> WorldObject:
    joints = np.asarray(key["joints"], dtype=float)
    radius = float(source["radius"])
    lower = joints.min(axis=0) - radius
    upper = joints.max(axis=0) + radius
    return WorldObject(
        object_id=str(source["id"]),
        category="unknown",
        position=((lower + upper) / 2.0).tolist(),
        rotation=None,
        scale=None,
        bbox_3d=(upper - lower).tolist(),
        appearance_embedding=None,
        attributes={
            "source": str(scene_path),
            "source_group": "characters",
            "radius": radius,
            "joints": joints.tolist(),
            "assumptions": assumptions,
            "semantic_category_missing": True,
            "source_metadata": {k: v for k, v in source.items() if k != "keyframes"},
        },
        state={"time": timestamp},
        confidence=None,
        provenance="INFERRED",
    )


def import_scene(scene_path: str | Path, timestamp: float = 0.0,
                 width: int | None = None, height: int | None = None) -> dict[str, Any]:
    path = Path(scene_path).resolve()
    scene = json.loads(path.read_text(encoding="utf-8"))
    if scene.get("version") not in (1, 2):
        raise ValueError("only whitebox scene versions 1 and 2 are supported")
    if (width is None) != (height is None) or (
        width is not None and (width <= 0 or height <= 0)
    ):
        raise ValueError("width and height must both be positive or both omitted")
    assumptions = list(scene.get("assumptions", []))
    objects: list[WorldObject] = []

    for item in scene.get("objects", []):
        item = {**item, "_scene_version": scene.get("version")}
        objects.append(_object(
            item, interpolate_keyframes(item["keyframes"], timestamp),
            path, assumptions, timestamp,
        ))
    for item in scene.get("characters", []):
        objects.append(_character(
            item, interpolate_keyframes(item["keyframes"], timestamp),
            path, assumptions, timestamp,
        ))

    camera_data = scene["camera"]
    camera_key = interpolate_keyframes(camera_data["keyframes"], timestamp)
    position = np.asarray(camera_key["position"], dtype=float)
    target = np.asarray(camera_key["target"], dtype=float)
    camera = Camera(
        projection=camera_data["projection"],
        position=position.tolist(),
        target=target.tolist(),
        fov_degrees=camera_data.get("fov_degrees"),
        ortho_scale=camera_data.get("ortho_scale"),
        width=width,
        height=height,
        world_to_camera=_world_to_camera(position, target),
    )
    ids = [obj.object_id for obj in objects]
    if len(set(ids)) != len(ids):
        raise ValueError("object and character IDs must be unique within the scene")
    return {
        "format": "agentic-world-whitebox-snapshot-v1",
        "source": str(path),
        "scene_version": scene.get("version"),
        "description": scene.get("description"),
        "assumptions": assumptions,
        "timestamp": timestamp,
        "objects": [item.to_dict() for item in objects],
        "camera": camera.to_dict(),
        "coordinate_convention": {
            "world_up": "+Z",
            "world_units": "source_scene_units_not_measured_depth",
            "camera_axes": "+X right, +Y down, +Z forward",
            "rotation": "XYZ radians, Rz Ry Rx",
            "camera_basis_source": "whitebox/render.py project convention",
        },
        "provenance": {
            "source_type": "video_to_whitebox_inverse",
            "default_object_provenance": "INFERRED",
            "confidence_is_unset": True,
            "depth_is_unset": True,
            "semantic_categories_are_not_invented": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--timestamp", type=float, default=0.0)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot = import_scene(args.scene, args.timestamp, args.width, args.height)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "timestamp": args.timestamp,
        "objects": len(snapshot["objects"]),
        "projection": snapshot["camera"]["projection"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
