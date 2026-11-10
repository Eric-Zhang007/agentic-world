"""Adapt supplied schemas without changing their databases or inventing calibration."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from state.models import WorldObject


def gta_world(record, source_root=None):
    if record.get("schema_version") == "0.1.0":
        if source_root is None:
            raise ValueError("Legacy GTA conversion requires the supplied GTA source_root")
        sys.path.insert(0, str(Path(source_root) / "src"))
        from wm_engine.dataset.legacy import world_record
        record = world_record(record, record["episode_id"],
                              record.get("world_id", "unreported:" + record["episode_id"])).model_dump()
    world = record.get("world", record)
    result = []
    for entity in world["entities"]:
        kind = entity["entity_type"]
        category = {"player": "person", "ped": "person", "npc": "person",
                    "vehicle": "vehicle"}.get(kind, "unknown")
        rotation = None
        if entity.get("orientation_quaternion") is not None:
            rotation = Rotation.from_quat(entity["orientation_quaternion"]).as_euler("xyz").tolist()
        result.append(WorldObject(
            object_id=entity["capture_entity_id"], category=category,
            position=entity.get("position_xyz"), rotation=rotation, scale=entity.get("scale_xyz"),
            attributes={"source_schema": "gta_canonical_1.0.0", "original_entity": entity,
                        "world_id": world.get("world_id")},
            state={**entity.get("state", {}), "timestamp": entity.get("timestamp"),
                   "linear_velocity": entity.get("linear_velocity"),
                   "despawn_tick": entity.get("despawn_tick")},
            provenance="OBSERVED"))
    return result


def interpolate(keys, timestamp):
    if timestamp < keys[0]["time"] or timestamp > keys[-1]["time"]:
        raise ValueError("Requested time outside recorded keyframes")
    for a, b in zip(keys, keys[1:]):
        if a["time"] <= timestamp <= b["time"]:
            alpha = (timestamp-a["time"]) / (b["time"]-a["time"])
            return {k: (np.asarray(a[k])*(1-alpha)+np.asarray(b[k])*alpha).tolist()
                    for k in a if k != "time"}
    return {k: v for k, v in keys[-1].items() if k != "time"}


def inverse_scene(path, timestamp=0.0, width=832, height=480):
    from scripts.import_whitebox import import_scene
    snapshot = import_scene(path, timestamp, width, height)
    objects = [WorldObject(**o) for o in snapshot["objects"]]
    camera = snapshot["camera"]
    if camera["projection"] == "perspective":
        focal = height / (2*np.tan(np.radians(camera["fov_degrees"])/2))
        camera["intrinsics"] = [[focal, 0, width/2], [0, focal, height/2], [0, 0, 1]]
    else:
        camera["projection_unavailable_reason"] = "Orthographic adapter not calibrated in pinhole matcher"
    return objects, camera
