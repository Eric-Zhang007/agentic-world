from itertools import product

import numpy as np
from scipy.spatial.transform import Rotation


def project_object(obj, camera):
    if camera is None or obj.position is None:
        return None
    if "intrinsics" not in camera or "world_to_camera" not in camera:
        return None
    transform = np.asarray(camera["world_to_camera"], float)
    intrinsics = np.asarray(camera["intrinsics"], float)
    anchor = np.asarray(obj.position,float) + np.asarray(
        obj.attributes.get("projection_anchor_offset_xyz",[0,0,0]),float)
    center = transform @ np.r_[anchor, 1.0]
    if center[2] <= 0:
        return {"visible": False}
    projected = intrinsics @ center[:3]
    center_px = projected[:2] / projected[2]
    bbox = None
    if obj.bbox_3d is not None:
        corners = np.asarray(list(product((-0.5, 0.5), repeat=3)))
        corners *= np.asarray(obj.bbox_3d) * np.asarray(obj.scale or [1, 1, 1])
        corners = Rotation.from_euler("xyz", obj.rotation or [0, 0, 0]).apply(corners)
        corners += np.asarray(obj.position)
        points = (transform @ np.c_[corners, np.ones(8)].T).T[:, :3]
        if np.any(points[:, 2] <= 0.01):
            return None  # Near-plane clipping is not approximated as a full bbox.
        pixels = (intrinsics @ points.T).T
        pixels = pixels[:, :2] / pixels[:, 2:]
        bbox = [*pixels.min(axis=0).tolist(), *pixels.max(axis=0).tolist()]
    w, h = camera["width"], camera["height"]
    for x0,y0,x1,y1 in camera.get("invalid_regions_normalized", []):
        if x0*w <= center_px[0] < x1*w and y0*h <= center_px[1] < y1*h:
            return {"center":center_px.tolist(),"bbox":bbox,"depth":float(center[2]),
                    "visible":False,"excluded_reason":"HUD_invalid_projection_anchor"}
    visible = (0 <= center_px[0] < w and 0 <= center_px[1] < h) if bbox is None else (
        bbox[2] > 0 and bbox[3] > 0 and bbox[0] < w and bbox[1] < h)
    return {"center": center_px.tolist(), "bbox": bbox, "depth": float(center[2]), "visible": bool(visible)}


def get_visible_objects(objects, camera):
    return [o for o in objects if o.state.get("visible", True)
            and (project_object(o, camera) or {}).get("visible", True)]
