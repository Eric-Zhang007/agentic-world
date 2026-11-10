import numpy as np


def iou(a, b):
    if a is None or b is None:
        return None
    a, b = np.asarray(a), np.asarray(b)
    intersect = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    union = np.maximum(0, a[2:]-a[:2]).prod()+np.maximum(0, b[2:]-b[:2]).prod()-intersect
    return float(intersect/union) if union else 0.0


def semantic(a, b):
    if a in (None, "unknown") or b in (None, "unknown"):
        return None
    if a == b:
        return 0.0
    if a == "vehicle" and b in ("car", "truck", "bus", "motorcycle"):
        return 0.1
    return 1.0


def pair_cost(world, observation, projection, track_world, config):
    parts, metrics = {}, {}
    semantic_cost = semantic(world.category, observation.category)
    if semantic_cost is not None:
        parts["semantic"] = semantic_cost
        metrics["semantic_score"] = 1-semantic_cost
    if world.appearance_embedding is not None and observation.visual_embedding is not None:
        a, b = np.asarray(world.appearance_embedding), np.asarray(observation.visual_embedding)
        if a.shape == b.shape and np.linalg.norm(a) and np.linalg.norm(b):
            similarity = float(np.clip(a @ b / (np.linalg.norm(a)*np.linalg.norm(b)), -1, 1))
            parts["visual"] = (1-similarity)/2
            metrics["visual_similarity"] = similarity
    if world.position is not None and observation.world_position is not None:
        distance = float(np.linalg.norm(np.asarray(world.position)-observation.world_position))
        parts["spatial"] = min(distance/config["spatial_scale"], 1)
        metrics["spatial_distance"] = distance
    if projection and projection.get("center"):
        box = observation.bbox_xyxy
        center = np.asarray([(box[0]+box[2])/2, (box[1]+box[3])/2])
        error = float(np.linalg.norm(center-projection["center"]))
        overlap = iou(projection.get("bbox"), box)
        parts["projection"] = min(error/config["projection_scale"], 1)
        metrics.update(projection_error_px=error, bbox_iou=overlap)
        if observation.depth is not None:
            metrics["depth_error"] = abs(observation.depth-projection["depth"])
    if observation.track_id in track_world:
        parts["temporal"] = float(track_world[observation.track_id] != world.object_id)
    weights = config["weights"]
    denominator = sum(weights[k] for k in parts)
    cost = sum(weights[k]*v for k, v in parts.items())/denominator if denominator else 1.0
    if semantic_cost == 1:
        cost = 1.0  # Incompatible known categories cannot become identity matches.
    return cost, metrics
