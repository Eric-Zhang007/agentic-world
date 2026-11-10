"""Separate detector ambiguity from evidence of renderer inconsistency."""
from matching.costs import iou, semantic


def conflicting_labels(observations, overlap_threshold=0.85):
    conflicts = []
    for i, a in enumerate(observations):
        for b in observations[i+1:]:
            overlap = iou(a.bbox_xyxy, b.bbox_xyxy)
            if semantic(a.category, b.category) == 1 and overlap >= overlap_threshold:
                conflicts.append({"obs_ids": [a.obs_id, b.obs_id],
                                  "categories": [a.category, b.category], "bbox_iou": overlap,
                                  "reason": "incompatible labels on nearly identical image support"})
    return conflicts


def reference_assessability(matching):
    matched = {m["world_id"] for m in matching["matched"]}
    missing = set(matching["missing_world_objects"])
    return {"source": "compiled_reference_not_independent_ground_truth",
            "reliably_detected_world_ids": sorted(matched),
            "unverified_world_ids": sorted(missing),
            "observation_conflicts": matching.get("observation_conflicts", []),
            "warning": "Failure on the input proxy prevents attributing the same miss solely to H3."}
