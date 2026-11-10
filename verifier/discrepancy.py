import numpy as np


def discrepancy(rollout_id, frame_results, config, reference_quality=None):
    matched, unmatched, missing, ambiguous, inconsistent = [], [], [], [], []
    expected_total = observation_total = 0
    conflicts = []
    for frame in frame_results:
        result = frame["matching"]
        for item in result["matched"]:
            item = {**item, "frame_id": frame["frame_id"]}
            matched.append(item)
            if item.get("projection_error_px", 0) > config["projection_tolerance_px"]:
                inconsistent.append(item)
        unmatched.extend(result["unmatched_observations"])
        missing.extend(result["missing_world_objects"])
        ambiguous.extend(result["ambiguous_matches"])
        conflicts.extend({**c, "frame_id": frame["frame_id"]} for c in result.get("observation_conflicts", []))
        expected_total += len(result["matched"])+len(result["missing_world_objects"])
        observation_total += len(result["matched"])+len(result["unmatched_observations"])
    errors = [m["projection_error_px"] for m in matched if m.get("projection_error_px") is not None]
    report = {"rollout_id": rollout_id, "matched": matched, "missing_world_objects": sorted(set(missing)),
            "unmatched_observations": [{k: v for k, v in o.items() if k not in ("visual_embedding", "mask")}
                                       for o in unmatched],
            "ambiguous_matches": ambiguous, "inconsistent": inconsistent, "observation_conflicts": conflicts,
            "summary": {"unmatched_ratio": len(unmatched)/max(1, observation_total),
                        "missing_ratio": len(missing)/max(1, expected_total),
                        "mean_projection_error": float(np.mean(errors)) if errors else None,
                        "inconsistent_ratio": len(inconsistent)/max(1, len(matched)),
                        "observed_count": observation_total, "expected_count": expected_total,
                        "frames": len(frame_results), "conflicting_observation_groups": len(conflicts)}}
    if reference_quality is not None:
        report["reference_assessability"] = reference_quality
        unreliable = set(reference_quality["unverified_world_ids"])
        report["missing_with_unreliable_reference_detection"] = sorted(set(missing) & unreliable)
        report["missing_after_reliable_reference_detection"] = sorted(set(missing) - unreliable)
    return report
