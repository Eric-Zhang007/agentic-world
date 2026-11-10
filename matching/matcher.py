from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from compiler.projection import get_visible_objects, project_object
from matching.costs import pair_cost
from perception.quality import conflicting_labels


@dataclass
class MatchingResult:
    matched: list = field(default_factory=list)
    unmatched_observations: list = field(default_factory=list)
    missing_world_objects: list = field(default_factory=list)
    ambiguous_matches: list = field(default_factory=list)
    observation_conflicts: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def match(objects, observations, camera, config, track_world=None):
    track_world = track_world if track_world is not None else {}
    expected = [o for o in get_visible_objects(objects, camera) if o.state.get("matchable", True)]
    n, m = len(expected), len(observations)
    result = MatchingResult()
    result.observation_conflicts = conflicting_labels(observations)
    conflicted = {obs_id for group in result.observation_conflicts for obs_id in group["obs_ids"]}
    used_w, used_o = set(), set()
    if n and m:
        costs = np.ones((n, m))
        metrics = {}
        for i, obj in enumerate(expected):
            projection = project_object(obj, camera)
            for j, obs in enumerate(observations):
                costs[i, j], metrics[i, j] = pair_cost(obj, obs, projection, track_world, config)
                if obs.obs_id in conflicted and costs[i, j] < config["max_cost"]:
                    result.ambiguous_matches.append({
                        "world_id": obj.object_id, "obs_id": obs.obs_id,
                        "reason": "conflicting detector labels on overlapping support"})
                    costs[i, j] = 1.0
        # Dummy columns let each world object remain unassigned; rejected pairs
        # cannot steal observations from otherwise valid low-cost assignments.
        augmented = np.c_[costs, np.full((n, n), config["max_cost"])]
        rows, cols = linear_sum_assignment(augmented)
        for i, j in zip(rows, cols):
            if j >= m or costs[i, j] >= config["max_cost"]:
                continue
            alternatives = [costs[k, j] for k in range(n) if k != i] + [
                costs[i, k] for k in range(m) if k != j]
            item = {"world_id": expected[i].object_id, "obs_id": observations[j].obs_id,
                    "track_id": observations[j].track_id, "cost": float(costs[i, j]),
                    "match_confidence": float(1-costs[i, j]), **metrics[i, j]}
            if alternatives and min(alternatives)-costs[i, j] < config["ambiguity_margin"]:
                result.ambiguous_matches.append(item)
                continue
            result.matched.append(item)
            used_w.add(i)
            used_o.add(j)
            if observations[j].track_id is not None:
                track_world[observations[j].track_id] = expected[i].object_id
    result.missing_world_objects = [o.object_id for i, o in enumerate(expected) if i not in used_w]
    result.unmatched_observations = [o.to_dict() for j, o in enumerate(observations) if j not in used_o]
    return result
