"""Read-only ordinal comparison of manual pixel patches and scene origins."""
from itertools import combinations

import numpy as np

from whitebox.image_alignment import camera_coordinates


def compare_depth_order(depth, scene, regions, *, nearer_is_larger):
    """Regions map entity IDs to half-open integer [x0,y0,x1,y1] boxes.

    Positive margins mean A is nearer than B. Margins retain separate units.
    Invalid samples produce unavailable pairs; exact ties are unresolved.
    """
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError('depth must be a 2D array')
    if not isinstance(nearer_is_larger, bool):
        raise ValueError('nearer_is_larger must be explicit bool')
    entities = {e['id']: e for e in scene['entities']}
    rows = {}
    h, w = depth.shape
    for name, box in regions.items():
        row = {'status': 'missing_region', 'median': None, 'camera_depth': None,
               'finite_pixels': 0, 'total_pixels': 0}
        rows[name] = row
        if box is None:
            continue
        if len(box) != 4 or any(not isinstance(v, (int, np.integer)) for v in box):
            row['status'] = 'invalid_region'
            continue
        x0, y0, x1, y1 = box
        if not (0 <= x0 <= x1 <= w and 0 <= y0 <= y1 <= h):
            row['status'] = 'invalid_region'
            continue
        patch = depth[y0:y1, x0:x1]
        finite = patch[np.isfinite(patch)]
        row.update(total_pixels=int(patch.size), finite_pixels=int(finite.size))
        if not finite.size:
            row['status'] = 'empty_region' if not patch.size else 'nonfinite_region'
            continue
        row['median'] = float(np.median(finite))
        entity = entities.get(name)
        if entity is None:
            row['status'] = 'missing_entity'
            continue
        z = float(camera_coordinates([entity['position']], scene['camera'])[0, 2])
        if not np.isfinite(z) or z <= 0:
            row['status'] = 'invalid_camera_depth'
            continue
        row.update(status='ok', camera_depth=z)
    pairs = []
    for a, b in combinations(rows, 2):
        pair = dict(a=a, b=b, predicted_margin=None, world_margin=None,
                    disagreement=None, status='unavailable')
        ra, rb = rows[a], rows[b]
        if ra['status'] == rb['status'] == 'ok':
            pm = (ra['median'] - rb['median']) * (1 if nearer_is_larger else -1)
            wm = rb['camera_depth'] - ra['camera_depth']
            resolved = pm != 0 and wm != 0
            pair.update(predicted_margin=pm, world_margin=wm,
                        disagreement=bool(pm * wm < 0) if resolved else None,
                        status='resolved' if resolved else 'tie')
        pairs.append(pair)
    return dict(nearer_is_larger=nearer_is_larger, regions=rows, pairs=pairs,
                evidence='Manual interior patches; predicted relative order versus scene origin order; no ground truth',
                world_committed=False)
