"""Local observed facades and conservative duplicate-mask suppression.

Coordinates are already ground aligned (Y up). Distances use the input world's
units. No scene compilation, inferred hidden volume, or source mutation occurs.
"""
import math

import numpy as np


def deduplicate_masks(masks, *, iou_threshold=.85, containment_threshold=.97,
                      min_area_ratio=.75):
    """Return kept input records and an auditable suppression report.

    Records require a 2-D binary ``mask`` and ``category`` (or ``semantic_id``).
    Optional ``frame_id``/``image_id`` scope comparisons; differing ``entity_id``
    values (including known versus absent) veto suppression. ``id`` is an
    observation ID, not an entity ID.
    Exact normalized labels are used: building and wall are NOT synonyms.
    Containment also requires comparable area to preserve small nested objects.
    Greedy suppression never unions masks or follows transitive overlap chains.
    Unknown semantics and empty masks remain separate. Image-only overlap cannot
    prove physical identity; callers with identity evidence should supply it.
    """
    for value in (iou_threshold, containment_threshold, min_area_ratio):
        if not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError('Overlap thresholds must be in (0, 1]')
    masks = list(masks)
    arrays, semantics, scores = [], [], []
    for item in masks:
        a = np.asarray(item['mask'])
        if a.ndim != 2 or not np.isin(a, [0, 1]).all():
            raise ValueError('Masks must be 2-D binary arrays')
        arrays.append(a.astype(bool, copy=False))
        semantic = item.get('semantic_id', item.get('category', ''))
        semantic = ' '.join(str(semantic).lower().strip().rstrip('.').split())
        semantics.append(semantic if semantic not in ('', 'unknown', 'none') else None)
        score = float(item.get('confidence', 0.))
        if not np.isfinite(score):
            raise ValueError('Mask confidence must be finite')
        scores.append(score)
    areas = [int(a.sum()) for a in arrays]
    kept, suppressed = [], []
    for i in sorted(range(len(masks)), key=lambda i: (-scores[i], -areas[i], i)):
        match = None
        for j in kept:
            if semantics[i] is None or semantics[i] != semantics[j]:
                continue
            if any(masks[i].get(key) != masks[j].get(key)
                   for key in ('frame_id', 'image_id')):
                continue
            ai, aj = masks[i].get('entity_id'), masks[j].get('entity_id')
            if ai != aj:
                continue
            if arrays[i].shape != arrays[j].shape:
                raise ValueError('Masks in the same image must have the same shape')
            if min(areas[i], areas[j]) == 0:
                continue
            intersection = int(np.count_nonzero(arrays[i] & arrays[j]))
            iou = intersection / (areas[i] + areas[j] - intersection)
            containment = intersection / min(areas[i], areas[j])
            area_ratio = min(areas[i], areas[j]) / max(areas[i], areas[j])
            if iou >= iou_threshold or (containment >= containment_threshold
                                       and area_ratio >= min_area_ratio):
                match = dict(index=i, kept_index=j, id=masks[i].get('id', i),
                             kept_id=masks[j].get('id', j), iou=iou,
                             containment=containment, area_ratio=area_ratio)
                break
        if match is None:
            kept.append(i)
        else:
            suppressed.append(match)
    kept.sort()
    return [masks[i] for i in kept], dict(kept_indices=kept, suppressed=suppressed)


def _patches(uv, cell_size, max_extent, min_fill, min_points):
    """Sparse grid components, then occupied rectangular subdivisions.

    No dense grid proportional to the world extent is allocated. Subdivision
    limits both unsupported gaps inside connected shapes and long thin boxes.
    """
    cells = np.floor((uv - uv.min(axis=0)) / cell_size).astype(np.int64)
    buckets = {}
    for i, cell in enumerate(cells):
        buckets.setdefault(tuple(cell), []).append(i)
    unseen = set(buckets)
    components = []
    for start in buckets:
        if start not in unseen:
            continue
        unseen.remove(start)
        stack, indices = [start], []
        while stack:
            x, y = stack.pop()
            indices.extend(buckets[(x, y)])
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = (x + dx, y + dy)
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(np.asarray(indices, dtype=int))
    stack = components
    while stack:
        ids = stack.pop()
        if len(ids) < min_points:
            continue
        extent = np.ptp(uv[ids], axis=0)
        occupied = np.unique(cells[ids], axis=0)
        grid_extent = np.ptp(occupied, axis=0) + 1
        fill = len(occupied) / float(np.prod(grid_extent.astype(float)))
        if extent.max() <= max_extent and fill >= min_fill:
            yield ids, fill
            continue
        axis = int(np.argmax(extent))
        mid = (uv[ids, axis].min() + uv[ids, axis].max()) / 2
        lower = uv[ids, axis] <= mid
        if lower.all() or not lower.any():
            continue
        stack.extend((ids[lower], ids[~lower]))


def fit_facade_boxes(points, *, min_points=30, plane_threshold=.15,
                     cell_size=1., max_patch_extent=10., min_fill=.6,
                     display_thickness=.05, max_planes=24,
                     ransac_iterations=256, seed=17):
    """Fit vertical planes and locally supported thin boxes to an (N,3) cloud.

    Returns ``boxes``, original ``unassigned_indices``/``invalid_indices`` and
    coverage. Each box uses geometry_assets' position/size/rotation_y convention
    and original ``point_indices``. Size Z is ONLY display thickness, never a
    building depth estimate. All inputs are copied or read without mutation.

    Plane tolerance, connectivity grid and maximum patch extent are explicit
    world-unit parameters, not inferred metric scale. RANSAC is deterministic;
    finite rejected points stay unassigned (no whole-cloud fallback box).
    """
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError('points must have shape (N, 3)')
    for value in (plane_threshold, cell_size, max_patch_extent, display_thickness):
        if not np.isfinite(value) or value <= 0:
            raise ValueError('Geometric distances must be positive and finite')
    for value in (min_points, max_planes, ransac_iterations):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError('Counts must be positive integers')
    if min_points < 3 or not np.isfinite(min_fill) or not 0 < min_fill <= 1:
        raise ValueError('min_points >= 3 and min_fill in (0, 1] required')
    finite = np.isfinite(p).all(axis=1)
    remaining = np.flatnonzero(finite)
    assigned = np.zeros(len(p), dtype=bool)
    boxes = []
    rng = np.random.default_rng(seed)
    for plane_id in range(max_planes):
        if len(remaining) < min_points:
            break
        horizontal = p[remaining][:, [0, 2]]
        sample = horizontal[rng.choice(len(horizontal), min(6000, len(horizontal)), replace=False)]
        best_count, best = 0, None
        for _ in range(ransac_iterations):
            a, b = sample[rng.choice(len(sample), 2, replace=False)]
            delta = b - a
            length = np.linalg.norm(delta)
            if length < 2 * plane_threshold:
                continue
            normal = np.array([-delta[1], delta[0]]) / length
            offset = float(a @ normal)
            count = int(np.count_nonzero(abs(sample @ normal - offset) <= plane_threshold))
            if count > best_count:
                best_count, best = count, (normal, offset)
        if best is None:
            break
        normal, offset = best
        for _ in range(3):
            inside = abs(horizontal @ normal - offset) <= plane_threshold
            if inside.sum() < min_points:
                break
            h = horizontal[inside]
            center = h.mean(axis=0)
            _, vectors = np.linalg.eigh((h - center).T @ (h - center))
            normal = vectors[:, 0]
            offset = float(center @ normal)
        inside = abs(horizontal @ normal - offset) <= plane_threshold
        if inside.sum() < min_points:
            break
        plane_indices = remaining[inside]
        remaining = remaining[~inside]
        # Canonical sign keeps output stable across eigenvector sign choices.
        if normal[np.argmax(abs(normal))] < 0:
            normal, offset = -normal, -offset
        tangent = np.array([normal[1], -normal[0]])
        yaw = math.atan2(-tangent[1], tangent[0])
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        local = p[plane_indices] @ rotation
        for patch, fill in _patches(local[:, :2], cell_size, max_patch_extent,
                                    min_fill, min_points):
            ids = plane_indices[patch]
            uv = local[patch, :2]
            singular = np.linalg.svd(uv - uv.mean(axis=0), compute_uv=False)
            if singular[1] <= np.finfo(float).eps * max(uv.shape) * singular[0]:
                continue
            lo, hi = local[patch, :2].min(axis=0), local[patch, :2].max(axis=0)
            # A line or a thin horizontal stripe does not establish a facade.
            if np.min(hi - lo) <= 2 * plane_threshold:
                continue
            residual = abs(local[patch, 2] - offset)
            position = np.r_[(lo + hi) / 2, offset] @ rotation.T
            boxes.append(dict(position=position.tolist(),
                              size=[float(hi[0]-lo[0]), float(hi[1]-lo[1]), display_thickness],
                              rotation_y=yaw, kind='box', solid=False,
                              provenance='INFERRED', point_indices=ids.tolist(),
                              geometry_evidence=dict(
                                  surface='observed_facade', visible_surface_only=True,
                                  occluded_extent_known=False, hidden_thickness=None,
                                  display_thickness=display_thickness,
                                  thickness_role='rendering_only_not_measured',
                                  plane_id=plane_id, point_count=len(ids), grid_fill=fill,
                                  plane_normal=[float(normal[0]), 0., float(normal[1])],
                                  plane_offset=offset,
                                  residual_median=float(np.median(residual)),
                                  residual_max=float(residual.max()))))
            assigned[ids] = True
    return dict(boxes=boxes, unassigned_indices=np.flatnonzero(finite & ~assigned).tolist(),
                invalid_indices=np.flatnonzero(~finite).tolist(),
                coverage=float(assigned.sum() / finite.sum()) if finite.any() else 0.,
                parameters=dict(min_points=min_points, plane_threshold=plane_threshold,
                                cell_size=cell_size, max_patch_extent=max_patch_extent,
                                min_fill=min_fill, display_thickness=display_thickness,
                                max_planes=max_planes, ransac_iterations=ransac_iterations,
                                seed=seed))
