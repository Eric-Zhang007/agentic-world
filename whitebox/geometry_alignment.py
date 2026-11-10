"""Pixel-corresponded incremental Sim3 alignment; NumPy only, no DB side effects.

Select identical anchor images (and identical pixel grids/crops) from old/new
VGGT reconstructions before calling fit_anchor_sim3. A batch here means paired
anchors, not arbitrary frames from the two reconstructions. The resulting
transform applies to EVERY pointmap in the new reconstruction's model world.
"""

from dataclasses import dataclass

import numpy as np


class AlignmentError(ValueError):
    """Geometric rejection. No usable transform is returned on failure."""

    def __init__(self, reason, metrics):
        super().__init__(reason)
        self.metrics = dict(metrics)


@dataclass(frozen=True)
class AlignmentResult:
    """transform is 4x4: old = scale * rotation @ new + translation.

    inliers is bool with the anchor pixel shape, false for excluded/invalid
    pixels. metrics uses unweighted Euclidean errors in old-world units.
    """

    transform: np.ndarray
    inliers: np.ndarray
    metrics: dict

    def apply(self, pointmaps):
        """Return float64 aligned points of any shape [..., 3], without mutation.

        Supports a full VGGT [N,H,W,3] reconstruction. Nonfinite input points
        remain invalid; this method does not infer a validity/static mask.
        """
        points = np.asarray(pointmaps, dtype=np.float64)
        if points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError('pointmaps must have shape [..., 3]')
        return points @ self.transform[:3, :3].T + self.transform[:3, 3]


def _umeyama(source, target, weights, degeneracy_ratio):
    weights = weights / weights.sum()
    source_mean = weights @ source
    target_mean = weights @ target
    x = source - source_mean
    y = target - target_mean
    # Rank two suffices: planar anchors are valid; lines and tiny support are not.
    for centered, mean in ((x, source_mean), (y, target_mean)):
        singular = np.linalg.svd(centered * np.sqrt(weights[:, None]),
                                 compute_uv=False)
        resolution = 32 * np.finfo(float).eps * max(1., np.linalg.norm(mean))
        if singular[0] <= resolution or singular[1] <= degeneracy_ratio * singular[0]:
            raise ValueError('degenerate point support')
    covariance = (y * weights[:, None]).T @ x
    u, singular, vt = np.linalg.svd(covariance)
    if singular[1] <= degeneracy_ratio ** 2 * singular[0]:
        raise ValueError('degenerate cross covariance')
    signs = np.ones(3)
    signs[-1] = 1. if np.linalg.det(u @ vt) > 0 else -1.
    rotation = (u * signs) @ vt
    scale = float((singular @ signs) / np.sum(weights[:, None] * x * x))
    translation = target_mean - scale * (rotation @ source_mean)
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(translation).all():
        raise ValueError('invalid similarity')
    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return transform, scale


def fit_anchor_sim3(
    new_anchor, old_anchor, *, valid_static_mask, new_confidence, old_confidence,
    inlier_threshold, max_rmse, min_confidence=0., min_inliers=20,
    min_inlier_ratio=0.5, ransac_iterations=512, seed=0,
    degeneracy_ratio=1e-3,
):
    """Fit new-model-world -> old-world using matching anchor pixels only.

    Required arrays:
      new_anchor, old_anchor: equal [H,W,3] or [K,H,W,3] arrays. K indexes
        the SAME anchors in the SAME order; callers establish that identity.
      valid_static_mask: bool [H,W] or [K,H,W], mandatory. True must mean
        valid static scene in BOTH reconstructions, excluding dynamic/HUD
        pixels. No automatic segmentation, correspondence search or resizing.
      new_confidence, old_confidence: same pixel shape, nonnegative scores
        (not necessarily probabilities). Nonfinite/zero scores are excluded;
        negative finite scores are an input error. Both must exceed
        min_confidence. Fit weights are min(new/max(new), old/max(old)) on
        eligible pixels. RANSAC samples uniformly and ranks by inlier count,
        then weighted squared error, so a few confident outliers cannot win.

    inlier_threshold and max_rmse are mandatory positive finite distances in
    OLD world units (not automatically meters). max_rmse gates final inlier
    RMSE; choose it below inlier_threshold for an additional quality gate.
    min_inlier_ratio's denominator is all mask/finite/confidence eligible
    correspondences. min_inliers >= 3; default 20. RANSAC uses fixed bounded
    iterations and a local seeded RNG. Nonconvergence of consensus refitting
    is a rejection. degeneracy_ratio in (0,1) gates second/first singular
    values on BOTH point sets, including the final weighted fit.

    Returns AlignmentResult(transform, inliers, metrics). Reflections are
    never returned (rotation determinant +1, scale > 0). Raises ValueError
    for malformed input/options, AlignmentError (with .metrics) for too few
    points, degeneracy, insufficient consensus, or excessive residual.
    No inputs, database, cameras, or persistent state are modified.
    """
    source_map = np.asarray(new_anchor, dtype=np.float64)
    target_map = np.asarray(old_anchor, dtype=np.float64)
    if (source_map.ndim not in (3, 4) or source_map.shape[-1] != 3
            or source_map.shape != target_map.shape):
        raise ValueError('anchors must have equal [H,W,3] or [K,H,W,3] shape')
    shape = source_map.shape[:-1]
    mask = np.asarray(valid_static_mask)
    if mask.shape != shape or mask.dtype != np.bool_:
        raise ValueError('valid_static_mask must be an explicit boolean pixel mask')
    confidences = [np.asarray(c, dtype=np.float64)
                   for c in (new_confidence, old_confidence)]
    for confidence in confidences:
        if confidence.shape != shape or np.any(confidence < 0):
            raise ValueError('confidence must match pixels and be nonnegative')
    for name, value in (('inlier_threshold', inlier_threshold), ('max_rmse', max_rmse)):
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(name + ' must be positive and finite')
    if not np.isfinite(min_confidence) or min_confidence < 0:
        raise ValueError('min_confidence must be finite and nonnegative')
    if not np.isfinite(min_inlier_ratio) or not 0 < min_inlier_ratio <= 1:
        raise ValueError('min_inlier_ratio must be in (0,1]')
    if not np.isfinite(degeneracy_ratio) or not 0 < degeneracy_ratio < 1:
        raise ValueError('degeneracy_ratio must be in (0,1)')
    for name, value, lower in (('min_inliers', min_inliers, 3),
                               ('ransac_iterations', ransac_iterations, 1)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < lower:
            raise ValueError(name + ' must be an integer >= ' + str(lower))
    eligible = (mask & np.isfinite(source_map).all(axis=-1)
                & np.isfinite(target_map).all(axis=-1))
    for confidence in confidences:
        eligible &= np.isfinite(confidence) & (confidence > min_confidence)
    count = int(eligible.sum())
    metrics = dict(total_pixels=int(mask.size), eligible_count=count,
                   excluded_count=int(mask.size) - count)
    if count < min_inliers:
        raise AlignmentError('insufficient eligible correspondences', metrics)
    source, target = source_map[eligible], target_map[eligible]
    weights = np.minimum(*(c[eligible] / c[eligible].max() for c in confidences))
    # Avoid underflow of extreme confidence ratios silently removing support.
    if np.any(weights <= 0):
        raise ValueError('confidence dynamic range is too large')

    def errors(transform):
        return np.linalg.norm(source @ transform[:3, :3].T + transform[:3, 3] - target, axis=1)

    rng = np.random.default_rng(seed)
    best_mask = None
    best_score = (-1, -np.inf)
    hypotheses = 0
    for _ in range(ransac_iterations):
        indices = rng.choice(count, 3, replace=False)
        try:
            transform, _ = _umeyama(source[indices], target[indices],
                                     weights[indices], degeneracy_ratio)
        except (ValueError, np.linalg.LinAlgError):
            continue
        hypotheses += 1
        residuals = errors(transform)
        inliers = residuals <= inlier_threshold
        n = int(inliers.sum())
        score = (n, -float(np.average(residuals[inliers] ** 2, weights=weights[inliers])) if n else -np.inf)
        if score > best_score:
            best_score, best_mask = score, inliers
    metrics.update(ransac_iterations=ransac_iterations, valid_hypotheses=hypotheses)
    required = max(min_inliers, int(np.ceil(min_inlier_ratio * count)))
    if best_mask is None:
        raise AlignmentError('degenerate correspondences: no valid hypothesis', metrics)
    inliers = best_mask
    for iteration in range(20):
        n = int(inliers.sum())
        metrics.update(inlier_count=n, inlier_ratio=n / count)
        if n < required:
            raise AlignmentError('insufficient inlier consensus', metrics)
        try:
            transform, scale = _umeyama(source[inliers], target[inliers],
                                         weights[inliers], degeneracy_ratio)
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise AlignmentError('degenerate inlier geometry', metrics) from exc
        residuals = errors(transform)
        updated = residuals <= inlier_threshold
        if np.array_equal(updated, inliers):
            break
        inliers = updated
    else:
        raise AlignmentError('inlier refinement did not converge', metrics)
    selected = residuals[inliers]
    rmse = float(np.sqrt(np.mean(selected ** 2)))
    metrics.update(scale=scale, rmse=rmse, median_residual=float(np.median(selected)),
                   max_residual=float(selected.max()),
                   refinement_iterations=iteration + 1,
                   inlier_threshold=float(inlier_threshold), max_rmse=float(max_rmse))
    if not np.isfinite(rmse) or rmse > max_rmse:
        raise AlignmentError('inlier residual exceeds max_rmse', metrics)
    pixel_inliers = np.zeros(shape, dtype=bool)
    pixel_inliers[eligible] = inliers
    return AlignmentResult(transform, pixel_inliers, metrics)
