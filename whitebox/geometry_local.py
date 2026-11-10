"""Region-wise evidence in a fixed world frame; never warp accepted geometry."""
import numpy as np


def assess_region(old, incoming, *, threshold, min_points=40):
    old, incoming = np.asarray(old, float), np.asarray(incoming, float)
    if old.shape != incoming.shape or old.ndim != 2 or old.shape[1] != 3:
        raise ValueError('Expected pixel-corresponded Nx3 arrays')
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Positive finite residual threshold required')
    finite = np.isfinite(old).all(axis=1) & np.isfinite(incoming).all(axis=1)
    old, incoming = old[finite], incoming[finite]
    if len(old) < min_points:
        return {'status': 'INSUFFICIENT', 'points': len(old), 'apply': False}
    # Interleaved rows reserve actual correspondences for evaluating the offset.
    train, held = np.arange(len(old)) % 2 == 0, np.arange(len(old)) % 2 == 1
    displacement = old-incoming
    offset = np.median(displacement[train], axis=0)
    global_error = np.linalg.norm(displacement[held], axis=1)
    local_error = np.linalg.norm(displacement[held]-offset, axis=1)
    before, after = np.quantile(global_error, .9), np.quantile(local_error, .9)
    if before <= threshold:
        status = 'CONSISTENT_KEEP_OLD'
    elif after <= threshold:
        status = 'LOCAL_OFFSET_REVIEW'
    else:
        status = 'SHAPE_CONFLICT_KEEP_OLD'
    matrix = np.eye(4); matrix[:3, 3] = offset
    return {'status': status, 'points': len(old), 'train_points': int(train.sum()),
            'heldout_points': int(held.sum()), 'threshold_old_units': threshold,
            'before_heldout_p90': float(before), 'after_heldout_p90': float(after),
            'before_heldout_median': float(np.median(global_error)),
            'after_heldout_median': float(np.median(local_error)),
            'candidate_local_offset': offset.tolist(), 'candidate_transform': matrix.tolist(),
            'offset_norm': float(np.linalg.norm(offset)), 'apply': False,
            'evidence': 'Same RGB anchor reconstructed twice; consistency, not absolute geometric truth'}
