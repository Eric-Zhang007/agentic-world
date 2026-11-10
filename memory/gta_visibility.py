"""Conservative visibility evidence: geometry is not an RGB visibility label."""
from itertools import product

import numpy as np
from scipy.spatial.transform import Rotation


def geometry_evidence(obj, origin, basis, near=0.05):
    try:
        origin = np.asarray(origin, dtype=float)
        basis = np.asarray(basis, dtype=float)
        position = np.asarray(obj['position'], dtype=float)
        lo = np.asarray(obj['model_bounds_min'], dtype=float)
        hi = np.asarray(obj['model_bounds_max'], dtype=float)
        quat = np.asarray(obj['orientation_quaternion'], dtype=float)
        if any(x.shape != (3,) for x in (origin, position, lo, hi)) or basis.shape != (3, 3) or quat.shape != (4,):
            raise ValueError('shape')
        if not all(np.isfinite(x).all() for x in (origin, basis, position, lo, hi, quat)) or np.any(lo >= hi):
            raise ValueError('invalid geometry')
        rotation = Rotation.from_quat(quat)
        local_origin = rotation.inv().apply(origin-position)
        corners = rotation.apply(np.array(list(product(*zip(lo, hi)))))+position
        q = np.linalg.solve(basis, (corners-origin).T).T
        if np.all(local_origin >= lo) and np.all(local_origin <= hi):
            return 'CAMERA_INSIDE_OBB'
        if np.all(q[:, 2] <= near):
            return 'BEHIND_NEAR_PLANE'
        if np.any(q[:, 2] <= near):
            return 'NEAR_PLANE_INTERSECTION'
        uv = q[:, :2]/q[:, 2:3]+.5
        if np.any(uv.max(axis=0) <= 0) or np.any(uv.min(axis=0) >= 1):
            return 'OUTSIDE_IMAGE'
        return 'PROJECTABLE_OBB'
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        return 'INVALID_GEOMETRY'


def visibility_evidence(obj, origin, basis, entities):
    geometry = geometry_evidence(obj, origin, basis)
    vehicle = obj.get('current_vehicle_network_id')
    occupied = vehicle not in (None, 0, '0')
    carriers = [str(e['dataset_entity_id']) for e in entities
                if occupied and e.get('network_id') is not None
                and str(e['network_id']) == str(vehicle)
                and str(e.get('dataset_entity_id')) != str(obj.get('dataset_entity_id'))
                and e.get('dataset_entity_id') is not None]
    return {
        'geometry_status': geometry,
        'rgb_visibility': 'UNKNOWN',
        'static_occlusion_checked': False,
        'in_vehicle': occupied,
        'carrier_entity_ids': sorted(set(carriers)),
        'review_reasons': ([geometry] if geometry != 'PROJECTABLE_OBB' else [])
                          + (['VEHICLE_OCCUPANT_REQUIRES_RGB_EVIDENCE'] if occupied else [])
                          + ['STATIC_OCCLUSION_UNVERIFIED'],
        'appearance_reference_approved': False,
    }
