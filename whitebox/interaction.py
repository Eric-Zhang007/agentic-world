"""Coarse interaction contracts; numerical checks are not visual evidence."""
import math
import numpy as np


def arrival_distance(scene, contract):
    if contract.get('kind') != 'arrival':
        raise ValueError('Only arrival contracts are currently supported')
    target = np.asarray(contract['target'], float)
    radius = contract['radius']
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError('Arrival target must be a finite 3D point')
    if not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius <= 0:
        raise ValueError('Arrival radius must be positive and finite')
    actor = next(e for e in scene['entities'] if e['id'] == contract['actor_id'])
    if actor['kind'] != 'actor':
        raise ValueError('Arrival subject must be an actor')
    position = np.asarray(actor['position'], float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError('Invalid actor position')
    distance = float(np.linalg.norm((position-target)[[0, 2]]))
    return {'distance': distance, 'radius': radius, 'inside': distance <= radius + 1e-8,
            'scope': 'coarse world coordinates, not independently measured video geometry'}


def check_arrival_review(scene, contract, review):
    """Reject an internally contradictory completion before committing history."""
    geometry = arrival_distance(scene, contract)
    assessment = review.get('interaction_assessment', {})
    if assessment.get('status') not in {'SUPPORTED', 'CONTRADICTED', 'UNOBSERVED'}:
        raise ValueError('Arrival review requires an explicit visual assessment')
    evidence = assessment.get('evidence')
    if not isinstance(evidence, list) or not evidence or not all(isinstance(x, str) and x.strip() for x in evidence):
        raise ValueError('Arrival assessment needs evidence or a visibility limitation')
    if assessment['status'] == 'SUPPORTED' and not geometry['inside']:
        raise ValueError('Visual arrival claim conflicts with the corrected coarse endpoint')
    if review['action_complete'] and assessment['status'] != 'SUPPORTED':
        raise ValueError('Arrival cannot be complete without supporting visual evidence')
    return dict(geometry, visual_status=assessment['status'])
