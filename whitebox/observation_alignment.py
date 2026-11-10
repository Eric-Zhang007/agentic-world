"""Sparse image maintenance candidates; never commit or interpolate history."""
from copy import deepcopy
import numpy as np
from scipy.optimize import least_squares

from whitebox.image_alignment import landmark_points, project_points, pixels_to_plane
from whitebox.world import validate_scene


def align_observation(scene, observation):
    """Fit camera translation from static anchors, then grounded actor roots.

    Rotation, intrinsics and geometry stay fixed. This deliberately restricted
    diagnostic exposes residual disagreement rather than moving static assets.
    Camera rotation or moving support surfaces require a different hypothesis.
    """
    validate_scene(scene)
    width, height = observation['width'], observation['height']
    anchors = observation['static_landmarks']
    entities = {e['id']: e for e in scene['entities']}
    actor_ids = [row['id'] for row in observation.get('actor_roots', [])]
    if len(set(actor_ids)) != len(actor_ids):
        raise ValueError('Duplicate actor root updates')
    if any(identity not in entities for identity in actor_ids):
        raise ValueError('Unknown actor root ID')
    if len(anchors) < 3 or len({r['id'] for r in anchors}) != len(anchors):
        raise ValueError('At least three distinct static anchors required')
    if any(entities[r['id']]['kind'] == 'actor' for r in anchors):
        raise ValueError('Actors cannot anchor the static world')
    points = landmark_points(scene, anchors)
    target = np.asarray([r['pixel'] for r in anchors], float)
    if not np.isfinite(target).all():
        raise ValueError('Nonfinite image landmarks')
    def camera_at(delta):
        camera = deepcopy(scene['camera'])
        for field in ('position', 'target'):
            camera[field] = (np.asarray(camera[field])+delta).tolist()
        return camera
    def residual(delta):
        uv, valid = project_points(points, camera_at(delta), width, height)
        return np.where(valid[:, None], np.nan_to_num(uv)-target, 1e5).ravel()
    bounds = np.asarray(observation['camera_translation_bounds'], float)
    if bounds.shape != (3, 2):
        raise ValueError('Expected three explicit camera translation bounds')
    fit = least_squares(residual, np.zeros(3), bounds=(bounds[:, 0], bounds[:, 1]))
    rank = int(np.linalg.matrix_rank(fit.jac))
    if not fit.success or rank != 3:
        raise ValueError('Camera translation is not identifiable from anchors')
    after = deepcopy(scene)
    after['camera'] = camera_at(fit.x)
    uv, valid = project_points(points, after['camera'], width, height)
    if not valid.all():
        raise ValueError('Fitted anchors cross near plane')
    changed = {e['id']: e for e in after['entities']}
    actors = []
    skipped_actor_roots = []
    for row in observation.get('actor_roots', []):
        entity = changed[row['id']]
        if entity['kind'] != 'actor':
            raise ValueError('Only actor roots can move in this diagnostic')
        visibility = row.get('contact_visibility', 'unknown')
        if visibility not in ('visible', 'occluded', 'unknown'):
            raise ValueError('Invalid contact_visibility: expected visible, occluded, or unknown')
        if visibility != 'visible':
            skipped_actor_roots.append(dict(id=row['id'],
                reason='Contact visibility is not visible', contact_visibility=visibility))
            continue
        pixel = np.asarray(row['pixel'], dtype=float)
        if pixel.shape != (2,) or not np.isfinite(pixel).all():
            raise ValueError('Visible actor root requires a finite pixel of shape (2,)')
        if not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
            raise ValueError('Visible actor root pixel is outside image bounds')
        support = row['support_plane']
        hit, valid = pixels_to_plane([row['pixel']], after['camera'],width,height,
                                    normal=support['normal'],offset=support['offset'])
        if not valid[0]:
            raise ValueError('Actor support ray has no forward plane intersection')
        delta = hit[0] - entity['position']
        entity['position'] = hit[0].tolist()
        # A root observation contains no new articulation or grip evidence.
        if 'pose' in entity:
            entity['pose'] = {key:(np.asarray(point)+delta).tolist()
                              for key,point in entity['pose'].items()}
        for held in after['entities']:
            if held['kind'] != 'staff' or held.get('holder') != entity['id']:
                continue
            held['position'] = (np.asarray(held['position'])+delta).tolist()
            if 'weapon' in held:
                for key in ('a','b'):
                    held['weapon'][key] = (np.asarray(held['weapon'][key])+delta).tolist()
        actors.append(dict(id=row['id'],position=hit[0].tolist(),support_plane=support))
    validate_scene(after)
    return after, dict(world_committed=False, timestamp_sec=observation['timestamp_sec'],
        camera_translation=fit.x.tolist(), camera_fit_rank=rank,
        static_errors_px=np.linalg.norm(uv-target,axis=1).tolist(),
        actors=actors, skipped_actor_roots=skipped_actor_roots, static_geometry_unchanged=True,
        limitation='Sparse observation candidate, not a motion trajectory. Fixed camera orientation and supplied support planes. Contact visibility is a supplied annotation, not algorithmic proof of visibility or contact; no occlusion or contact verification.')
