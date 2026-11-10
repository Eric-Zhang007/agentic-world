"""Compile segmented point maps into coarse, explicitly inferred scene proposals."""
import math

import numpy as np


def transform_points(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def fit_ground(points, threshold, seed=17):
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Plane threshold must be positive and finite')
    points = np.asarray(points, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 80:
        raise ValueError('At least 80 valid ground points required')
    rng = np.random.default_rng(seed)
    points = points[rng.choice(len(points), min(len(points), 12000), replace=False)]
    best = None
    for _ in range(160):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b-a, c-a)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal /= length
        # The initial camera is roughly upright, but its pitch is not assumed zero.
        if abs(normal[1]) < math.cos(math.radians(55)):
            continue
        inside = abs((points-a) @ normal) < threshold
        if best is None or inside.sum() > best.sum():
            best = inside
    if best is None or best.mean() < .45:
        raise ValueError('No reliable dominant ground plane; supply a better ground mask')
    selected = points[best]
    center = selected.mean(axis=0)
    _, singular, vt = np.linalg.svd(selected-center, full_matrices=False)
    if singular[1] < threshold:
        raise ValueError('Ground points are nearly collinear')
    normal = vt[-1]
    if normal[1] < 0:
        normal = -normal
    return normal, center, {'inlier_fraction': float(best.mean()),
                            'median_residual': float(np.median(abs((selected-center) @ normal)))}


def rotation_between(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if c < -1+1e-8:
        raise ValueError('Antiparallel ground normal')
    skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3)+skew+(skew @ skew)/(1+c)


def upright_box(points, min_size=.02):
    points = np.asarray(points, float)
    if len(points) < 20 or not np.isfinite(points).all():
        raise ValueError('Insufficient finite box points')
    horizontal = points[:, [0, 2]]
    _, axes = np.linalg.eigh(np.cov(horizontal.T))
    direction = axes[:, -1]
    yaw = math.atan2(-direction[1], direction[0])
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    local = points @ rotation
    lo, hi = np.quantile(local, [.02, .98], axis=0)
    size = np.maximum(hi-lo, min_size)
    center = ((lo+hi)/2) @ rotation.T
    return center, size, yaw


def compile_assets(points, image, masks, extrinsic, intrinsic, *, scale=1.,
                   scale_source='uncalibrated model units', plane_threshold=.02):
    """Masks are already in processed-image pixels and confidence-filtered.

    No database writes. Model geometry always remains INFERRED. Scale is explicit.
    """
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(plane_threshold) or plane_threshold <= 0:
        raise ValueError('Positive finite scale and plane threshold required')
    flip = np.diag([1., -1., -1.])
    initial = np.asarray(points) @ flip.T
    ground_items = [m for m in masks if m['category'] in ('road', 'ground', 'sidewalk')]
    if not ground_items:
        raise ValueError('A segmented ground/road/sidewalk region is required')
    ground = np.logical_or.reduce([m['mask'] for m in ground_items])
    normal, center, ground_metrics = fit_ground(initial[ground], plane_threshold)
    level = rotation_between(normal, [0, 1, 0])
    matrix = np.eye(4)
    matrix[:3, :3] = scale * level @ flip
    matrix[1, 3] = -scale * (level @ center)[1]
    world_points = transform_points(points, matrix)
    entities, rejected = [], []
    people = [m for m in masks if m['category'] in ('person', 'pedestrian')]
    player_id = max(people, key=lambda m: int(m['mask'].sum()))['id'] if people else None
    for item in masks:
        mask, category = item['mask'], item['category']
        cloud = world_points[mask]
        if len(cloud) < 30:
            rejected.append({'id': item['id'], 'reason': 'fewer than 30 confident interior pixels'})
            continue
        if category in ('sky', 'unknown'):
            rejected.append({'id': item['id'], 'reason': 'not a supported solid asset'})
            continue
        position, size, yaw = upright_box(cloud)
        color = np.median(image[mask], axis=0).tolist()
        kind = 'actor' if category in ('person', 'pedestrian') else 'box'
        if kind == 'actor':
            position[1] -= size[1]/2
            # Horizontal PCA measures visible shape, not which way a person faces.
            # Keep a neutral heading and a coarse symmetric footprint pending review.
            lo, hi = np.quantile(cloud, [.02, .98], axis=0)
            position = np.median(cloud, axis=0)
            position[1] = lo[1]
            body_height = max(.02, hi[1]-lo[1])
            # Use the existing coarse human's proportions. A moving silhouette's
            # depth spread is not a reliable estimate of torso thickness.
            size = np.maximum(body_height*np.array([.28, 1., .22]), .02)
            yaw = 0.
        if category in ('road', 'ground', 'sidewalk'):
            # Ground is a visible planar patch; its hidden thickness is not measured.
            position[1] = float(np.median(cloud[:, 1])) - .025
            size[1] = .05
        entity = {'id': item['id'], 'kind': kind, 'category': category,
                  'position': position.tolist(), 'size': size.tolist(),
                  'rotation_y': float(yaw), 'color': color, 'provenance': 'INFERRED',
                  'reason': 'SAM mask lifted through predicted depth; visible extent only',
                  'geometry_evidence': {'pixel_count': len(cloud),
                                        'visible_surface_only': True,
                                        'occluded_extent_known': False,
                                        'scale_source': scale_source},
                  'behavior': {}, 'solid': category not in ('road', 'ground', 'sidewalk')}
        if kind == 'actor':
            entity['asset_id'] = 'gta_human'
            entity['behavior'] = ({'type': 'player', 'speed': 0} if item['id'] == player_id else
                                  {'type': 'patrol', 'speed': 0,
                                   'waypoints': [position.tolist(), position.tolist()]})
            entity['geometry_evidence']['orientation_known'] = False
            entity['geometry_evidence']['body_shape'] = 'coarse human proportion prior, not reconstructed anatomy'
            entity['geometry_evidence']['behavior'] = 'unplanned stationary placeholder'
        elif category in ('car', 'truck', 'vehicle'):
            entity['asset_id'] = 'gta_vehicle'
        entities.append(entity)
    rotation, translation = extrinsic[:, :3], extrinsic[:, 3]
    camera_model = -rotation.T @ translation
    camera_position = transform_points(camera_model, matrix)
    forward = matrix[:3, :3] @ rotation.T[:, 2]
    up = matrix[:3, :3] @ -rotation.T[:, 1]
    height, width = points.shape[:2]
    k = np.asarray(intrinsic)
    if abs(k[0, 2]-width/2) > 1e-3 or abs(k[1, 2]-height/2) > 1e-3:
        raise ValueError('Off-center principal point unsupported by current whitebox renderer')
    camera = {'position': camera_position.tolist(),
              'target': (camera_position+forward/np.linalg.norm(forward)).tolist(),
              'up': (up/np.linalg.norm(up)).tolist(),
              'fov': math.degrees(2*math.atan(height/(2*k[1, 1]))),
              'intrinsics': {'width': width, 'height': height, 'fx': float(k[0, 0]),
                             'fy': float(k[1, 1]), 'cx': width/2, 'cy': height/2}}
    scene = {'schema_version': 1, 'seed': 17,
             'lore': 'Geometry initialization proposal; scale, hidden geometry and actor heading are inferred.',
             'terrain': {'size': [220, 220], 'resolution': 32, 'seed': 17,
                         'height_amplitude': 0, 'flatten_radius': 30},
             'camera': camera, 'entities': entities}
    return scene, {'model_to_scene': matrix.tolist(), 'ground': ground_metrics,
                   'scale': scale, 'scale_source': scale_source, 'player_candidate': player_id,
                   'rejected': rejected,
                   'limitations': ['Boxes bound visible surfaces, not complete hidden objects',
                                   'Ground orientation inferred from dominant plane; hidden support remains unknown',
                                   'Actor heading and executable future behaviors require agent planning',
                                   'Proposal only; no authoritative world or history modified']}
