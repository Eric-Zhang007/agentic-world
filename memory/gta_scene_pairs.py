"""Scene revisit mining. No object identity or paid annotation is required."""
from collections import defaultdict
import math

import cv2
import numpy as np
from scipy.spatial import cKDTree


def angle_degrees(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b)), -1, 1))))


def mine_pairs(points, radius=5., min_age=30., max_angle=35., excursion=15., spacing=30.):
    """Require actual leave/return for one camera, or two distinct trajectories.

    Geometry is a candidate criterion, not proof of matching background pixels.
    Never bridge episodes/clocks. Unknown sampling intervals cannot prove travel.
    """
    groups = defaultdict(list)
    for p in points:
        groups[(p['episode_id'], p['server_clock_domain'])].append(p)
    result = []
    for group in groups.values():
        group.sort(key=lambda p: (p['time_ns'], p['point_id']))
        xyz = np.array([p['camera_position'] for p in group])
        tree = cKDTree(xyz)
        tracks = defaultdict(list)
        for i, p in enumerate(group):
            tracks[(p['agent_id'], p['client_clock_domain'])].append(i)
        last = {}
        for j, target in enumerate(group):
            options = []
            for i in tree.query_ball_point(xyz[j], radius):
                ref = group[i]
                age = (target['time_ns']-ref['time_ns'])/1e9
                if age < min_age:
                    continue
                angle = angle_degrees(ref['forward'], target['forward'])
                if angle > max_angle or abs(ref['camera_position'][2]-target['camera_position'][2]) > 2.:
                    continue
                same = ref['agent_id'] == target['agent_id']
                if same and ref['client_clock_domain'] != target['client_clock_domain']:
                    continue
                travel = None
                if same:
                    between = [k for k in tracks[(ref['agent_id'], ref['client_clock_domain'])] if i <= k <= j]
                    times = [group[k]['time_ns'] for k in between]
                    if not between or max((b-a)/1e9 for a, b in zip(times, times[1:])) > 7.5:
                        continue
                    travel = float(np.linalg.norm(xyz[between]-xyz[i], axis=1).max())
                    if travel < excursion:
                        continue
                distance = float(np.linalg.norm(xyz[j]-xyz[i]))
                kind = 'same_camera_leave_and_return' if same else 'cross_camera_trajectory_intersection'
                options.append((distance+angle/10., i, age, angle, travel, kind))
            if not options:
                continue
            options.sort()
            for _, i, age, angle, travel, kind in options:
                ref = group[i]
                # Avoid exporting every nearby frame of the same crossing.
                key = (ref['agent_id'], target['agent_id'], kind,
                       tuple(math.floor(v/(radius*2)) for v in xyz[j][:2]))
                if target['time_ns']-last.get(key, -10**20) < spacing*1e9:
                    continue
                last[key] = target['time_ns']
                result.append({'reference_point': ref['point_id'], 'target_point': target['point_id'],
                               'episode_id': target['episode_id'], 'kind': kind,
                               'age_sec': age, 'camera_distance': float(np.linalg.norm(xyz[j]-xyz[i])),
                               'view_angle_deg': angle, 'max_excursion': travel})
                break
    return result


def background_evidence(ref, target, ref_mask, target_mask):
    """Classical features on background only; output is evidence, not a label."""
    sift = cv2.SIFT_create(nfeatures=2500)
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), ref_mask)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), target_mask)
    base = {'reference_features': len(k1), 'target_features': len(k2),
            'good_matches': 0, 'inliers': 0, 'inlier_ratio': 0., 'occupied_grid_cells': 0,
            'reference_background_fraction': float((ref_mask>0).mean()),
            'target_background_fraction': float((target_mask>0).mean())}
    if d1 is None or d2 is None or len(d2) < 2:
        return base
    matches = [m[0] for m in cv2.BFMatcher().knnMatch(d1, d2, k=2)
               if len(m)==2 and m[0].distance < .7*m[1].distance]
    # Enforce unique target matches before estimating a geometric transform.
    unique = {}
    for m in sorted(matches, key=lambda m: m.distance):
        unique.setdefault(m.trainIdx, m)
    matches = list(unique.values())
    base['good_matches'] = len(matches)
    if len(matches) < 8:
        return base
    a = np.float32([k1[m.queryIdx].pt for m in matches])
    b = np.float32([k2[m.trainIdx].pt for m in matches])
    _, mask = cv2.findHomography(a, b, cv2.RANSAC, 3.)
    if mask is None:
        return base
    keep = mask.ravel().astype(bool)
    cells = {(min(3,int(x/ref.shape[1]*4)), min(2,int(y/ref.shape[0]*3))) for x,y in a[keep]}
    base.update(inliers=int(keep.sum()), inlier_ratio=float(keep.mean()), occupied_grid_cells=len(cells))
    return base
