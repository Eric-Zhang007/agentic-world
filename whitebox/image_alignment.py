"""Source-view landmarks in the same Y-up camera convention as Godot.

These are geometric projections, not visibility/occlusion measurements.
"""
import numpy as np

from whitebox.camera_reconcile import validate_camera
from whitebox.posture import body_size


def camera_coordinates(points, camera):
    validate_camera(camera)
    eye = np.asarray(camera['position'], float)
    forward = np.asarray(camera['target'], float) - eye
    forward /= np.linalg.norm(forward)
    default_up = [0, 0, -1] if abs(forward[1]) > .99 else [0, 1, 0]
    right = np.cross(forward, camera.get('up', default_up))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    delta = np.asarray(points, float).reshape(-1, 3) - eye
    return delta @ np.stack([right, up, forward], axis=1)


def focal_lengths(camera, width, height):
    """Effective focal lengths after Godot's integer viewport adjustment."""
    if width <= 0 or height <= 0:
        raise ValueError('Image dimensions must be positive')
    if 'intrinsics' in camera:
        k = camera['intrinsics']
        if abs(k['cx'] - k['width']/2) > 1e-6 or abs(k['cy'] - k['height']/2) > 1e-6:
            raise ValueError('Godot currently requires centered intrinsics')
        fx = float(k['fx']) * width / k['width']
        fy = float(k['fy']) * height / k['height']
        viewport_width = np.floor(width * fy / fx + .5)
        if min(fx, fy, viewport_width) <= 0:
            raise ValueError('Invalid focal length or viewport')
        fx = fy * width / viewport_width
    else:
        fx = fy = height / (2 * np.tan(np.radians(camera.get('fov', 50))/2))
    return fx, fy


def project_points(points, camera, width, height):
    """Project without clipping offscreen landmarks; flag near-plane failures."""
    coords = camera_coordinates(points, camera)
    fx, fy = focal_lengths(camera, width, height)
    valid = np.isfinite(coords).all(axis=1) & (coords[:, 2] >= camera.get('near', .05))
    pixels = np.full((len(coords), 2), np.nan)
    pixels[valid] = np.column_stack([
        width/2 + fx * coords[valid, 0] / coords[valid, 2],
        height/2 - fy * coords[valid, 1] / coords[valid, 2]])
    return pixels, valid


def pixel_rays(pixels, camera, width, height):
    """World rays parameterized by camera-forward depth, not unit length."""
    eye = np.asarray(camera['position'], float)
    # Transform world basis vectors through the exact forward projection basis.
    basis = camera_coordinates(eye + np.eye(3), camera)
    fx, fy = focal_lengths(camera, width, height)
    uv = np.asarray(pixels, float).reshape(-1, 2)
    rays = np.column_stack(((uv[:, 0]-width/2)/fx,
                            -(uv[:, 1]-height/2)/fy, np.ones(len(uv)))) @ basis.T
    return eye, rays


def pixels_to_plane(pixels, camera, width, height, normal=(0, 1, 0), offset=0):
    """Intersect rays with an explicit n.dot(X) + offset = 0 support prior."""
    eye, rays = pixel_rays(pixels, camera, width, height)
    uv = np.asarray(pixels, float).reshape(-1, 2)
    n = np.asarray(normal, float)
    if n.shape != (3,) or not np.isfinite(n).all() or np.linalg.norm(n) == 0 or not np.isfinite(offset):
        raise ValueError('Plane must have a finite nonzero normal and offset')
    norm = np.linalg.norm(n)
    n = n / norm
    offset = offset / norm
    denominator = rays @ n
    valid = np.isfinite(rays).all(axis=1) & (np.abs(denominator) > 1e-10)
    distance = np.full(len(uv), np.nan)
    np.divide(-(eye @ n + offset), denominator, out=distance, where=valid)
    valid &= distance >= camera.get('near', .05)
    points = np.full((len(uv), 3), np.nan)
    points[valid] = eye + distance[valid, None] * rays[valid]
    return points, valid


def landmark_points(scene, landmarks):
    """local_fraction is relative to entity size and its actual root/center origin."""
    entities = {e['id']: e for e in scene['entities']}
    points = []
    for row in landmarks:
        entity = entities[row['id']]
        size = body_size(entity) if entity.get('kind') == 'actor' else entity['size']
        local = np.asarray(row['local_fraction'], float) * size
        yaw = entity.get('rotation_y', 0)
        c, s = np.cos(yaw), np.sin(yaw)
        rotated = [c*local[0]+s*local[2], local[1], -s*local[0]+c*local[2]]
        points.append(np.asarray(entity['position']) + rotated)
    return np.asarray(points)
