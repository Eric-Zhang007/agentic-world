"""Evidence-labeled coarse camera correction, never a metric reconstruction."""
from copy import deepcopy

import numpy as np


def effective_fov(camera):
    if 'intrinsics' in camera:
        k = camera['intrinsics']
        return float(np.degrees(2 * np.arctan(float(k['height']) / (2 * float(k['fy'])))))
    return camera.get('fov', 50.)


def update_camera(camera, patch):
    """Copy a camera; FOV edits preserve calibration aspect and principal point."""
    updated = deepcopy(camera)
    updated.update(deepcopy(patch))
    if 'fov' in patch and 'intrinsics' in camera:
        fov = patch['fov']
        if isinstance(fov, bool) or not np.isfinite(fov) or not 5 <= fov <= 150:
            raise ValueError('Invalid vertical field of view')
        k = updated['intrinsics']
        fy = float(k['height']) / (2 * np.tan(np.radians(fov) / 2))
        k['fx'] = float(k['fx']) * fy / float(k['fy'])
        k['fy'] = float(fy)
    validate_camera(updated)
    return updated


def validate_camera(camera):
    eye = np.asarray(camera['position'], dtype=float)
    target = np.asarray(camera['target'], dtype=float)
    fov = camera.get('fov', 50.)
    if (eye.shape != (3,) or target.shape != (3,) or
            not np.isfinite(eye).all() or not np.isfinite(target).all() or
            isinstance(fov, bool) or not np.isfinite(fov) or not 5 <= fov <= 150):
        raise ValueError('Invalid camera position, target or vertical field of view')
    direction = target-eye
    if np.linalg.norm(direction) < 1e-6:
        raise ValueError('Camera look direction is degenerate')
    direction = direction / np.linalg.norm(direction)
    up = np.asarray(camera.get('up', [0, 0, -1] if abs(direction[1]) > .99 else [0, 1, 0]), dtype=float)
    if (up.shape != (3,) or not np.isfinite(up).all() or
            np.linalg.norm(np.cross(direction, up)) < 1e-6):
        raise ValueError('Camera look direction is degenerate with camera up')


def correct_cameras(before, frames, fps, correction):
    """Interpolate offsets from the plan, anchored at the accepted start camera."""
    if correction is None:
        return None
    if (set(correction)-{'evidence', 'uncertainty', 'keyframes'} or
            not correction.get('evidence') or not correction.get('uncertainty') or
            not correction.get('keyframes')):
        raise ValueError('Camera correction needs background evidence, uncertainty and keyframes')
    if not frames or isinstance(fps, bool) or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Camera correction requires frames and positive finite fps')
    times = np.asarray([frame['time'] for frame in frames], dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Camera frame times must be finite and strictly increasing')
    start = times[0]-1/fps
    validate_camera(before['camera'])
    keys = [(start, np.zeros(7))]
    previous = -1
    for key in correction['keyframes']:
        if set(key)-{'frame_index', 'position_offset', 'target_offset', 'fov_offset'}:
            raise ValueError('Unknown camera correction field')
        index = key['frame_index']
        if type(index) is not int or not previous < index < len(frames):
            raise ValueError('Camera indices must be chronological planned-frame indices')
        eye = np.asarray(key['position_offset'], dtype=float)
        target = np.asarray(key['target_offset'], dtype=float)
        fov = key.get('fov_offset', 0.)
        if (eye.shape != (3,) or target.shape != (3,) or isinstance(fov, bool) or
                not np.isfinite(eye).all() or not np.isfinite(target).all() or not np.isfinite(fov)):
            raise ValueError('Nonfinite camera offset')
        keys.append((times[index], np.r_[eye, target, fov]))
        previous = index
    if previous != len(frames)-1:
        raise ValueError('Camera correction requires explicit endpoint')
    corrected = deepcopy(frames)
    for frame in corrected:
        offset = np.array([np.interp(frame['time'], [k[0] for k in keys],
                                    [k[1][axis] for k in keys]) for axis in range(7)])
        camera = frame['camera']
        patch = {'position': (np.asarray(camera['position'])+offset[:3]).tolist(),
                 'target': (np.asarray(camera['target'])+offset[3:6]).tolist()}
        if offset[6]:
            patch['fov'] = effective_fov(camera) + float(offset[6])
        frame['camera'] = update_camera(camera, patch)
    for frame, updated in zip(frames, corrected):
        frame['camera'] = updated['camera']
    return {'provenance': 'AGENT_ACCEPTED_VIDEO_INTERPRETATION',
            'scope': 'Coarse single-view camera estimate; not calibrated metric extrinsics',
            'start_camera': deepcopy(before['camera']), **deepcopy(correction)}
