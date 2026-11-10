"""Camera-only diagnostic against fixed geometry and explicit image evidence."""
from copy import deepcopy
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from whitebox.camera_reconcile import effective_fov, update_camera
from whitebox.image_alignment import landmark_points, project_points


def camera_delta(camera, x):
    """Apply translation, yaw, pitch and vertical-FOV deltas to a copied camera."""
    x = np.asarray(x, float)
    if x.shape != (6,) or not np.isfinite(x).all():
        raise ValueError('Six finite camera deltas required')
    eye = np.asarray(camera['position'], float)
    direction = np.asarray(camera['target'], float)-eye
    up = np.array(camera.get('up', [0,0,-1] if abs(direction[1])/np.linalg.norm(direction) > .99 else [0,1,0]), dtype=float)
    up /= np.linalg.norm(up)
    right = np.cross(direction, up)
    right /= np.linalg.norm(right)
    yaw = Rotation.from_rotvec(up*np.radians(x[3]))
    pitch = Rotation.from_rotvec(yaw.apply(right)*np.radians(x[4]))
    rotated = pitch.apply(yaw.apply(direction))
    return update_camera(camera, dict(position=(eye+x[:3]).tolist(),
        target=(eye+x[:3]+rotated).tolist(), fov=effective_fov(camera)+x[5]))


def fit_camera(scene, data):
    camera = scene['camera']
    points = landmark_points(scene, data['landmarks'])
    pixels = np.asarray([r['pixel'] for r in data['landmarks']],float)
    if len(points) < 4 or pixels.shape != (len(points),2) or not np.isfinite(pixels).all():
        raise ValueError('Camera fitting requires at least four finite correspondences')
    bounds = np.asarray(data['bounds'],float)
    if bounds.shape != (6,2):
        raise ValueError('Bounds must cover dx,dy,dz,yaw_deg,pitch_deg,fov_delta_deg')
    if not np.isfinite(bounds).all() or np.any(bounds[:,0]>bounds[:,1]):
        raise ValueError('Finite ordered parameter bounds required')
    active=bounds[:,0]<bounds[:,1]
    if not active.any():raise ValueError('At least one free camera parameter required')
    base=bounds[:,0].copy()
    def expand(x):
        value=base.copy();value[active]=x
        return value
    def unpack(x):
        return camera_delta(camera, x)
    def projection(c):
        return project_points(points,c,data['width'],data['height'])
    before, valid = projection(camera)
    if not valid.all():
        raise ValueError('Initial correspondences must be in front of near plane')
    if not 5 <= effective_fov(camera)+bounds[5,0] <= effective_fov(camera)+bounds[5,1] <= 150:
        raise ValueError('FOV bounds exceed valid camera range')
    def residual(x):
        uv, valid = projection(unpack(expand(x)))
        return np.where(valid[:,None],np.nan_to_num(uv)-pixels,1e5).ravel()
    started = time.perf_counter()
    fit = least_squares(residual,np.clip(np.zeros(active.sum()),bounds[active,0],bounds[active,1]),bounds=(bounds[active,0],bounds[active,1]),
                        x_scale='jac',max_nfev=500)
    result = deepcopy(scene)
    result['camera'] = unpack(expand(fit.x))
    after, valid = projection(result['camera'])
    if not valid.all():
        raise ValueError('Fitted landmarks cross camera near plane')
    singular = np.linalg.svd(fit.jac,compute_uv=False)
    report = dict(world_commit=False,seconds=time.perf_counter()-started,
        optimizer_success=bool(fit.success),evaluations=fit.nfev,
        parameters=expand(fit.x).tolist(),free_parameter_indices=np.flatnonzero(active).tolist(),active_bounds=fit.active_mask.tolist(),
        jacobian_singular_values=singular.tolist(),
        before_errors_px=np.linalg.norm(before-pixels,axis=1).tolist(),
        after_errors_px=np.linalg.norm(after-pixels,axis=1).tolist(),
        camera_before=camera,camera_after=result['camera'],
        scope='Camera fit to inferred fixed geometry, not metric reconstruction or acceptance.')
    if data.get('heldout'):
        rows = data['heldout']
        expected = np.asarray([r['pixel'] for r in rows],float)
        test_points = landmark_points(scene,rows)
        report['heldout'] = []
        for c in (camera,result['camera']):
            uv, valid = project_points(test_points,c,data['width'],data['height'])
            report['heldout'].append([float(np.linalg.norm(p-q)) if ok else None
                                     for p,q,ok in zip(uv,expected,valid)])
    return result,report
