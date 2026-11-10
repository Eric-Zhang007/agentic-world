"""Fit explicit candidate transform parameters to source-image annotations."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from whitebox.image_alignment import landmark_points, project_points
from whitebox.world import validate_scene
from whitebox.motion import validate_geometry
from whitebox.camera_fit import camera_delta
from whitebox.camera_reconcile import effective_fov
from whitebox.surface_attachment import apply_surface_attachments


def fit_scene(scene, data):
    validate_scene(scene)
    params = data['parameters']
    views = data.get('views') if 'views' in data else [dict(data, camera=scene['camera'])]
    if not isinstance(views, list) or not views:
        raise ValueError('At least one view is required')
    targets = []
    for view in views:
        observed = np.asarray([r['pixel'] for r in view['landmarks']], float)
        if not len(observed) or observed.shape != (len(view['landmarks']), 2) or not np.isfinite(observed).all():
            raise ValueError('Each view requires finite 2D landmark pixels')
        targets.append(observed)
    observed = np.concatenate(targets)
    by_id = {e['id']: e for e in scene['entities']}
    if any(p['field'] not in ('position', 'size') or p['axis'] not in (0, 1, 2) for p in params):
        raise ValueError('Only explicit position/size axes can be fitted')
    if len({(p['id'], p['field'], p['axis']) for p in params}) != len(params):
        raise ValueError('Duplicate optimization parameter')
    writes = set()
    for p in params:
        linked = p.get('linked_ids', [])
        if not isinstance(linked, list) or (linked and p['field'] != 'position'):
            raise ValueError('Linked parts require a list of IDs and shared translation')
        for identity in [p['id'], *linked]:
            if identity not in by_id:
                raise ValueError('Unknown linked or parameter entity: '+str(identity))
            target = (identity, p['field'], p['axis'])
            if target in writes:
                raise ValueError('Optimization parameters overlap on '+str(target))
            writes.add(target)
    relations = data.get('attachments', [])
    apply_surface_attachments(scene, relations)
    supports = data.get('support_heights', {})
    if not isinstance(supports, dict):
        raise ValueError('Support heights must map explicit box IDs to plane heights')
    attached_ids = {relation['child'] for relation in relations}
    for identity, height in supports.items():
        if identity not in by_id or by_id[identity]['kind'] != 'box':
            raise ValueError('Support height requires a known static box')
        if isinstance(height, bool) or not isinstance(height, (int,float)) or not np.isfinite(height):
            raise ValueError('Support height must be finite')
        if identity in attached_ids or (identity, 'position', 1) in writes:
            raise ValueError('Supported box height is derived; cannot also attach or optimize center Y')
    for relation in relations:
        parent = by_id[relation['parent']]
        c, s = np.cos(parent.get('rotation_y', 0)), np.sin(parent.get('rotation_y', 0))
        normal = np.asarray([[c,0,s],[0,1,0],[-s,0,c]])[:,relation['axis']]
        for identity, field, axis in writes:
            if identity == relation['child'] and field == 'position' and abs(normal[axis]) > 1e-8:
                raise ValueError('Attached child normal position is derived; fit tangential axes or parent instead')
    start = np.asarray([by_id[p['id']][p['field']][p['axis']] for p in params])
    lower, upper = np.asarray([p['bounds'] for p in params], float).T
    camera_specs = []
    for view in views:
        bounds = np.asarray(view.get('camera_bounds', [[0,0]]*6), float)
        if bounds.shape != (6,2) or not np.isfinite(bounds).all() or np.any(bounds[:,0]>bounds[:,1]):
            raise ValueError('Camera bounds require six finite ordered pairs')
        if not 5 <= effective_fov(view['camera'])+bounds[5,0] <= effective_fov(view['camera'])+bounds[5,1] <= 150:
            raise ValueError('Camera FOV bounds exceed valid range')
        active = bounds[:,0] < bounds[:,1]
        camera_specs.append((bounds, active, slice(len(start), len(start)+int(active.sum()))))
        start = np.r_[start, np.clip(np.zeros(active.sum()), bounds[active,0], bounds[active,1])]
        lower = np.r_[lower, bounds[active,0]]
        upper = np.r_[upper, bounds[active,1]]
    if not any(not active.any() for _,active,_ in camera_specs):
        raise ValueError('Joint geometry fitting requires at least one fixed camera view')
    def cameras(x):
        result = []
        for view, (bounds,active,section) in zip(views,camera_specs):
            delta = bounds[:,0].copy()
            delta[active] = x[section]
            result.append(camera_delta(view['camera'], delta) if np.any(delta) else deepcopy(view['camera']))
        return result
    def candidate(x):
        value = deepcopy(scene)
        entities = {e['id']: e for e in value['entities']}
        for p, v in zip(params, x):
            entities[p['id']][p['field']][p['axis']] = float(v)
            if p.get('linked_ids'):
                if p['field'] != 'position':
                    raise ValueError('Linked parts support shared translation only')
                delta = v - by_id[p['id']]['position'][p['axis']]
                for identity in p['linked_ids']:
                    entities[identity]['position'][p['axis']] = float(by_id[identity]['position'][p['axis']] + delta)
        for identity, height in supports.items():
            entities[identity]['position'][1] = height + entities[identity]['size'][1]/2
        return apply_surface_attachments(value, data.get('attachments', []))
    def pixels(value, view_cameras):
        projections = [project_points(landmark_points(value, v['landmarks']),
                       c, v['width'], v['height']) for v,c in zip(views,view_cameras)]
        return np.concatenate([p[0] for p in projections]), np.concatenate([p[1] for p in projections])
    initial, valid = pixels(scene, [v['camera'] for v in views])
    if not valid.all():
        raise ValueError('Initial landmarks cross camera near plane')
    def residual(x):
        uv, visible = pixels(candidate(x), cameras(x))
        return np.where(visible[:, None], np.nan_to_num(uv)-observed, 1e5).ravel()
    started = time.perf_counter()
    optimized = least_squares(residual, start, bounds=(lower, upper), max_nfev=300)
    after = candidate(optimized.x)
    validate_scene(after)
    fitted_cameras = cameras(optimized.x)
    final, valid = pixels(after, fitted_cameras)
    if not valid.all():
        raise ValueError('Optimized landmarks cross camera near plane')
    errors = validate_geometry(after)
    view_reports = []
    offset = 0
    for view, expected, camera in zip(views, targets, fitted_cameras):
        count = len(expected)
        section = slice(offset, offset+count)
        item = dict(before_errors_px=np.linalg.norm(initial[section]-expected, axis=1).tolist(),
                    after_errors_px=np.linalg.norm(final[section]-expected, axis=1).tolist())
        if view.get('heldout'):
            rows = view['heldout']
            heldout = np.asarray([r['pixel'] for r in rows], float)
            if heldout.shape != (len(rows), 2) or not np.isfinite(heldout).all():
                raise ValueError('Held-out pixels must be finite 2D points')
            uv, valid = project_points(landmark_points(after, rows), camera, view['width'], view['height'])
            item['heldout_errors_px'] = [float(np.linalg.norm(p-q)) if ok else None
                                        for p,q,ok in zip(uv,heldout,valid)]
        view_reports.append(item)
        offset += count
    return after, dict(seconds=time.perf_counter()-started, evaluations=optimized.nfev,
        optimizer_success=bool(optimized.success), geometry_errors=errors,
        world_committed=False, annotations=data,
        before_pixels=initial.tolist(), after_pixels=final.tolist(),
        before_errors_px=np.linalg.norm(initial-observed, axis=1).tolist(),
        after_errors_px=np.linalg.norm(final-observed, axis=1).tolist(),
        fitted_parameters=optimized.x.tolist(),
        active_bounds=optimized.active_mask.tolist(),
        jacobian_singular_values=np.linalg.svd(optimized.jac, compute_uv=False).tolist(),
        views=view_reports,
        view_cameras=fitted_cameras,
        limitation='Shared geometry with explicitly bounded cameras and a fixed-view anchor; scale and geometry still depend on priors.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('scene', 'annotations', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--image', type=Path)
    a = p.parse_args()
    scene = json.loads(a.scene.read_text())
    data = json.loads(a.annotations.read_text())
    if 'views' in data:
        after, report = fit_scene(scene, data)
        a.output.mkdir(parents=True, exist_ok=False)
        for name, value in [('scene_before', scene), ('scene', after), ('report', report)]:
            (a.output/(name+'.json')).write_text(json.dumps(value, indent=2))
        for index, view in enumerate(data['views']):
            candidate = deepcopy(after)
            candidate['camera'] = deepcopy(report['view_cameras'][index])
            (a.output/f'view_{index:02d}.json').write_text(json.dumps(candidate, indent=2))
        print(json.dumps({k:v for k,v in report.items() if k != 'annotations'}))
        return
    if a.image is None:
        p.error('--image is required for a single-view fit')
    with Image.open(a.image) as source:
        image = source.convert('RGB')
    if image.size != (data['width'], data['height']):
        raise ValueError('Annotation and source dimensions differ')
    after, report = fit_scene(scene, data)
    a.output.mkdir(parents=True, exist_ok=False)
    for name, value in [('scene_before', scene), ('scene', after), ('report', report)]:
        (a.output/(name+'.json')).write_text(json.dumps(value, indent=2))
    draw = ImageDraw.Draw(image)
    for row, before, final in zip(data['landmarks'], report['before_pixels'], report['after_pixels']):
        x, y = row['pixel']
        draw.ellipse((x-4,y-4,x+4,y+4),outline='white',width=2)
        for point, color in [(before, 'red'), (final, 'lime')]:
            u, v = point
            draw.line((x,y,u,v),fill=color,width=2)
            draw.rectangle((u-3,v-3,u+3,v+3),outline=color,width=2)
        draw.text((x+6,y+4),row['id'],fill='white',stroke_width=1,stroke_fill='black')
    image.save(a.output/'overlay.jpg')
    print(json.dumps({k:v for k,v in report.items() if k not in ('annotations','before_pixels','after_pixels')}))


if __name__ == '__main__':
    main()
