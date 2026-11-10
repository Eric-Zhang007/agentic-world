"""Fit an uncommitted camera to explicit approximate image landmarks."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from whitebox.motion import prepare_scene
from whitebox.image_alignment import project_points
from whitebox.camera_reconcile import effective_fov, update_camera


def project(points,camera,width,height):
    pixels, valid = project_points(points, camera, width, height)
    if not valid.all():
        raise ValueError('Camera-fit landmarks must remain in front of the near plane')
    return pixels


def main():
    p=argparse.ArgumentParser()
    for key in ['scene','landmarks','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    scene=json.loads(a.scene.read_text());data=json.loads(a.landmarks.read_text())
    if 'landmarks' in data:
        from whitebox.camera_fit import fit_camera
        candidate,report=fit_camera(scene,data)
        report.update(source_scene=str(a.scene),landmarks=data)
        (a.output/'scene.json').write_text(json.dumps(candidate,indent=2))
        (a.output/'fit.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report))
        return
    scene=prepare_scene(scene);entities={e['id']:e for e in scene['entities']}
    points=[]
    for row in data['points']:
        e=entities[row['id']]
        points.append(e['position'] if row['joint']=='root' else e['pose'][row['joint']])
    targets=np.array([r['pixel'] for r in data['points']]);camera=scene['camera']
    target_z=camera['target'][2]
    start=np.array([*camera['position'],*camera['target'][:2],effective_fov(camera)],float)
    def unpack(x):
        return update_camera(camera, {'position':x[:3].tolist(),
            'target':[float(x[3]),float(x[4]),target_z],'fov':float(x[5])})
    def residual(x):
        pixel=(project(points,unpack(x),data['width'],data['height'])-targets).ravel()
        return np.r_[pixel,(x-start)*np.array([.2,.2,.2,.2,.2,.01])]
    fit=least_squares(residual,start,bounds=([-10,.5,2.5,-10,0,12],[10,8,18,10,5,90]),max_nfev=2000)
    after=unpack(fit.x);candidate=deepcopy(scene);candidate['camera']=after
    (a.output/'scene.json').write_text(json.dumps(candidate,indent=2))
    report={'source_scene':str(a.scene),'landmarks':data,'camera_before':camera,'camera_after':after,
            'before_errors_px':np.linalg.norm(project(points,camera,data['width'],data['height'])-targets,axis=1).tolist(),
            'after_errors_px':np.linalg.norm(project(points,after,data['width'],data['height'])-targets,axis=1).tolist(),
            'optimizer_success':bool(fit.success),'world_commit':False,
            'scope':'Approximate manual image landmarks, fitted camera only; static/weapon alignment needs separate visual review.'}
    (a.output/'fit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
