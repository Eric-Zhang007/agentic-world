"""Exercise recorded urban collision geometry, without H3 or world commits."""
import json
from pathlib import Path
import numpy as np
from whitebox.surfaces import SurfaceWorld
from whitebox.motion import prepare_scene,simulate
from scripts.test_whitebox_world import fixture


def main():
    root=Path('outputs/gta_asset_alignment').resolve()
    collision=root/'urban_region/collision.bin'
    surface=SurfaceWorld(np.fromfile(collision,dtype='<f4').reshape(-1,3,3))
    scene=json.loads((root/'calibrated_scene.json').read_text())
    scene['static_geometry']['path']=str(root/'urban_region/surfaces.bin')
    scene['static_geometry']['collision_path']=str(collision)
    scene['terrain']['size']=[200,200]
    actor=next(e for e in fixture()['entities'] if e['id']=='player')
    actor['size']=[.65,1.8,.4]
    actor['asset_id']='gta_human'
    x,z=2.21765136719,-3.59020996094
    y=surface.floor(x,z,-1.7628,max_drop=2)
    if y is None:raise RuntimeError('Recorded player has no nearby support')
    actor['position']=[x,y,z];actor['rotation_y']=0
    scene['entities']=[actor]
    report=[]
    for i,delta in enumerate(([3,0,0],[-3,0,0],[0,0,3],[0,0,-3],[2,0,2],[-2,0,-2],[8,0,0],[-8,0,0],[0,0,8],[0,0,-8])):
        count=2 if np.linalg.norm(delta)>4 else 1
        action={'actor_id':'player','phases':[{'type':'move','delta':(np.asarray(delta)/count).tolist(),'duration_sec':3,
            'end_velocity':(np.asarray(delta)/(3*count)).tolist(),'criterion':'Test recorded street support and obstacles'} for _ in range(count)]}
        start=prepare_scene(scene)
        timeline,end,checks=simulate(start,action)
        report.append({'delta':delta,'start':actor['position'],'end':end['entities'][0]['position'],
            'checks':checks,'root_path':[f['entities'][0]['position'] for f in timeline['frames']]})
        (root/f'surface_motion_{i}.json').write_text(json.dumps(timeline,indent=2))
    (root/'surface_motion_scene.json').write_text(json.dumps(scene,indent=2))
    (root/'surface_motion_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps([{'delta':r['delta'],'end':r['end'],'valid':r['checks']['valid'],
        'violation_count':r['checks']['violation_count']} for r in report]))
    # Projection grid in the recovered dataset basis, with expected pixel coordinates.
    k=scene['camera']['intrinsics'];c=scene['camera']
    f=np.subtract(c['target'],c['position']);f/=np.linalg.norm(f)
    up=np.asarray(c['up']);right=np.cross(f,up);right/=np.linalg.norm(right)
    points=[];pixels=[]
    for depth in (4,12,40):
        for u,v in ((.25,.25),(.5,.5),(.75,.75),(.25,.75),(.75,.25)):
            points.append((np.asarray(c['position'])+depth*(f+(u*k['width']-k['cx'])/k['fx']*right-(v*k['height']-k['cy'])/k['fy']*up)).tolist())
            pixels.append([u*960,v*540])
    scene['projection_check_points']=points
    scene['entities']=[]
    (root/'calibrated_review_scene.json').write_text(json.dumps(scene,indent=2))
    (root/'projection_expected.json').write_text(json.dumps(pixels))


if __name__=='__main__':main()
