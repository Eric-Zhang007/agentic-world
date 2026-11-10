"""Prepare two isolated, dataset-calibrated demonstration branches."""
import json
import math
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from harness.pi.world_tools import read,save
from whitebox.surfaces import floor_height
from whitebox.motion import validate_geometry


def main():
    out=ROOT/'outputs/meeting_demo';out.mkdir(exist_ok=False)
    source=read(ROOT/'outputs/gta_pi_loop_20261102/urban/source.json')
    seed=read(ROOT/'outputs/gta_asset_alignment/surface_motion_scene.json')
    seed.pop('projection_check_points',None)
    camera=seed['camera'];forward=np.subtract(camera['target'],camera['position']);forward/=np.linalg.norm(forward)
    right=np.cross(forward,camera['up']);right/=np.linalg.norm(right)
    player=seed['entities'][0]
    player['rotation_y']=math.atan2(forward[0],forward[2])
    player['description']='Short blonde-haired woman, dark brown tailored jacket and knee-length skirt, bare lower legs and dark shoes.'
    player['provenance']='INFERRED';player['reason']='Dataset-calibrated initial placement, coarse adult body on recorded street collision surface.'
    k=camera['intrinsics']
    ray=forward+(618-k['cx'])/k['fx']*right-(443-k['cy'])/k['fy']*np.array(camera['up'])
    position=None
    for t in np.linspace(1,30,2901):
        p=np.array(camera['position'])+ray*t
        floor=floor_height(seed,p[0],p[2],player['position'][1])
        if floor is not None and p[1]<=floor:
            p[1]=floor;position=p.tolist();break
    if position is None:raise ValueError('Cannot ground visible NPC')
    npc=deepcopy(player);npc.update(id='pedestrian',position=position,rotation_y=math.atan2(right[0],right[2]),
        description='Dark-haired male pedestrian ahead and to the right, bare upper body, dark trousers and bright shoes.',
        reason='Visible pedestrian foot point projected onto recorded street surface; depth remains an estimate.',
        behavior={'type':'patrol','speed':.35,'waypoints':[(np.array(position)+right*1.5).tolist(),position]},
        color=[.7,.35,.2])
    seed['entities'].append(npc)
    errors=validate_geometry(seed)
    if errors:raise ValueError(errors)
    save(out/'initial_scene_local.json',seed)
    remote=deepcopy(seed)
    remote['static_geometry']={'path':'/home/star/agentic_world/outputs/gta_asset_alignment/urban_region/surfaces.bin',
        'collision_path':'/home/star/agentic_world/outputs/gta_asset_alignment/urban_region/collision.bin'}
    save(out/'initial_scene.json',remote)
    Image.open(source['rgb']).convert('RGB').save(out/'input.png')
    save(out/'source.json',{'dataset_sample':Path(source['sample']).parent.name,
        'initialization':'Imported calibrated static geometry plus coarse grounded player and visible pedestrian; not RGB-only reconstruction'})
    for index in range(2):
        inputs=[]
        for tick in range(72):
            inputs.append({'keys':['w'], 'mouse':[0. if index==0 else .5,0.]})
        save(out/f'keyboard_input_{index+1}.json',{'type':'controls','actor_id':'player',
            'goal':'Follow recorded keyboard movement through the street', 'inputs':inputs,
            'input_provenance':'Scripted WASD/mouse demonstration, executed by the deterministic controller'})
    print(out)


if __name__=='__main__':main()
