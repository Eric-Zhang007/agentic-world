"""Uncommitted coarse-condition probe: continuous actor/camera motion, one H3 clip."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'harness/pi'))
from world_tools import dispatch
from whitebox.motion import simulate
from renderer.comfy_h3 import ComfyH3Renderer


def save(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--first-frame',type=Path,required=True)
    p.add_argument('--gpu',default='2')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    scene=json.loads(a.scene.read_text())
    rates={'translation_velocity':[0,0,0],'orbit_velocity_deg':3.,'distance_velocity':0.}
    paths=[]
    for index in range(2):
        folder=a.output/f'segment_{index+1:02d}';plan=folder/'plan';plan.mkdir(parents=True)
        action={'goal':'Keep walking and orbiting through the clip boundary; no endpoint stop',
            'actor_id':'player','phases':[{'type':'move','duration_sec':3,
            'delta':[0,0,-2.4 if index==0 else -2.7],'end_velocity':[0,0,-.9],
            'camera_follow_actor':'player','camera_orbit_yaw_deg':6 if index==0 else 9,
            'camera_end_motion':rates,'criterion':'Ongoing movement, no settling or recovery'}]}
        timeline,end,diagnostics=simulate(scene,action,12.+index*3)
        if not diagnostics['valid']:raise ValueError(diagnostics['violations'])
        for name,value in [('scene',scene),('action',action),('timeline',timeline),('endpoint',end),('diagnostics',diagnostics)]:
            save(plan/(name+'.json'),value)
        paths.append(folder);scene=end
    save(a.output/'probe.json',{'scope':'Engineering input probe, NOT an accepted autonomous world history',
        'source_scene':str(a.scene),'source_frame':str(a.first_frame),
        'second_plan':'Simulation-only continuation from planned first endpoint; not an observed outcome',
        'actor_shape':'Coarse capsule and head; no articulated limbs in RGB or depth',
        'final_camera_motion':scene['camera']['motion']})
    compiled=dispatch('compile',{'plan':str(paths[0]/'plan'),'output':str(paths[0]/'compiled'),'width':512,'height':288})
    print(json.dumps(compiled),flush=True)
    prompt=('Continue the GTA street scene from the supplied RGB frame. The blonde woman in the brown outfit '
        'walks forward along the clear road with natural alternating footsteps, weight transfer and arm swing. '
        'The other pedestrian continues walking away. The camera follows the woman and slowly orbits in the '
        'direction shown by the whitebox video. The whitebox actors are coarse occupancy volumes, not literal '
        'bodies or poses: retain the realistic people and their clothing from the RGB frame. Match the coarse '
        'routes and camera trajectory; invent natural gait and secondary motion. Both walking and camera motion '
        'continue smoothly past the final frame. This is the middle of an ongoing action, with no pause, '
        'deceleration, final pose or return to the starting position. Preserve the street layout and identities.')
    (paths[0]/'prompt.txt').write_text(prompt,encoding='utf-8')
    renderer=ComfyH3Renderer(gpu=a.gpu,steps=20,frames=73,width=512,height=288)
    video=renderer.render(prompt,source_video=None,output_dir=paths[0]/'render',
        first_frame=a.first_frame,reference_video=compiled['reference_video'],seed=47,anchor_first=True,
        controls=[{'path':compiled['depth_control'],'strength':.5,'end':1.}],cache_threshold=0)
    evidence=dispatch('inspect_video',{'plan':str(paths[0]/'plan'),'video':str(video),
        'output':str(paths[0]/'evidence'),'interval':.125})
    save(a.output/'result.json',{'video':str(video),'evidence':evidence,'world_committed':False})
    print(json.dumps({'video':str(video),'world_committed':False}),flush=True)


if __name__=='__main__':main()
