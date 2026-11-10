"""Generate an isolated native-video/depth rollout from the new initial scene."""
import json
from pathlib import Path
import sys
import subprocess

import numpy as np
from PIL import Image
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from whitebox.director import plan
from whitebox.motion import simulate
from agent.render_contract import compile_prompt
from renderer.comfy_h3 import ComfyH3Renderer
from scripts.run_comfy_control import save_video


def save(path,data):path.write_text(json.dumps(data,indent=2),encoding='utf-8')


def main():
    output=ROOT/'outputs/aligned_rollout_20261014'
    output.mkdir(exist_ok=False)
    source=ROOT/'outputs/aligned_world_20261014_repaired/world_current.json'
    scene=json.loads(source.read_text())
    reference=ROOT/'outputs/aligned_seed_20261014_b/frame_011.png'
    cfg=yaml.safe_load((ROOT/'configs/default.yaml').read_text())
    goal=('A continuous camera-only 90-degree orbit around the current look-at target over exactly '
          '5.125 seconds. Keep radius, height and lens fixed. Keep both actors at their current positions '
          'and world-facing directions with subtle idle behavior; retain the staff and visual identities. '
          'Use one wait phase lasting 5.125s with camera_orbit_yaw_deg=90. No attack, cut or return. '
          'Actual H3 inputs will be Picture 1 (this new appearance image), Video 1 (the corresponding '
          'whitebox RGB motion/layout video), and depth strength 0.5. Existing proxy grip differences '
          'remain known uncertainty; no extra appearance memory images will be supplied.')
    print('Planning camera action',flush=True)
    action=plan(cfg['agent'],scene,goal,images=[reference])
    save(output/'action.json',action)
    timeline,end,report=simulate(scene,action)
    save(output/'scene.json',scene);save(output/'timeline.json',timeline)
    save(output/'candidate_world.json',end);save(output/'numerical_report.json',report)
    if len(timeline['frames'])!=123 or not report['valid']:
        raise ValueError('Expected valid 123-tick plan: '+str(report))
    save(output/'experiment.json',{'source_world':str(source),'reference':str(reference),
         'world_commit':False,'depth_strength':.5,'seed':102,
         'known_limitations':['initial staff/grip alignment approximate','coarse static geometry']})
    for mode in ['preview','depth']:
        folder=output/mode
        with (output/(mode+'.log')).open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'whitebox/godot/render.py'),
                '--scene',str(output/'scene.json'),'--timeline',str(output/'timeline.json'),
                '--output',str(folder),'--render-pass',mode,'--frames','all',
                '--width','1024','--height','768'],stdout=log,stderr=subprocess.STDOUT,check=True)
        files=[folder/'camera.png',*sorted(folder.glob('frame_*.png'))]
        pixels=np.stack([np.asarray(Image.open(f).convert('RGB')) for f in files])
        assert pixels.shape==(124,768,1024,3)
        if mode=='preview':save_video(pixels.astype(np.float32)/255,output/'whitebox.mp4')
        else:np.savez_compressed(output/'depth.npz',rgb=pixels)
    print('Compiling prompt using existing render-rollout skill',flush=True)
    prompt=compile_prompt(cfg['agent'],scene,action,timeline,reference,output,
                          reference_video=output/'whitebox.mp4')
    print('Rendering native video plus depth 0.5',flush=True)
    result=ComfyH3Renderer(gpu='0',steps=40,frames=124,width=1024,height=768,
        timeout=5400,kernel_backend='triton').render(prompt,None,output/'render',
        first_frame=reference,anchor_first=True,reference_video=output/'whitebox.mp4',
        control_video=output/'depth.npz',strength=.5,seed=102,cache_threshold=0)
    save(output/'status.json',{'status':'GENERATED_NOT_ACCEPTED','video':str(result),'world_commit':False})
    print(result,flush=True)


if __name__=='__main__':main()
