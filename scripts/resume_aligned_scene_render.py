"""Resume the saved new-scene trial after prompt compilation, without replanning."""
import json
from pathlib import Path
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent.render_contract import compile_prompt
from renderer.comfy_h3 import ComfyH3Renderer

out=ROOT/'outputs/aligned_rollout_20261014'
read=lambda name:json.loads((out/name).read_text())
reference=Path(read('experiment.json')['reference'])
cfg=yaml.safe_load((ROOT/'configs/default.yaml').read_text())
prompt=compile_prompt(cfg['agent'],read('scene.json'),read('action.json'),read('timeline.json'),
                      reference,out,reference_video=out/'whitebox.mp4')
print('Prompt compiled; rendering',flush=True)
video=ComfyH3Renderer(gpu='0',steps=40,frames=124,width=1024,height=768,
    timeout=5400,kernel_backend='triton').render(prompt,None,out/'render',
    first_frame=reference,anchor_first=True,reference_video=out/'whitebox.mp4',
    control_video=out/'depth.npz',strength=.5,seed=102,cache_threshold=0)
(out/'status.json').write_text(json.dumps({'status':'GENERATED_NOT_ACCEPTED','video':str(video),'world_commit':False}))
print(video,flush=True)
