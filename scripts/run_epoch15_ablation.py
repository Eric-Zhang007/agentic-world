"""Frozen orbit baseline with epoch-15 LoRA; only depth changes between cases."""
import json
import os
from pathlib import Path
import subprocess

root=Path('/home/star/agentic_world')
out=root/'outputs/epoch15_orbit_20261103'
out.mkdir(exist_ok=False)
source=root/'outputs/native_whitebox_reference_20261013/depth_half/experiment.json'
command=json.loads(source.read_text())['command']
command[2]=str(root/'scripts/run_epoch15_control.py')
cases=out/'cases.json'
depth=command[command.index('--control-video')+1]
cases.write_text(json.dumps([
    {'label':'depth_off','controls':[],'cache_threshold':0},
    {'label':'depth_half','controls':[{'path':depth,'strength':0.5}],'cache_threshold':0}],indent=2))
command[command.index('--output')+1]=str(out/'render')
command[command.index('--cases')+1]=str(cases)
command.extend(['--lora','/home/star/agentic-world-comfy/ComfyUI/models/loras/epoch-15.safetensors'])
(out/'experiment.json').write_text(json.dumps({'command':command,'source':str(source),'gpu':'0','world_commit':False,'only_case_difference':'depth off versus 0.5; LoRA strength 1 in both'},indent=2))
with (out/'generation.log').open('w') as log:
    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_ENDPOINT='https://hf-mirror.com'))
(out/'exit.json').write_text(json.dumps({'returncode':result.returncode}))
raise SystemExit(result.returncode)
