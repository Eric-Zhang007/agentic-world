"""Run one isolated case from the frozen epoch-15 orbit recipe."""
import argparse
import json
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--label',choices=['depth_off','depth_half'],required=True)
p.add_argument('--gpu',required=True)
a=p.parse_args()
root=Path('/home/star/agentic_world/outputs/epoch15_orbit_20261103')
out=root/a.label
out.mkdir(exist_ok=False)
experiment=json.loads((root/'experiment.json').read_text())
command=experiment['command']
case=next(r for r in json.loads((root/'cases.json').read_text()) if r['label']==a.label)
case['strength']=0 if a.label=='depth_off' else .5
path=out/'cases.json';path.write_text(json.dumps([case],indent=2))
command[command.index('--cases')+1]=str(path)
command[command.index('--output')+1]=str(out/'render')
(out/'experiment.json').write_text(json.dumps({'command':command,'gpu':a.gpu,'world_commit':False},indent=2))
with (out/'generation.log').open('w') as log:
    done=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,HF_ENDPOINT='https://hf-mirror.com'))
(out/'exit.json').write_text(json.dumps({'returncode':done.returncode}))
raise SystemExit(done.returncode)
