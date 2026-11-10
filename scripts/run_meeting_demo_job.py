"""Durable Pi job with a real transcript and explicit completion receipt."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--lane',choices=['action','keyboard'],required=True)
    p.add_argument('--attempt',type=int,default=1)
    p.add_argument('--detach',action='store_true')
    a=p.parse_args();root=ROOT/'outputs/meeting_demo';lane=root/a.lane
    lane.mkdir(exist_ok=True)
    if a.detach:
        with (lane/f'launcher_{a.attempt:02d}.log').open('x') as log:
            child=subprocess.Popen([sys.executable,__file__,'--lane',a.lane,'--attempt',str(a.attempt)],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'pid':child.pid,'lane':a.lane}));return
    prompt=root/('action_prompt.txt' if a.lane=='action' else 'keyboard_prompt.txt')
    env=dict(os.environ)
    env['PATH']=str(Path.home()/'.nvm/versions/node/v22.23.2/bin')+os.pathsep+env.get('PATH','')
    for name in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(name,None)
    start=time.time()
    command=[sys.executable,str(ROOT/'harness/pi/run.py'),'--session',f'meeting-{a.lane}-20261103',
             '--prompt-file',str(prompt),'--events',str(lane/f'events_{a.attempt:02d}.jsonl')]
    with (lane/f'console_{a.attempt:02d}.log').open('x') as log:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    (lane/f'job_exit_{a.attempt:02d}.json').write_text(json.dumps({'returncode':result.returncode,'seconds':time.time()-start},indent=2))
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
