"""Run one real Pi/GPT -> H3 -> review/commit cycle from a UI request."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from harness.pi.world_tools import dispatch,read,save


def retry_available(out):
    result=out/'result.json'
    calls=0
    for path in out.glob('events*.jsonl'):
        for line in path.read_text(encoding='utf-8').splitlines():
            try:event=json.loads(line)
            except json.JSONDecodeError:continue
            if (event.get('type')=='tool_execution_start' and event.get('toolName')=='world'
                    and event.get('args',{}).get('operation')=='render'):calls+=1
    return (result.exists() and read(result).get('decision')=='REGENERATE'
            and max(calls,len(list(out.rglob('generation_metadata.json'))))<2)


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True)
    p.add_argument('--resume-rejected',action='store_true');a=p.parse_args()
    request=read(a.request);out=a.request.parent;start=time.time()
    before=dispatch('status',{'db':request['db']})
    if before['revision']!=request['revision']:raise ValueError('Accepted world changed before job start')
    mode=request['mode']
    task=(f"Use this exact recorded action: {request['action']}. Do not change the keys or mouse deltas. "
          "Your responsibility is world extension, rendering and video maintenance, not replacing input movement. "
          if mode=='keyboard' else f"User action prompt: {request['goal']}. Plan a 3-second window toward this goal. ")
    prompt=f"""Operate one real video rollout in {out}, using the registered world tools.
Read skills/render-rollout/SKILL.md. Database: {request['db']}. Starting revision: {request['revision']}.
Current actual RGB: {request['first_frame']}. Do not modify source code or call another model.
{task}
First query status. Consider whether the upcoming view needs a prospective world extension;
use world extend only when useful, preserving observed history and existing objects.
An offscreen addition is AGENT_CREATED, not visually confirmed. Do not add objects gratuitously.
For keyboard use the exact supplied input; for text write action.json with explicit moving
endpoint and camera motion when appropriate. No forced stopping at video boundaries.
Then plan -> compile (1024x576) -> read proxy opening/end -> author render_prompt.txt -> render.
Use existing ComfyH3Renderer options gpu="{request['gpu']}",steps=40,frames=returned_video_frames,
width=1024,height=576,timeout=2700. Render request: source_video=null,
first_frame=current RGB,reference_video=compiled whitebox.mp4,anchor_first=true,
controls=[],appearance_images=[],cache_threshold=0. Pass db in render tool args for paired memory selection.
seed=93+attempt. Use new directories per attempt, at most two H3 calls.
Inspect actual video via inspect_video interval=.25, then read first/middle/end plus necessary
extra frames. Judge appearance, natural movement, layout, actual endpoint and camera.
Do not accept mannequins, identity replacement or fake walking. Useful partial progress can be
accepted only by correcting actual trajectories/terminal motion and leaving action_complete=false.
Assess every actor and endpoint motion; unobserved actors are UNOBSERVED_SIMULATED.
Write review.json, call commit, query status, then write result.json with real paths and outcome.
No accepted history may be rewritten. The current planned endpoint is never proof of video state.
"""
    if not a.resume_rejected:(out/'agent_prompt.txt').write_text(prompt,encoding='utf-8')
    env=dict(os.environ)
    for name in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(name,None)
    node=Path.home()/'.nvm/versions/node/v22.23.2/bin'
    if node.is_dir():env['PATH']=str(node)+os.pathsep+env.get('PATH','')
    def run_agent(prompt_file,events,log_file):
        command=[sys.executable,str(ROOT/'harness/pi/run.py'),'--session',out.name,
                 '--prompt-file',str(prompt_file),'--events',str(events)]
        with log_file.open('x') as log:
            return subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if a.resume_rejected:
        if not retry_available(out):raise ValueError('No rejected rollout with remaining generation budget')
        prior=read(out/'job_exit.json')
        if prior['history_after']!=before['history_length']:raise ValueError('World history changed before retry')
        save(out/'job_exit_before_retry.json',prior)
        result=None
    else:
        result=run_agent(out/'agent_prompt.txt',out/'events.jsonl',out/'agent.log')
    current=dispatch('status',{'db':request['db']})
    if ((result is None or result.returncode==0) and current['history_length']==before['history_length']
            and retry_available(out)):
        previous=read(out/'result.json');save(out/'result_before_retry.json',previous)
        retry_prompt=prompt+f"""\nThis request has a rejected first generation. Read {out/'result_before_retry.json'}
and its review, including retry_prompt. The user input is unchanged and the world has not
advanced. Perform the remaining regeneration now, with a new output directory and seed 95.
Reuse the exact recorded input/accepted starting state; improve the render prompt using the
review. There is exactly ONE H3 call left. Do not rewrite the failed attempt. Review the new
actual frames and either commit useful supported progress or reject again. Write the final
result.json with decision and both attempt paths. Never relax acceptance to force success.
"""
        (out/'retry_prompt.txt').write_text(retry_prompt,encoding='utf-8')
        result=run_agent(out/'retry_prompt.txt',out/'events_retry.jsonl',out/'agent_retry.log')
    if result is None:raise ValueError('Retry was not executable')
    after=dispatch('status',{'db':request['db']})
    save(out/'job_exit.json',{'returncode':result.returncode,'seconds':time.time()-start,
        'revision_before':before['revision'],'revision_after':after['revision'],
        'accepted':after['history_length']>before['history_length'],
        'history_before':before['history_length'],'history_after':after['history_length']})
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
