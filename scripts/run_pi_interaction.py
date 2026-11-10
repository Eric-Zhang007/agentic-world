"""Launch, inspect and collect isolated continuations of accepted GTA worlds."""
import argparse
import json
from pathlib import Path
import shlex
import stat
import time

from run_gta_pi_cases import ROOT,LOCAL,REMOTE,OUT,connect,mkdir


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['start','status','fetch'])
    parser.add_argument('--branch',required=True,help='Relative trial directory, e.g. rural/road_edge')
    parser.add_argument('--source',help='Relative directory containing accepted world.sqlite')
    parser.add_argument('--action')
    parser.add_argument('--generations',type=int,default=2)
    parser.add_argument('--gpu',default='2')
    parser.add_argument('--width',type=int,default=512)
    parser.add_argument('--height',type=int,default=288)
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--max-duration',type=int,choices=[3,5],default=3)
    args=parser.parse_args()
    if any(part in ['..',''] for part in args.branch.split('/')) or args.branch.startswith('/'):
        parser.error('Use a relative branch directory')
    folder=OUT+'/'+args.branch;local=LOCAL/args.branch
    client=connect()
    try:
        with client.open_sftp() as sftp:
            if args.operation=='start':
                if not args.source or not args.action:parser.error('start requires --source and --action')
                try:sftp.stat(folder+'/events.jsonl')
                except FileNotFoundError:pass
                else:raise ValueError('This branch already has a run; inspect it before another launch')
                mkdir(sftp,folder);local.mkdir(parents=True,exist_ok=True)
                prompt=f'''Continue the existing GTA interaction using the actual Pi world tools, on an isolated fork.
User action: {args.action}
Fork {OUT}/{args.source}/world.sqlite into {folder}/world.sqlite. Never modify the source world or its earlier artifacts. Query the fork's pending action, accepted scene, last RGB and time. If an action is pending, finish its exact continuation_goal before starting another intention. Preserve causal history and identities.
You plan, render, inspect and reconcile autonomously; no external model calls and no legacy nested-agent runners. Read skills/render-rollout/SKILL.md. Use coarse actor roots, facing and needed interaction states; do not add detailed joint constraints. NPC policies must be meaningful prospective behaviors, not frozen poses merely to simplify generation. Update them through npc_behaviors when justified. Keep established static geometry unchanged.
Use up to {args.generations} NEW H3 generations in this branch, including retries. Total segment duration may be 3 seconds (73 frames){' or 5.125 seconds (124 frames)' if args.max_duration==5 else ''}; choose enough time for the remaining action. H3 requires 17*n+5 frames, including the initial frame; executed duration is (frames-1)/24. Do not request 121 frames or pad conditions silently. Do not teleport or speed up implausibly to claim completion. Compile {args.width}x{args.height}. Render with GPU {args.gpu}, {args.steps} steps, cache_threshold=0, native reference_video=compiled whitebox.mp4, actual accepted RGB first_frame, controls=[] (no depth). Pass db in render args for historical-pair retrieval. Use the default epoch-15 LoRA. Write action prose without Picture/Video/Object tags; the tool binds these. No extra appearance crops. Consult renderer/comfy_h3.py for exact parameters. This is {'a low-resolution interaction test, not a high-resolution quality result' if args.width<1024 else 'the existing high-resolution recipe'}.
Use NEW segment/attempt directories under {folder}. A clip boundary is not an action endpoint: continuing moves need explicit end_velocity, and continuing camera paths need camera_end_motion. Do not insert wait/recover or ease the camera to rest just to end a clip. Keep walking/following/orbiting until the user intention actually calls for a stop. The compiler uses coarse actor volumes in both RGB and depth, not articulated limbs. Inspect actual generated video via inspect_video and its embedded contact sheet/opening/ending images, with extra frames only when needed. Separately judge action progress, natural motion, NPC behavior, identity/background and clearance. Screen displacement under camera motion alone is not proof of locomotion. Use review_contract IDs; optional existing NPC endpoint motion assessments are allowed. If accepted, commit the actual supported state (including partial progress and remaining goal) BEFORE planning the next segment. Retain uncertainty for monocular metric estimates. Never force ACCEPT or retain a visibly wrong endpoint to finish the task. Rejected videos must not advance world time.
Continue the same action until visibly complete or the generation budget is exhausted. Stop afterward; do not invent a new user intention. Write {folder}/trial_summary.json with inherited history, each generation's outcome, accepted revisions/times, remaining action, quality failures and artifact paths. Do not overwrite older summaries. Keep accepted historical segments untouched.'''
                (local/'prompt.txt').write_text(prompt,encoding='utf-8')
                sftp.put(str(local/'prompt.txt'),folder+'/prompt.txt')
                session='gta-'+args.branch.split('/')[0]+'-20261102'
                command=[REMOTE+'/.venv/bin/python',REMOTE+'/harness/pi/run.py','--session',session,
                         '--prompt-file',folder+'/prompt.txt','--events',folder+'/events.jsonl']
                script='export PATH=/home/star/.nvm/versions/node/v22.23.2/bin:$PATH\n'
                script+='unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy\n'
                script+='cd '+shlex.quote(REMOTE)+'\n'+shlex.join(command)+'\n'
                script+='result=$?\nprintf "%s\\n" "$result" > '+shlex.quote(folder+'/exit.txt')+'\n'
                with sftp.open(folder+'/run.sh','w') as f:f.write(script)
                _,stdout,_=client.exec_command('nohup bash '+shlex.quote(folder+'/run.sh')+' > '+shlex.quote(folder+'/console.log')+' 2>&1 < /dev/null & echo $!')
                receipt={'branch':args.branch,'pid':stdout.read().decode().strip(),'started_epoch':time.time(),
                         'source':args.source,'generation_budget':args.generations}
                (local/'launch.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt))
            elif args.operation=='status':
                code='''import json,pathlib,time
p=pathlib.Path(FOLDER);events=[]
for line in (p/'events.jsonl').open():
 try:events.append(json.loads(line))
 except json.JSONDecodeError:pass
calls=[e for e in events if e.get('type')=='tool_execution_start']
errors=[e for e in events if e.get('type')=='tool_execution_end' and e.get('isError')]
last=calls[-1] if calls else {};a=last.get('args',{})
completed={e.get('toolCallId') for e in events if e.get('type')=='tool_execution_end'}
print(json.dumps({'exit':(p/'exit.txt').read_text().strip() if (p/'exit.txt').exists() else None,
'calls':len(calls),'errors':len(errors),'last_tool':last.get('toolName'),'last_operation':a.get('operation'),
'last_path':a.get('path'),'waiting_on_tool':bool(last) and last.get('toolCallId') not in completed,
'event_log_age_sec':round(time.time()-(p/'events.jsonl').stat().st_mtime,1),
'last_error':str(errors[-1].get('result'))[-1600:] if errors and errors[-1].get('toolCallId')==last.get('toolCallId') else None}))
for log in p.rglob('generation.log'):print(str(log.relative_to(p))+': '+log.read_text(errors='replace')[-180:])
if (p/'trial_summary.json').exists():print((p/'trial_summary.json').read_text())
'''.replace('FOLDER',repr(folder))
                _,stdout,_=client.exec_command('python3 -c '+shlex.quote(code));print(stdout.read().decode())
            else:
                def fetch(remote,dest):
                    dest.mkdir(parents=True,exist_ok=True)
                    for entry in sftp.listdir_attr(remote):
                        if stat.S_ISDIR(entry.st_mode):
                            if entry.filename not in ['proxy','depth','frames']:fetch(remote+'/'+entry.filename,dest/entry.filename)
                        elif Path(entry.filename).suffix in ['.json','.txt','.mp4','.png','.jpg','.sqlite']:
                            sftp.get(remote+'/'+entry.filename,str(dest/entry.filename))
                fetch(folder,local);print('Fetched '+args.branch)
    finally:client.close()


if __name__=='__main__':main()
