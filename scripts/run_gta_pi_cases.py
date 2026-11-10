"""Stage RGB-only cases and run the real Pi CLI on the existing GPU server."""
import argparse
import json
from pathlib import Path
import shlex
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from server_exec import connect
LOCAL = ROOT/'outputs/gta_pi_loop_20261102'
REMOTE = '/home/star/agentic_world'
OUT = REMOTE+'/outputs/gta_pi_loop_20261102'
CASES = {0:'rural', 3:'urban', 7:'harbor'}


def mkdir(sftp, path):
    try: sftp.stat(path)
    except FileNotFoundError:
        mkdir(sftp, path.rsplit('/',1)[0])
        sftp.mkdir(path)


def init_prompt(case):
    folder = OUT+'/'+case
    return f'''Perform an RGB-only single-image whitebox reconstruction benchmark.
Work only in {folder}. Read {folder}/input.png with your image tool.
DO NOT read any dataset whitebox, source state, other experiment scenes, original
camera calibration, dataset JSON or other output directories. You may read the
implementation and skill files to learn the schema. No extra API/model calls.
You are the agent: author the scene yourself, not an old nested-agent runner.
Use world preview to compare your proposed whitebox camera image to the RGB.
Allow at most THREE proposal previews, then initialize world.sqlite once from
your chosen scene. Save final proposal as scene.json and reconstruction_report.json
with attempts, alignment issues, supported and missing assets, uncertainty.
Do not render H3 yet. This stage should be quick and reveal resource limitations.

Represent the visible walking player and relevant NPCs, main terrain/road surface,
major building facades, barriers, poles/trees, vehicles if present. Background is
important. Use the available actor/box/staff/stairs resources; compose boxes for
coarse structures. Do not invent weapon props. Actor roots are on terrain;
box positions are CENTERS. Ground markings/road slabs may be non-solid boxes
just below y=0. Infer a convenient local coordinate system and keep units plausible.
Camera fov is vertical degrees. Match visible screen locations and silhouettes,
not just categories. Every actor needs player/guard/patrol behavior; avoid guard
chasing when no chase was requested. Inferred geometry is INFERRED with reason.
Read whitebox/world.py validation rules and whitebox/motion.py before proposing.
Scene schema: schema_version=1, seed, lore, terrain(size,resolution,seed,
height_amplitude,flatten_radius), camera(position,target,fov), entities(id,kind,
category,position,rotation_y,size,color,provenance,reason,description,behavior).
Use flat terrain for this baseline if visible ground is locally flat. Distant
unseen geometry must not be claimed as observed. A maximum of 25 coarse entities
is sufficient for the first pass. Do not sacrifice visible background landmarks
merely to get a valid scene. Inspect all returned previews visually.
Final output should name chosen preview, rough deficiencies, and world DB.
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['stage','start','status','fetch'])
    p.add_argument('--case',choices=list(CASES.values()),default='rural')
    p.add_argument('--phase',default='init');a=p.parse_args()
    client=connect()
    try:
        with client.open_sftp() as sftp:
            if a.mode=='stage':
                candidates=json.loads((LOCAL/'candidates.json').read_text())
                for index,name in CASES.items():
                    row=candidates[index];folder=LOCAL/name;folder.mkdir(exist_ok=True)
                    from PIL import Image
                    Image.open(row['rgb']).save(folder/'input.png')
                    # Evaluation references remain local; the agent only receives RGB.
                    Image.open(row['whitebox']).save(folder/'dataset_whitebox.png')
                    (folder/'source.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
                    (folder/'init_prompt.txt').write_text(init_prompt(name),encoding='utf-8')
                    mkdir(sftp,OUT+'/'+name)
                    for filename in ['input.png','init_prompt.txt']:
                        sftp.put(str(folder/filename),OUT+'/'+name+'/'+filename)
                for filename in ['run.py','world_tools.py','world.ts','world_instructions.md']:
                    sftp.put(str(ROOT/'harness/pi'/filename),REMOTE+'/harness/pi/'+filename)
                print('Staged three RGB-only cases and current harness')
            elif a.mode=='start':
                folder=OUT+'/'+a.case
                prompt=LOCAL/a.case/(a.phase+'_prompt.txt')
                if not prompt.exists() and a.phase.startswith('init'):
                    prompt.write_text(init_prompt(a.case),encoding='utf-8')
                sftp.put(str(prompt),folder+'/'+prompt.name)
                events=folder+'/'+a.phase+'_events.jsonl'
                command=['/home/star/agentic_world/.venv/bin/python','-X','utf8',REMOTE+'/harness/pi/run.py',
                    '--session','gta-'+a.case+'-20261102','--prompt-file',folder+'/'+prompt.name,'--events',events]
                script='export PATH=/home/star/.nvm/versions/node/v22.23.2/bin:$PATH\n'
                script+='unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy\n'
                script+='cd '+shlex.quote(REMOTE)+'\n'+shlex.join(command)+'\n'
                script+='result=$?\nprintf "%s\\n" "$result" > '+shlex.quote(folder+'/'+a.phase+'_exit.txt')+'\n'
                scriptpath=folder+'/'+a.phase+'_run.sh'
                with sftp.open(scriptpath,'w') as f:f.write(script)
                _,stdout,_=client.exec_command('nohup bash '+shlex.quote(scriptpath)+' > '+shlex.quote(folder+'/'+a.phase+'_console.log')+' 2>&1 < /dev/null & echo $!')
                pid=stdout.read().decode().strip()
                print(json.dumps({'case':a.case,'phase':a.phase,'pid':pid,'started':time.time()}))
            elif a.mode=='status':
                code='''import json,pathlib
for name in ['rural','urban','harbor']:
 p=pathlib.Path(BASE)/name; f=p/(PHASE+'_events.jsonl')
 if not f.exists(): print(name+': not started'); continue
 events=[]
 for line in f.open():
  try: events.append(json.loads(line))
  except json.JSONDecodeError: pass
 calls=[e for e in events if e['type']=='tool_execution_start']
 errors=[e for e in events if e['type']=='tool_execution_end' and e.get('isError')]
 messages=[e['message'] for e in events if e['type']=='message_end' and e.get('message',{}).get('role')=='assistant']
 last=calls[-1] if calls else {}; args=last.get('args',{})
 if 'content' in args: args=dict(args,content='<file content omitted>')
 if args.get('operation')=='render':
  args=dict(operation='render',renderer=args['args']['renderer'],output=args['args']['request']['output_dir'])
 exitfile=p/(PHASE+'_exit.txt')
 print(json.dumps(dict(case=name,exit=exitfile.read_text().strip() if exitfile.exists() else None,
  tool_calls=len(calls),errors=len(errors),last_tool=last.get('toolName'),last_args=args,
  last_error=str(errors[-1].get('result',''))[-1600:] if errors else None,
  elapsed_model_wall_sec=(messages[-1]['timestamp']-messages[0]['timestamp'])/1000 if messages else None,
  last_event=events[-1]['type'] if events else None)))
 for log in p.rglob('generation.log'):
  print(str(log)+': '+log.read_text(errors='replace')[-240:].replace(chr(13),' '))
'''.replace('BASE',repr(OUT)).replace('PHASE',repr(a.phase))
                _,stdout,_=client.exec_command('/home/star/agentic_world/.venv/bin/python -c '+shlex.quote(code))
                print(stdout.read().decode())
            else:
                def fetch(remote, local):
                    import stat
                    local.mkdir(parents=True,exist_ok=True)
                    for entry in sftp.listdir_attr(remote):
                        if stat.S_ISDIR(entry.st_mode):
                            if entry.filename not in ['proxy','frames','depth']:fetch(remote+'/'+entry.filename,local/entry.filename)
                        elif Path(entry.filename).suffix in ['.png','.jpg','.json','.txt','.mp4','.md','.sqlite']:
                            sftp.get(remote+'/'+entry.filename,str(local/entry.filename))
                fetch(OUT+'/'+a.case,LOCAL/a.case)
                print('Fetched '+a.case)
    finally:client.close()


if __name__=='__main__':main()
