"""Run the real Pi CLI with isolated project sessions and the existing gateway."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/default.yaml')
    p.add_argument('--session',required=True,help='Explicit isolated session ID')
    p.add_argument('--prompt')
    p.add_argument('--prompt-file',type=Path)
    p.add_argument('--image',type=Path,action='append',default=[])
    p.add_argument('--review-only',action='store_true')
    p.add_argument('--check',action='store_true')
    p.add_argument('--events',type=Path)
    a=p.parse_args()
    if a.prompt_file:
        if a.prompt: p.error('Choose --prompt or --prompt-file')
        a.prompt=a.prompt_file.read_text(encoding='utf-8')
    import yaml
    cfg=yaml.safe_load(a.config.read_text(encoding='utf-8'))['agent']
    cli=HERE/'node_modules/@earendil-works/pi-coding-agent/dist/cli.js'
    if not cli.exists():raise SystemExit('Run npm install in harness/pi first.')
    runtime=ROOT/'outputs/pi_harness'
    runtime.mkdir(parents=True,exist_ok=True)
    api=cfg.get('api','openai-completions')
    models={'providers':{'world-gateway':{'baseUrl':cfg['base_url'],'api':api,
        'apiKey':'${WORLD_AGENT_API_KEY}','models':[{'id':cfg['model'],'name':cfg['model'],
        'input':['text','image'],'contextWindow':65536,'maxTokens':4096,
        'compat':{'supportsStore':False,'supportsDeveloperRole':False,'maxTokensField':'max_tokens'}}]}}}
    if api == 'openai-responses':
        model=models['providers']['world-gateway']['models'][0]
        model.pop('compat')
        model['reasoning']=True
    (runtime/'models.json').write_text(json.dumps(models,indent=2),encoding='utf-8')
    env=dict(os.environ,PI_CODING_AGENT_DIR=str(runtime),PI_OFFLINE='1',PI_TELEMETRY='0',
             WORLD_PYTHON=sys.executable,
             WORLD_AGENT_API_KEY=os.environ.get(cfg.get('env_key','OPENAI_API_KEY')) or cfg.get('api_key',''))
    env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
    if not env['WORLD_AGENT_API_KEY']:raise SystemExit('No configured gateway key')
    command=[shutil.which('node') or 'node',str(cli),'--offline','--no-approve',
             '--provider','world-gateway','--model',cfg['model'],'--session-id',a.session,
             '--session-dir',str(runtime/'sessions'),'--append-system-prompt',str(HERE/'world_instructions.md'),
             '--skill',str(ROOT/'skills/render-rollout/SKILL.md')]
    if cfg.get('reasoning_effort'):
        command+=['--thinking',cfg['reasoning_effort']]
    if a.check:command+=['--list-models',cfg['model']]
    else:
        if not a.review_only:command+=['--extension',str(HERE/'world.ts')]
        if a.review_only:command+=['--tools','read,grep,find,ls']
        if a.events:command+=['--mode','json']
        elif a.prompt:command+=['--print']
        command += ['@'+str(image.resolve()) for image in a.image]
        if a.prompt:command+=[a.prompt]
    if a.events:
        a.events.parent.mkdir(parents=True,exist_ok=True)
        with a.events.open('x',encoding='utf-8') as out:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=out)
        last_assistant=None
        for line in a.events.read_text(encoding='utf-8').splitlines():
            event=json.loads(line)
            if event.get('type')=='message_end' and event.get('message',{}).get('role')=='assistant':
                last_assistant=event['message']
        if last_assistant and last_assistant.get('stopReason')=='error':
            print('Pi model request failed; see event log.',file=sys.stderr)
            raise SystemExit(1)
    else:result=subprocess.run(command,cwd=ROOT,env=env)
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
