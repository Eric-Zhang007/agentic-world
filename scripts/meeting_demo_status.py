"""Fetch compact real Pi/tool/render progress, never image payloads or credentials."""
import json
from pathlib import Path
import shlex
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
import server_exec

CODE=r'''
import json,pathlib,sqlite3
root=pathlib.Path('/home/star/agentic_world/outputs/meeting_demo')
result={}
for lane in ('action','keyboard','interactive'):
 p=root/lane; calls=[]; messages=[]; errors=[]
 for f in sorted(p.rglob('events*.jsonl')):
  for line in f.open():
   try:e=json.loads(line)
   except json.JSONDecodeError:continue
   if e.get('type')=='tool_execution_start':
    a=e.get('args',{});calls.append({'tool':e.get('toolName'),'operation':a.get('operation'),'path':a.get('path'),'timestamp':e.get('timestamp')})
   if e.get('type')=='message_end' and e.get('message',{}).get('role')=='assistant':
    m=e['message'];messages.append(' '.join(c.get('text','') for c in m.get('content',[]) if c.get('type')=='text')[-1200:])
   if e.get('type')=='tool_execution_end' and e.get('isError'):
    errors.append(' '.join(c.get('text','') for c in e.get('result',{}).get('content',[]) if c.get('type')=='text')[-900:])
 generations=[]
 for f in p.rglob('generation.log'):
  content=f.read_text(errors='replace').splitlines()
  generations.append({'path':str(f.parent),'tail':content[-1][-250:] if content else ''})
 histories=[]
 if (p/'world.sqlite').exists():
  with sqlite3.connect('file:'+str(p/'world.sqlite')+'?mode=ro',uri=True) as db:
   row=db.execute("SELECT value FROM world_meta WHERE key='whitebox_history'").fetchone()
   histories=json.loads(row[0]) if row else []
 exits=list(p.rglob('job_exit*.json'))
 exit_file=max(exits,key=lambda x:x.stat().st_mtime) if exits else None
 result[lane]={'accepted_segments':len(histories),'recent_calls':calls[-6:],'last_message':messages[-1] if messages else '',
  'recent_errors':errors[-4:],'generations':generations,
  'job_exit':json.loads(exit_file.read_text()) if exit_file and (lane=='interactive' or exit_file.name=='job_exit_02.json') else None,
  'result':json.loads((p/'result.json').read_text()) if (p/'result.json').exists() else None}
print(json.dumps(result))
'''

def main():
    client=server_exec.connect()
    _,out,err=client.exec_command('python3 -c '+shlex.quote(CODE));raw=out.read().decode()
    if out.channel.recv_exit_status():raise RuntimeError(err.read().decode())
    client.close();result=json.loads(raw)
    path=ROOT/'outputs/meeting_demo/live_status.json';path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    if '--brief' in sys.argv:
        result={k:{'accepted_segments':v['accepted_segments'],'last_call':v['recent_calls'][-1:] ,
                   'generations':v['generations'],'job_exit':v['job_exit']} for k,v in result.items()}
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
