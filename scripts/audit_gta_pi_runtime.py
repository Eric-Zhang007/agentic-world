"""Summarize trial runtime and gateway usage without copying image event payloads."""
import json
from pathlib import Path
import shlex
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from server_exec import connect

REMOTE_CODE=r'''
import json,pathlib
base=pathlib.Path('/home/star/agentic_world/outputs/gta_pi_loop_20261102')
results=[]
trials=[('rural','loop_debug'),('urban','loop'),('harbor','loop_npc'),('rural/continuation',''),('rural/road_edge',''),('urban/npc_walk','')]
for name,phase in trials:
    folder=base/name
    event_file=folder/(phase+'_events.jsonl' if phase else 'events.jsonl')
    if not event_file.exists():continue
    messages=[];calls=[];errors=0;error_messages=[]
    for line in event_file.open():
        try:e=json.loads(line)
        except json.JSONDecodeError:continue
        if e.get('type')=='message_end' and e.get('message',{}).get('role')=='assistant':
            messages.append(e['message'])
        if e.get('type')=='tool_execution_start':calls.append(e)
        if e.get('type')=='tool_execution_end' and e.get('isError'):
            errors+=1
            error_messages.append(' '.join(c.get('text','') for c in e.get('result',{}).get('content',[]) if c.get('type')=='text')[-1200:])
    usage={}
    for message in messages:
        for k,v in message.get('usage',{}).items():
            if isinstance(v,(int,float)):usage[k]=usage.get(k,0)+v
    generations=[]
    for log in sorted(folder.rglob('generation.log')):
        if any(other!=name and other.startswith(name+'/') and log.is_relative_to(base/other) for other,_ in trials):continue
        for line in log.read_text(errors='replace').splitlines():
            try:event=json.loads(line)
            except json.JSONDecodeError:continue
            if isinstance(event,dict) and event.get('stage')=='complete':
                generations.append({'attempt':str(log.parent.parent.relative_to(folder)),
                    'elapsed_sec':event.get('elapsed')})
    exitfile=folder/(phase+'_exit.txt' if phase else 'exit.txt')
    results.append({'case':name,'phase':phase,
        'exit_code':exitfile.read_text().strip() if exitfile.exists() else None,
        'assistant_turns':len(messages),'tool_calls':len(calls),'tool_errors':errors,
        'tool_error_messages':error_messages,
        'last_tool':calls[-1].get('toolName') if calls else None,
        'last_operation':calls[-1].get('args',{}).get('operation') if calls else None,
        'first_to_last_assistant_sec':(messages[-1]['timestamp']-messages[0]['timestamp'])/1000 if messages else None,
        'gateway_reported_usage':usage,'completed_generations':generations})
print(json.dumps(results))
'''


def main():
    client=connect()
    try:
        _,stdout,stderr=client.exec_command('python3 -c '+shlex.quote(REMOTE_CODE))
        raw=stdout.read().decode()
        if stdout.channel.recv_exit_status():raise RuntimeError(stderr.read().decode())
    finally:client.close()
    results=json.loads(raw)
    path=ROOT/'outputs/gta_pi_loop_20261102/runtime_audit.json'
    path.write_text(json.dumps(results,indent=2),encoding='utf-8')
    if '--brief' in sys.argv:
        print(json.dumps([{k:v for k,v in result.items() if k not in ['gateway_reported_usage','completed_generations','tool_error_messages']} for result in results],indent=2))
    else:print(json.dumps(results,indent=2))


if __name__=='__main__':main()
