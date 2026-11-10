"""Test revised image tool in an isolated harness while baseline trials run."""
import json
from pathlib import Path
import shlex
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from server_exec import connect


def main():
    remote='/home/star/agentic_world'
    harness=remote+'/harness/pi_review_probe'
    out=remote+'/outputs/gta_pi_loop_20261102/review_interface_probe'
    client=connect()
    try:
        if '--collect' in sys.argv:
            code='''import json,pathlib
p=pathlib.Path(OUT);events=[]
for line in (p/'events.jsonl').open():
 try:events.append(json.loads(line))
 except json.JSONDecodeError:pass
messages=[e['message'] for e in events if e.get('type')=='message_end' and e.get('message',{}).get('role')=='assistant']
calls=[e for e in events if e.get('type')=='tool_execution_start']
ends=[e for e in events if e.get('type')=='tool_execution_end']
print(json.dumps({'exit':(p/'exit.txt').read_text().strip() if (p/'exit.txt').exists() else None,
'tool_calls':[{'tool':e.get('toolName'),'operation':e.get('args',{}).get('operation')} for e in calls],
'tool_results':[{'isError':e.get('isError'),'image_count':sum(c.get('type')=='image' for c in e.get('result',{}).get('content',[]))} for e in ends],
'last_assistant':[c.get('text') for c in messages[-1].get('content',[]) if c.get('type')=='text'] if messages else []}))
'''.replace('OUT',repr(out))
            _,stdout,stderr=client.exec_command('python3 -c '+shlex.quote(code))
            raw=stdout.read().decode()
            if stdout.channel.recv_exit_status():raise RuntimeError(stderr.read().decode())
            result=json.loads(raw)
            (ROOT/'outputs/gta_pi_loop_20261102/review_interface_probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(result,ensure_ascii=False,indent=2))
            return
        _,stdout,stderr=client.exec_command('mkdir -p '+shlex.quote(harness)+' '+shlex.quote(out))
        if stdout.channel.recv_exit_status():raise RuntimeError(stderr.read().decode())
        with client.open_sftp() as sftp:
            for name in ['run.py','world.ts','world_tools.py','world_instructions.md']:
                sftp.put(str(ROOT/'harness/pi'/name),harness+'/'+name)
        _,stdout,stderr=client.exec_command('test -e '+harness+'/node_modules || ln -s '+remote+'/harness/pi/node_modules '+harness+'/node_modules')
        if stdout.channel.recv_exit_status():raise RuntimeError(stderr.read().decode())
        prompt='''This is an isolated read-only video evidence tool verification. Do not plan, render, commit, edit existing files, or call any external model yourself.
Call world inspect_video EXACTLY ONCE with plan=%s/outputs/gta_pi_loop_20261102/rural/segment01_attempt02/plan, video=%s/outputs/gta_pi_loop_20261102/rural/segment01_attempt02/render/controlled.mp4, output=%s/evidence, interval=0.5.
Use the images embedded in that tool result directly; do not read other files/images unless the image content is missing. Then answer briefly: which images you received, the required actor/motion IDs from review_contract, whether the foreground woman advances, and whether a final stable stop is clearly visible. State uncertainty. Do not inspect or quote the earlier agent review; judge the actual returned images.'''%(remote,remote,out)
        command=['/home/star/agentic_world/.venv/bin/python','-X','utf8',harness+'/run.py',
                 '--session','gta-review-interface-20261102','--events',out+'/events.jsonl','--prompt',prompt]
        script='export PATH=/home/star/.nvm/versions/node/v22.23.2/bin:$PATH\n'
        script+='unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy\n'
        script+='cd '+remote+'\n'+shlex.join(command)+'\n'
        script+='result=$?\nprintf "%s\\n" "$result" > '+out+'/exit.txt\n'
        with client.open_sftp() as sftp:
            with sftp.open(out+'/run.sh','w') as file:file.write(script)
        _,stdout,_=client.exec_command('nohup bash '+out+'/run.sh > '+out+'/console.log 2>&1 < /dev/null & echo $!')
        print(json.dumps({'pid':stdout.read().decode().strip(),'output':out}))
    finally:client.close()


if __name__=='__main__':main()
