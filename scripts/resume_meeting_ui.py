"""Resume the preserved rejected browser request after fixing reroll orchestration."""
from pathlib import Path
import shlex
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import server_exec
code='''import subprocess
root='/home/star/agentic_world'
request=root+'/outputs/meeting_demo/interactive/jobs/live_20261103_004549_188088'
with open(request+'/retry_launcher.log','x') as log:
 p=subprocess.Popen([root+'/.venv/bin/python',root+'/scripts/run_interactive_rollout.py','--request',request+'/request.json','--resume-rejected'],cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
print(p.pid)
'''
client=server_exec.connect()
_,out,err=client.exec_command('python3 -c '+shlex.quote(code))
print(out.read().decode(),err.read().decode())
result=out.channel.recv_exit_status();client.close()
raise SystemExit(result)
