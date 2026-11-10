"""Fork accepted evidence for a real UI test; never mutate the two demo lanes."""
from pathlib import Path
import shlex
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import server_exec

code = r'''
from pathlib import Path
import sqlite3, subprocess, json
root=Path('/home/star/agentic_world')
out=root/'outputs/meeting_demo/interactive'
out.mkdir(exist_ok=False)
source=root/'outputs/meeting_demo/action/world.sqlite'
with sqlite3.connect('file:'+str(source)+'?mode=ro',uri=True) as src:
 with sqlite3.connect(out/'world.sqlite') as dst:src.backup(dst)
with sqlite3.connect(out/'world.sqlite') as db:
 meta={k:json.loads(v) for k,v in db.execute('SELECT key,value FROM world_meta')}
assert meta.get('whitebox_history'), 'An accepted source segment is required'
(out/'source.json').write_text(json.dumps({'source':str(source),'revision':meta['whitebox_revision'],'history_length':len(meta['whitebox_history']),'purpose':'UI integration fork; existing demo history unchanged'},indent=2))
with (out/'server.log').open('x') as log:
 p=subprocess.Popen([str(root/'.venv/bin/python'),str(root/'scripts/serve_controls.py'),'--db',str(out/'world.sqlite'),'--port','8768','--gpu','2','--output',str(out/'jobs')],cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
print(json.dumps({'pid':p.pid,'database':str(out/'world.sqlite'),'port':8768}))
'''
client=server_exec.connect()
_,out,err=client.exec_command('python3 -c '+shlex.quote(code))
print(out.read().decode(),err.read().decode())
status=out.channel.recv_exit_status()
client.close()
raise SystemExit(status)
