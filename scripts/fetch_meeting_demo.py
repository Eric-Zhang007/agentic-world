"""Collect real demo evidence; omit embedded image payloads and private reasoning."""
import json
from pathlib import Path
import shlex
import stat
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
import server_exec

CODE=r'''
import json,pathlib,sqlite3
root=pathlib.Path('/home/star/agentic_world/outputs/meeting_demo')
for lane in ('action','keyboard','interactive'):
 folder=root/lane
 if (folder/'world.sqlite').exists():
  with sqlite3.connect('file:'+str(folder/'world.sqlite')+'?mode=ro',uri=True) as src:
   with sqlite3.connect(folder/'delivery_world.sqlite') as dst:src.backup(dst)
 records=[]
 for file in sorted(folder.rglob('events*.jsonl')):
  for line in file.open():
   try:event=json.loads(line)
   except json.JSONDecodeError:continue
   kind=event.get('type')
   if kind=='message_end':
    message=event.get('message',{})
    text=[x.get('text','') for x in message.get('content',[]) if x.get('type')=='text']
    if text:records.append({'event':kind,'role':message.get('role'),'timestamp':message.get('timestamp'),'text':text})
   elif kind=='tool_execution_start':
    records.append({'event':kind,'tool':event.get('toolName'),'id':event.get('toolCallId'),'args':event.get('args')})
   elif kind=='tool_execution_end':
    text=[x.get('text','') for x in event.get('result',{}).get('content',[]) if x.get('type')=='text']
    records.append({'event':kind,'id':event.get('toolCallId'),'error':event.get('isError',False),'text':text})
 (folder/'interaction_record.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
print('Prepared consistent database snapshots and visible interaction records')
'''


def main():
    client=server_exec.connect();_,out,err=client.exec_command('python3 -c '+shlex.quote(CODE))
    print(out.read().decode())
    if out.channel.recv_exit_status():raise RuntimeError(err.read().decode())
    sftp=client.open_sftp();remote='/home/star/agentic_world/outputs/meeting_demo';local=ROOT/'outputs/meeting_demo'
    count=0
    def walk(path):
        nonlocal count
        for entry in sftp.listdir_attr(path):
            name=entry.filename;full=path+'/'+name;relative=full.removeprefix(remote+'/')
            parts=Path(relative).parts
            if stat.S_ISDIR(entry.st_mode):
                if name in ('deploy_backup','depth','proxy','views'):continue
                walk(full);continue
            if name.endswith('.sqlite') and name!='delivery_world.sqlite':continue
            if name.startswith('events') and name.endswith('.jsonl'):continue
            if Path(name).suffix.lower() not in ('.json','.txt','.mp4','.png','.jpg','.log','.sqlite'):continue
            if name=='source_resampled.mp4':continue
            target=local/relative;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and target.stat().st_size==entry.st_size:continue
            sftp.get(full,str(target));count+=1
    for lane in ('action','keyboard','interactive'):walk(remote+'/'+lane)
    sftp.close();client.close();print('Downloaded',count,'evidence files')


if __name__=='__main__':main()
