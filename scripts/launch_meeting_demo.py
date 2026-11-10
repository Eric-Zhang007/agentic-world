"""Deploy scoped demo changes and launch separate GPU jobs on the configured host."""
import sys
from pathlib import Path
import posixpath
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import server_exec


def main():
    local=Path(__file__).resolve().parents[1];remote='/home/star/agentic_world'
    files=['harness/pi/world_tools.py','harness/pi/world.ts','harness/pi/world_instructions.md',
           'harness/pi/test_extension.py','scripts/run_meeting_demo_job.py']
    files += ['outputs/meeting_demo/'+name for name in ('initial_scene.json','input.png','source.json',
        'keyboard_input_1.json','keyboard_input_2.json','action_prompt.txt','keyboard_prompt.txt')]
    client=server_exec.connect();sftp=client.open_sftp()
    def run(command):
        _,out,err=client.exec_command(command);text=out.read().decode();error=err.read().decode()
        if out.channel.recv_exit_status():raise RuntimeError(error)
        return text
    run('mkdir -p '+remote+'/outputs/meeting_demo/deploy_backup')
    for name in files:
        target=remote+'/'+name;run('mkdir -p '+posixpath.dirname(target))
        if not name.startswith('outputs/'):
            try:
                old=sftp.open(target,'rb').read()
                with sftp.open(remote+'/outputs/meeting_demo/deploy_backup/'+name.replace('/','__'),'wb') as f:f.write(old)
            except FileNotFoundError:pass
        sftp.put(str(local/name),target)
    print(run('cd '+remote+' && .venv/bin/python harness/pi/test_extension.py'))
    for lane in ('action','keyboard'):
        print(lane,run('cd '+remote+' && nohup .venv/bin/python scripts/run_meeting_demo_job.py --lane '+lane+
            ' > outputs/meeting_demo/'+lane+'_launcher.log 2>&1 < /dev/null & echo $!').strip())
    sftp.close();client.close()


if __name__=='__main__':main()
