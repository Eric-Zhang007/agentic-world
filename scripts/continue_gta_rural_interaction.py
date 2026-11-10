"""Run the remaining rural intention on a fork with the verified image-tool update."""
from pathlib import Path
import shlex
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from server_exec import connect


def main():
    remote='/home/star/agentic_world'
    base=remote+'/outputs/gta_pi_loop_20261102/rural'
    output=base+'/continuation'
    prompt=f'''Continue the remaining third user intention from your completed rural trial, on an isolated branch.
User action: move near the road edge and stop to wait, while the other pedestrian continues a slow natural walk. Preserve the existing world and avoid entering traffic.
This is a post-interface-update continuation, not part of the original four-generation baseline. The only change is richer deterministic tool output (embedded video images and review contract). Keep the same 512x288,73-frame,20-step,depth0.5,cache0 native-whitebox render recipe on GPU2. Do not claim high-resolution quality.
First fork {base}/world.sqlite into {output}/world.sqlite using world fork; never modify the source DB or earlier artifacts. Query the fork's actual state and last accepted RGB. Use this state and frame for the next plan, not the original image. Do not modify established scene geometry or identities. Execute the remaining user action using the existing NPC policy or a justified prospective policy. Use the render-rollout skill.
Use only ONE new H3 generation for this continuation, 3 seconds, starting from the inherited time9. No extra model requests outside this Pi session. If the action is only partially achieved, commit the actual supported state with continuation_goal and report the remaining action; if it fails, reject without advancing. Never force an ACCEPT just to finish.
Call inspect_video and actually review its embedded contact sheet and endpoint images; read individual frames only if needed. Honor review_contract IDs and final indices. Include explicit endpoint motion assessment. Use actual final_proxy returned by compile instead of guessed frame indices.
Write all new plans, videos, review and commit artifacts below {output}. Finish with {output}/trial_summary.json recording decision, action completion, world revision/time, inheritance of three earlier segments, limitations and whether extra image reads were needed. Do not overwrite {base}/trial_summary.json. Then stop; do not start another user intention.'''
    command=['/home/star/agentic_world/.venv/bin/python','-X','utf8',remote+'/harness/pi_review_probe/run.py',
             '--session','gta-rural-20261102','--events',output+'/events.jsonl','--prompt',prompt]
    client=connect()
    try:
        _,stdout,stderr=client.exec_command('mkdir -p '+output)
        if stdout.channel.recv_exit_status():raise RuntimeError(stderr.read().decode())
        script='export PATH=/home/star/.nvm/versions/node/v22.23.2/bin:$PATH\n'
        script+='unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy\n'
        script+='cd '+remote+'\n'+shlex.join(command)+'\n'
        script+='result=$?\nprintf "%s\\n" "$result" > '+output+'/exit.txt\n'
        with client.open_sftp() as sftp:
            with sftp.open(output+'/run.sh','w') as file:file.write(script)
        _,stdout,_=client.exec_command('nohup bash '+output+'/run.sh > '+output+'/console.log 2>&1 < /dev/null & echo $!')
        print('Started isolated rural continuation PID '+stdout.read().decode().strip())
    finally:client.close()


if __name__=='__main__':main()
