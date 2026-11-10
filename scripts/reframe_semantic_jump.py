"""Preserve the agent motion and widen only the isolated test camera."""
import json
import sys
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'harness/pi'))
from world_tools import read,save,dispatch


def main():
    source=ROOT/'outputs/semantic_actions/jump/plan_resumed_01'
    target=ROOT/'outputs/semantic_actions/jump_framed'
    target.mkdir(exist_ok=False)
    scene=read(source/'scene.json');x,y,z=scene['entities'][0]['position']
    scene['camera']={'position':[x+4,y+4,z+7],'target':[x+1.5,y+1.5,z],'fov':50}
    save(target/'scene.json',scene)
    dispatch('initialize',{'scene':str(target/'scene.json'),'db':str(target/'world.sqlite')})
    dispatch('plan',{'db':str(target/'world.sqlite'),'action':str(source/'action.json'),'output':str(target/'plan')})
    old=read(source/'timeline.json');new=read(target/'plan/timeline.json')
    assert [f['entities'] for f in old['frames']]==[f['entities'] for f in new['frames']]
    save(target/'provenance.json',{'agent_action':str(source/'action.json'),
        'change':'Test camera only, to include jump apex; identical entity trajectories',
        'world_committed':False})
    subprocess.run([sys.executable,str(ROOT/'scripts/compile_semantic_trial.py'),
        '--plan',str(target/'plan'),'--output',str(target/'compiled')],check=True)


if __name__=='__main__':main()
