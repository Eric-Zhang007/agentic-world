"""Verify an archived plan still replays exactly without writing its artifacts."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from whitebox.motion import simulate


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('plan',type=Path);args=p.parse_args()
    scene=json.loads((args.plan/'scene.json').read_text())
    action=json.loads((args.plan/'action.json').read_text())
    timeline=json.loads((args.plan/'timeline.json').read_text())
    start=json.loads((args.plan/'status.json').read_text())['world_start']
    replay,_,_=simulate(scene,action,start,timeline['fps'])
    assert replay==timeline,'Archived plan differs under current simulator'
    print(json.dumps({'plan':str(args.plan),'exact_replay':True,'frames':len(timeline['frames'])}))
