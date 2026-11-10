"""Compile task memory against an unchanged saved plan; do not commit or render."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys

import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from state.db import WorldDB
from whitebox.world import scene_from_db
from whitebox.motion import simulate
from whitebox.appearance_rollout import retrieve_for_task
from agent.render_contract import compile_prompt


def plan_prefix_frames(plan, timeline):
    alignment=json.loads((plan/'alignment.json').read_text())
    count=alignment.get('prefix_frames')
    if type(count) is not int or count<1 or (count+len(timeline['frames']))%17!=5:
        raise ValueError('Saved plan has invalid historical prefix/frame alignment')
    return count


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    db=WorldDB(':memory:')
    try:
        with sqlite3.connect(a.world.resolve().as_uri()+'?mode=ro',uri=True) as source:
            source.backup(db.connection)
        before=list(db.connection.iterdump())
        scene=json.loads((a.plan/'scene.json').read_text())
        action=json.loads((a.plan/'action.json').read_text())
        timeline=json.loads((a.plan/'timeline.json').read_text())
        prefix_frames=plan_prefix_frames(a.plan,timeline)
        if scene!=scene_from_db(db):raise ValueError('Plan does not start at accepted world')
        replay,_,report=simulate(scene,action,db.get_meta('tick'),timeline['fps'])
        if replay!=timeline or not report['valid']:raise ValueError('Plan deterministic replay failed')
        config=yaml.safe_load(a.config.read_text())['agent']
        reference=Path(db.get_meta('whitebox_last_preview'))
        memory=retrieve_for_task(db,config,a.output/'appearance_memory',
            {'goal':action.get('goal'),'action':action,'camera_end':timeline['frames'][-1]['camera']},reference)
        compile_prompt(config,scene,action,timeline,reference,a.output,
                       prefix_frames=prefix_frames,memory=memory)
        if list(db.connection.iterdump())!=before:raise ValueError('Compilation changed world snapshot')
        (a.output/'source.json').write_text(json.dumps({'world':str(a.world.resolve()),
            'plan':str(a.plan.resolve()),'reference':str(reference),'world_commit':False,
            'deterministic_plan_replay':True,'prefix_frames':prefix_frames},indent=2))
        print('Saved-plan memory prompt compiled without world changes',flush=True)
    finally:
        db.connection.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['world','plan','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/default.yaml')
    main(p.parse_args())
