"""Branch accepted history, then exercise persistent posture through real H3."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from state.db import WorldDB
from whitebox.world import scene_from_db
from whitebox.video_reconcile import BOUNDARY_PROMPT
from scripts.reconcile_whitebox_video import prepare_plan,main as review,save
from scripts.run_video_reconcile_trial import conditions,render_proxy
from renderer.comfy_h3 import ComfyH3Renderer
from agent.action_planner import json_call


def main(a):
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    with sqlite3.connect(a.parent.resolve().as_uri()+'?mode=ro',uri=True) as source:
        with sqlite3.connect(root/'world.sqlite') as target:source.backup(target)
    db=WorldDB(root/'world.sqlite');config=yaml.safe_load(a.config.read_text())
    baseline=deepcopy(db.get_meta('whitebox_history'));save(root/'parent_history.json',baseline)
    save(root/'world_initial.json',scene_from_db(db))
    save(root/'branch.json',{'parent':str(a.parent.resolve()),'history_records':len(baseline),
                             'revision':db.get_meta('whitebox_revision'),'world_time':db.get_meta('tick')})
    db.set_meta('posture_trial_branch',str(a.parent.resolve()))
    goal=('Have tiger_guard lower its body into a supported low_guard stance over about three seconds, '
          'planting both forepaws on the courtyard without translating its root or attacking. '
          'This is an alert defensive posture, not a hit, injury or defeat. player remains watching '
          'and retains player_staff; natural grip adjustment is allowed. Fixed camera and no new props. '
          'Use tiger_guard as actor_id and a posture phase targeting low_guard.')
    next_goal=('Observe for three seconds from the ACTUAL accepted endpoint. Preserve any accepted tiger '
               'low_guard or crouched support posture; do not reset it to standing. Player retains the '
               'staff with natural idle grip adjustment. No attack, new props, root travel or camera motion. '
               'Use a wait phase, not a posture reset.')
    reference=Path(db.get_meta('whitebox_last_preview'))
    pending=None;records=[]
    try:
        for index in range(a.segments):
            segment=root/f'segment_{index:02d}';segment.mkdir()
            plan=segment/'plan'
            if pending:
                import shutil
                shutil.copytree(pending,plan)
            else:prepare_plan(db,config['agent'],goal,plan,str(reference))
            save(segment/'history_before.json',db.get_meta('whitebox_history'))
            render_proxy(plan,segment/'planned_preview')
            first,last,count=conditions(plan,reference,segment/'conditions',config,'depth')
            check=json_call(config['agent'],
                'Review a coarse WHITEBOX plan, not generated-video evidence. Return '
                '{usable:boolean,evidence:[string],issues:[string]}. Images are the accepted visual reference, '
                'initial whitebox and terminal whitebox. Judge readable posture, ground support, weapon '
                'retention and stable scene. Coarse mannequins need not match textures or fine articulation. '
                'A low_guard actor is supported on feet and forepaws, still alert, not injured. '
                'Do not impose two-handed grip if the user only requires retaining the staff.',
                {'action':json.loads((plan/'action.json').read_text())},
                [reference,segment/'planned_preview/camera.png',sorted((segment/'planned_preview').glob('frame_*.png'))[-1]])
            save(segment/'condition_review.json',check)
            if check.get('usable') is not True:raise ValueError('Agent rejected whitebox preview')
            action=json.loads((plan/'action.json').read_text())
            prompt=('Continuous fantasy game courtyard matching <Picture 1>. Same monkey warrior holding '
                    'the same staff and same unarmed tiger guardian; no new weapons or debris. '
                    'No strike, impact, injury or defeat. Fixed camera. '+action['goal']+'\n'+BOUNDARY_PROMPT)
            (segment/'render_prompt.txt').write_text(prompt,encoding='utf-8')
            video=ComfyH3Renderer(gpu=a.gpu,steps=40,frames=count,width=1024,height=768,
                kernel_backend='triton').render(prompt,None,segment/'render',first_frame=first,
                anchor_first=True,control_video=segment/'conditions/depth.npz',strength=.6,seed=a.seed+index)
            review(argparse.Namespace(world=root/'world.sqlite',prepare_goal=None,plan=plan,video=video,
                output=segment/'review',config=a.config,next_goal=next_goal,resume_review=False))
            status=json.loads((segment/'review/status.json').read_text())
            history=db.get_meta('whitebox_history')
            prior=json.loads((segment/'history_before.json').read_text())
            assert history[:len(prior)]==prior
            if status['status']!='COMMITTED':
                assert history==prior
                records.append({'segment':index,'status':'REGENERATE','history_unchanged':True})
                break
            render_proxy(plan,segment/'corrected_preview',timeline=segment/'review/corrected_timeline.json')
            current=scene_from_db(db);pending=segment/'review/next_plan'
            assert json.loads((pending/'scene.json').read_text())==current
            reference=Path(db.get_meta('whitebox_last_preview'))
            assert Path(json.loads((pending/'status.json').read_text())['reference'])==reference
            records.append({'segment':index,'status':'COMMITTED','revision':db.get_meta('whitebox_revision'),
                'world_time':db.get_meta('tick'),'history_prefix_unchanged':True,'next_plan_uses_actual':True,
                'actors':[{'id':e['id'],'posture_weights':e.get('posture_weights'),
                           'support_contacts':e.get('support_contacts'),'grip':e.get('grip','both')}
                          for e in current['entities'] if e['kind']=='actor']})
    finally:
        assert db.get_meta('whitebox_history')[:len(baseline)]==baseline
        save(root/'world_current.json',scene_from_db(db))
        save(root/'posture_verification.json',{'segments':records,'parent_history_unchanged':True,
                                              'revision':db.get_meta('whitebox_revision'),'world_time':db.get_meta('tick')})
        db.connection.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--segments',type=int,default=2)
    p.add_argument('--seed',type=int,default=47);p.add_argument('--gpu',default='3')
    p.add_argument('--config',type=Path,default=ROOT/'configs/default.yaml')
    main(p.parse_args())
