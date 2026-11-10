"""Bounded real rollouts using current accepted state and generic user goals."""
import argparse
import json
from pathlib import Path
import sqlite3
import shutil
import sys

import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent.action_planner import json_call
from renderer.comfy_h3 import ComfyH3Renderer
from scripts.reconcile_whitebox_video import prepare_plan,main as review_video,save
from scripts.run_video_reconcile_trial import conditions,render_proxy
from state.db import WorldDB
from whitebox.world import scene_from_db
from agent.render_contract import compile_prompt
from renderer.continuation import window,historical_prefix,trim_generated
from whitebox.appearance_rollout import capture_accepted,retrieve


def load_prepared_plan(db,source,destination):
    from whitebox.motion import simulate
    scene=json.loads((source/'scene.json').read_text())
    action=json.loads((source/'action.json').read_text())
    timeline=json.loads((source/'timeline.json').read_text())
    if scene!=scene_from_db(db):raise ValueError('Prepared plan does not start at current accepted state')
    expected,endpoint,report=simulate(scene,action,db.get_meta('tick'),timeline['fps'])
    if expected!=timeline or not report['valid']:raise ValueError('Prepared plan replay failed')
    shutil.copytree(source,destination)
    return action


def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    with sqlite3.connect(f'file:{a.parent.resolve()}?mode=ro',uri=True) as source:
        with sqlite3.connect(a.output/'world.sqlite') as target:source.backup(target)
    db=WorldDB(a.output/'world.sqlite')
    records=[]
    baseline=db.get_meta('whitebox_history',[])
    source_scene=scene_from_db(db)
    try:
        config=yaml.safe_load(a.config.read_text())
        save(a.output/'world_before.json',source_scene)
        save(a.output/'history_before.json',baseline)
        save(a.output/'branch.json',{'parent':str(a.parent.resolve()),'goals':a.goal})
        capture_accepted(db,config['agent'],a.output/'initial_memory')
        goal_index=0
        inherited=db.get_meta('whitebox_video_action')
        finishing_inherited=bool(inherited and not inherited['action_complete'])
        for index in range(a.segments):
            segment=a.output/f'segment_{index:02d}';segment.mkdir()
            reference=Path(db.get_meta('whitebox_last_preview'))
            plan=segment/'plan'
            goal=a.goal[min(goal_index,len(a.goal)-1)]
            prefix_count=getattr(a,'prefix_frames',1) if index else 1
            schedule=window(prefix_frames=prefix_count) if prefix_count>1 else None
            future_ticks=schedule['new_ticks'] if schedule else 123
            prepared=getattr(a,'prepared_plan',None) if index==0 else None
            if prepared:
                action=load_prepared_plan(db,prepared,plan)
            else:action=prepare_plan(db,config['agent'],goal,plan,str(reference),
                feedback={'available_generation_window_sec':future_ticks/24,'exact_future_ticks':future_ticks,
                          'historical_prefix_frames':prefix_count,
                          'instruction':'Plan exactly the available window. Preserve the requested outcome and moving boundary. '
                                        'Use end_velocity for continuous locomotion; never add settling unless the user requests stopping. '
                                        'A long action must be partial; do not compress several consequential events.'},prefix_frames=prefix_count)
            timeline=json.loads((plan/'timeline.json').read_text());count=len(timeline['frames'])+1
            camera_requirement=getattr(a,'camera_excursion_deg',None)
            if camera_requirement is not None:
                from scripts.evaluate_camera_revisit import validate_camera_excursion
                save(segment/'camera_task_check.json',validate_camera_excursion(
                    scene_from_db(db),timeline,camera_requirement))
            if len(timeline['frames'])!=future_ticks:
                save(segment/'status.json',{'status':'PLAN_EXCEEDS_WINDOW','frames':count});break
            render_proxy(plan,segment/'planned_preview')
            proxy_frames=sorted((segment/'planned_preview').glob('frame_*.png'))
            preview_indices=sorted({round(i*(len(proxy_frames)-1)/8) for i in range(9)})
            check=json_call(config['agent'],
                'Review a planned coarse action before generating its video. Return '
                '{usable:boolean,evidence:[string],issues:[string]}. Images: actual accepted boundary, '
                'initial proxy, nine chronological proxy samples, side view. Verify initial actor and key '
                'static layout, readable requested movement, route clear of obstacles, preserved held '
                'objects and supported posture. NPCs can have simulated offscreen behavior. Do not '
                'demand fine anatomy, appearance, ornament or exact grips from procedural geometry. '
                'Use side view and holder metadata to assess a foreshortened weapon rather than '
                'requiring its source-image angle. Judge support class rather than fine limb silhouette. '
                'Conditions are plans, never proof that the video outcome occurred. If the user requires '
                'visible whole-body locomotion and the head/body/feet become cropped, return usable=false; '
                'a numerical route without an observable outcome is not usable for that request.',
                {'action':action,'world_before':scene_from_db(db),
                 'proxy_sample_times':[timeline['frames'][i]['time'] for i in preview_indices],
                 'diagnostic_semantics':'proxy_contacts are sampled WEAPON contacts with actors, not walking foot contacts. '
                     'No proxy_contact during walking is expected. Coarse gait conveys movement; natural fine footwork is renderer freedom. '
                     'A foreshortened staff is not absent: cross-check side view and exact attached weapon endpoints.',
                 'diagnostics':json.loads((plan/'numerical_report.json').read_text())},
                [reference,segment/'planned_preview/camera.png',
                 *[proxy_frames[i] for i in preview_indices],segment/'planned_preview/side.png'])
            save(segment/'condition_review.json',check)
            if check.get('usable') is not True:
                records.append({'segment':index,'status':'CONDITIONS_REJECTED'});break
            from whitebox.appearance_rollout import retrieve_for_task
            memory=retrieve_for_task(db,config['agent'],segment/'appearance_memory',
                                     {'goal':goal,'action':action,'camera_end':timeline['frames'][-1]['camera']},reference)
            prompt=compile_prompt(config['agent'],scene_from_db(db),action,timeline,reference,segment,prefix_count,memory)
            render_refs=json.loads((segment/'renderer_references.json').read_text())
            prefix=historical_prefix(db,segment/'history_prefix',prefix_count) if schedule else None
            first,_,count=conditions(plan,reference,segment/'conditions',config,'depth',
                                     history_prefix=segment/'history_prefix' if prefix else None)
            video=ComfyH3Renderer(gpu=a.gpu,steps=40,frames=count,width=1024,height=768,
                kernel_backend='triton').render(prompt,None,segment/'render',first_frame=first,
                anchor_first=True,control_video=segment/'conditions/depth.npz',strength=1.,seed=a.seed+index,
                prefix_video=prefix,appearance_images=render_refs['images'])
            if schedule:video=trim_generated(video,segment/'continuation.mp4',schedule)
            if camera_requirement is not None or any('camera_orbit_yaw_deg' in phase for phase in action['phases']):
                from scripts.evaluate_camera_revisit import evaluate
                camera_review=evaluate(argparse.Namespace(plan=plan,preview=segment/'planned_preview',
                    video=video,reference=reference,output=segment/'camera_review',config=a.config))
                if camera_review.get('overall')!='PASS':
                    records.append({'segment':index,'status':'CAMERA_TEST_NOT_PASSED',
                                    'camera_review':camera_review})
                    break
            review_video(argparse.Namespace(world=a.output/'world.sqlite',plan=plan,video=video,
                output=segment/'review',config=a.config,prepare_goal=None,resume_review=False,
                next_goal=None,interaction_contract=None,camera_evidence=None,defer_next_plan=True))
            status=json.loads((segment/'review/status.json').read_text());records.append({'segment':index,**status})
            if status['status']!='COMMITTED':break
            if status.get('memory_status')=='FAILED':break
            render_proxy(plan,segment/'corrected_preview',timeline=segment/'review/corrected_timeline.json')
            if status['action_complete']:
                if finishing_inherited:finishing_inherited=False
                else:goal_index+=1
            if goal_index>=len(a.goal):break
    finally:
        assert db.get_meta('whitebox_history',[])[:len(baseline)]==baseline
        save(a.output/'world_after.json',scene_from_db(db))
        save(a.output/'result.json',{'segments':records,'revision':db.get_meta('whitebox_revision'),
             'world_time':db.get_meta('tick'),'prior_history_preserved':True})
        db.connection.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--goal',action='append',required=True);p.add_argument('--segments',type=int,default=2)
    p.add_argument('--gpu',default='3');p.add_argument('--seed',type=int,default=73)
    p.add_argument('--prefix-frames',type=int,default=1,help='Accepted historical guide frames in later segments, e.g. 22')
    p.add_argument('--camera-excursion-deg',type=float,help='Required planned yaw excursion for a camera test; missing/tiny camera motion fails before generation')
    p.add_argument('--prepared-plan',type=Path,help='Reuse an uncommitted plan after exact state and deterministic replay validation')
    p.add_argument('--config',type=Path,default=ROOT/'configs/default.yaml')
    run(p.parse_args())
