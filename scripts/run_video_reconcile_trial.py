"""Fresh H3 -> agent acceptance -> corrected DB -> next H3 diagnostic loop."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image,ImageDraw
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from state.db import WorldDB
from whitebox.world import initialize,scene_from_db
from whitebox.video_reconcile import BOUNDARY_PROMPT
from scripts.reconcile_whitebox_video import prepare_plan,main as review_main,save
from scripts.evaluate_motion_ablation import project,expected_actors,assign
from perception.detector import PerceptionPipeline
from renderer.comfy_h3 import ComfyH3Renderer
from agent.action_planner import json_call


def render_proxy(plan,output,mode='preview',timeline=None,width=960,height=640):
    command=[str(ROOT/'.venv/bin/python'),str(ROOT/'whitebox/godot/render.py'),
             '--scene',str(plan/'scene.json'),'--timeline',str(timeline or plan/'timeline.json'),
             '--output',str(output),'--render-pass',mode,'--frames','all',
             '--width',str(width),'--height',str(height)]
    with (output.parent/(output.name+'.log')).open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)


def read_depth_frames(files, width=1024, height=768):
    frames = []
    for path in files:
        with Image.open(path) as image:
            if image.size != (width, height):
                raise ValueError(f'Depth must be rendered at the target canvas, not stretched: {path}: {image.size}')
            frames.append(np.asarray(image.convert('RGB')))
    return np.stack(frames)


def conditions(plan,reference,output,config,control,history_prefix=None):
    """Create boundary target via explicit actor-mask translation, not observation."""
    output.mkdir(exist_ok=False)
    before=json.loads((plan/'scene.json').read_text())
    timeline=json.loads((plan/'timeline.json').read_text())
    last=timeline['frames'][-1]
    render_proxy(plan,output/'depth','depth',width=1024,height=768)
    files=[output/'depth/camera.png']+sorted((output/'depth').glob('frame_*.png'))
    frames=read_depth_frames(files)
    assert len(frames)==len(timeline['frames'])+1
    if history_prefix is not None:
        if control!='depth':raise ValueError('Historical prefix currently requires depth conditions')
        render_proxy(history_prefix,output/'prefix_depth','depth',width=1024,height=768)
        prefix_files=[output/'prefix_depth/camera.png']+sorted((output/'prefix_depth').glob('frame_*.png'))
        prefix=read_depth_frames(prefix_files)
        if not np.array_equal(prefix[-1],frames[0]):raise ValueError('Historical/current boundary depth mismatch')
        frames=np.concatenate([prefix,frames[1:]])
    assert len(frames)%17==5
    np.savez_compressed(output/'depth.npz',rgb=frames)
    image=Image.open(reference).convert('RGB').resize((1024,768));image.save(output/'first.png')
    if control=='depth':
        save(output/'target_metadata.json',{'kind':'NATIVE_FULL_DEPTH_CONDITION_WITH_BOUNDARY_STATES',
                                           'native_rgb_guide_indices':[0],'depth_frames':len(frames),
                                           'terminal_rgb_guide':None,'no_image_compositing':True})
        return output/'first.png',None,len(frames)
    detector=PerceptionPipeline({**config['perception'],'device':'cuda:0',
                                'categories':['monkey warrior','tiger warrior','staff']})
    frame={'path':str(output/'first.png'),'frame_id':0,'timestamp':0.}
    obs=detector.detect_frame(frame,output/'perception','boundary_target',source='CONDITION_INPUT')
    expected=expected_actors(before,1024,768)
    # GroundingDINO gives several conflicting labels for the same fantasy actor.
    # Match deduplicated mask proposals by the already fitted projection here;
    # the image review below checks identities before these become guide images.
    proposals=deepcopy(obs)
    for proposal in proposals:proposal.category='actor'
    pairs,missing=assign(expected,proposals,1024,768)
    if missing:raise ValueError('Cannot build explicit endpoint target; missing actor masks: '+str(missing))
    by_id={e['id']:e for e in before['entities']};after={e['id']:e for e in last['entities']}
    array=np.asarray(image);union=np.zeros(array.shape[:2],np.uint8);layers=[];records=[]
    for pair in pairs:
        identity=pair['world_id'];observation=next(o for o in obs if o.obs_id==pair['obs_id'])
        mask=cv2.imread(str(observation.mask),cv2.IMREAD_GRAYSCALE)>0
        if identity=='player':
            for o in obs:
                if o.category and 'staff' in o.category and (o.bbox_xyxy[0]+o.bbox_xyxy[2])/2<512:
                    mask|=cv2.imread(str(o.mask),cv2.IMREAD_GRAYSCALE)>0
            # Preserve the whole thin weapon, including holes missed by SAM.
            # This guide-only hull deliberately includes some background pixels.
            hull=cv2.convexHull(cv2.findNonZero(mask.astype(np.uint8)))
            filled=np.zeros_like(mask,dtype=np.uint8)
            cv2.fillConvexPoly(filled,hull,255)
            mask=cv2.dilate(filled,np.ones((5,5),np.uint8))>0
        Image.fromarray(mask.astype(np.uint8)*255).save(output/(identity+'_edit_mask.png'))
        start=project([by_id[identity]['position']],before['camera'],1024,768)[0]
        end=project([after[identity]['position']],last['camera'],1024,768)[0]
        delta=end-start
        if np.linalg.norm(delta)>180:raise ValueError('Endpoint image edit exceeds small-translation scope')
        if np.linalg.norm(delta)<1:continue
        union[mask]=255
        transform=np.float32([[1,0,delta[0]],[0,1,delta[1]]])
        moved=cv2.warpAffine(array,transform,(1024,768))
        moved_mask=cv2.warpAffine(mask.astype(np.uint8),transform,(1024,768),flags=cv2.INTER_NEAREST)>0
        layers.append((moved,moved_mask))
        records.append({'world_id':identity,'translation_px':delta.tolist(),'mask_obs_id':observation.obs_id})
    target=cv2.inpaint(array,cv2.dilate(union,np.ones((3,3),np.uint8)),5,cv2.INPAINT_TELEA) if union.any() else array.copy()
    for moved,mask in reversed(layers):target[mask]=moved[mask]
    Image.fromarray(target).save(output/'last_target.png')
    save(output/'target_metadata.json',{'kind':'EDITED_TARGET_CONDITION_NOT_OBSERVATION','operations':records,
                                      'limitations':['2D mask translation/inpaint; no new-view reconstruction; player/weapon convex hull carries local background',
                                                     'Pose and appearance inherited from accepted reference; minor guide artifacts possible'],
                                      'native_guides':[0,len(frames)-1]})
    return output/'first.png',output/'last_target.png',len(frames)


def verify_chain(db,root):
    history=db.get_meta('whitebox_history',[])
    records=[]
    for folder in sorted(root.glob('segment_*')):
        review=folder/'review'
        if not (review/'status.json').exists():continue
        status=json.loads((review/'status.json').read_text())
        if status['status']!='COMMITTED':continue
        before=json.loads((review/'history_before.json').read_text())
        after=json.loads((review/'history_after.json').read_text())
        assert after[:-1]==before and history[:len(after)]==after
        actual=json.loads((review/'world_after.json').read_text())
        planned=json.loads((folder/'plan/candidate_world.json').read_text())
        delta={e['id']:(np.array(e['position'])-next(p['position'] for p in planned['entities'] if p['id']==e['id'])).tolist()
               for e in actual['entities'] if e['kind']=='actor'}
        next_plan=review/'next_plan/scene.json'
        if next_plan.exists():
            assert json.loads(next_plan.read_text())==actual
            next_status=json.loads((next_plan.parent/'status.json').read_text())
            assert Path(next_status['reference'])==Path(after[-1]['artifacts']['last_frame'])
            assert next_status['world_start']==after[-1]['end']
        records.append({'segment':folder.name,'revision':status['revision'],
                        'action_complete':status['action_complete'],'actual_minus_planned_endpoint':delta,
                        'next_plan_starts_from_actual':next_plan.exists(),
                        'next_visual_reference_is_accepted_final_frame':next_plan.exists(),
                        'older_history_unchanged':True})
    if records:
        assert scene_from_db(db)==actual
        assert len(history)==len(after)
        assert db.get_meta('tick')==after[-1]['end']
        assert db.get_meta('whitebox_revision')==after[-1]['revision_after']
    save(root/'chain_verification.json',{'accepted_segments':len(records),'history_entries':len(history),
                                        'revision':db.get_meta('whitebox_revision'),'world_time':db.get_meta('tick'),
                                        'current_world_matches_last_accepted_endpoint':bool(records),
                                        'segments':records,'scope':'Agent-estimated video reconciliation, not measured 3D ground truth'})


def main(args):
    config=yaml.safe_load(args.config.read_text())
    root=args.output.resolve()
    if args.continue_trial:
        if not (root/'world.sqlite').is_file():raise FileNotFoundError('Existing trial world required')
    else:root.mkdir(parents=True,exist_ok=False)
    scene=json.loads(args.scene.read_text())
    # This is an explicitly new experimental world, never a rewind of the live DB.
    db=WorldDB(root/'world.sqlite')
    if args.continue_trial:
        if json.loads((root/'world_initial.json').read_text())!=scene:
            raise ValueError('Continuation must name this trial original scene')
        verify_chain(db,root)
    else:
        initialize(db,scene);db.set_meta('whitebox_source_image',str(args.reference.resolve()))
        db.set_meta('trial_origin',{'scene':str(args.scene.resolve()),'new_world':True,'prior_history_not_imported':True})
        save(root/'world_initial.json',scene)
    first=root/'segment_00';first.mkdir(exist_ok=args.continue_trial)
    goal=args.goal or ('Make one small sideways step of approximately 0.30 metres away from the tiger, without an attack. '
          'Keep the staff and settle at the new position. Plan one move of about 1.5 seconds followed by '
          'about 1.5 seconds of holding; no camera movement. Natural intermediate body motion is allowed. '
          'If a generated segment completes only part of the step, accept useful causal progress and continue '
          'from its observed position; do not assume the intended endpoint occurred.')
    try:
        if args.continue_trial:
            if args.initial_plan:raise ValueError('Cannot replace initial plan while continuing')
        elif args.initial_plan:
            if json.loads((args.initial_plan/'scene.json').read_text())!=scene:
                raise ValueError('Imported pending plan does not start from this fresh world')
            shutil.copytree(args.initial_plan,first/'plan')
        else:prepare_plan(db,config['agent'],goal,first/'plan',str(args.reference.resolve()))
        reference=args.reference.resolve();pending=first/'plan'
        for index in range(args.segments):
            segment=root/f'segment_{index:02d}'
            if args.continue_trial and (segment/'review/status.json').exists():
                status=json.loads((segment/'review/status.json').read_text())
                if status['status']!='COMMITTED':raise ValueError('Rejected segment requires a new trial, not overwrite')
                if not (segment/'corrected_preview').exists():
                    render_proxy(segment/'plan',segment/'corrected_preview',timeline=segment/'review/corrected_timeline.json')
                pending=segment/'review/next_plan'
                reference=Path(json.loads((pending/'status.json').read_text())['reference'])
                continue
            if index:
                segment.mkdir();shutil.copytree(pending,segment/'plan')
            plan=segment/'plan'
            first_frame,last_frame,frames=conditions(plan,reference,segment/'conditions',config,args.control)
            boundary_review=json_call(config['agent'],
                'Review generation conditions, NOT video observations or accepted events. '
                'Return {usable:boolean,issues:[string],evidence:[string]}. Images: start appearance, '
                'optional edited target appearance (only when terminal_rgb is true), coarse start depth, coarse end depth. Accept small inpaint/edge artifacts '
                'if the same two characters, staff and scene remain recognizable. Reject missing/duplicated '
                'actors, lost weapon, a gross scene change or an obviously impossible endpoint. '
                'Depth uses deliberately coarse mannequins; do not require texture agreement.',
                {'action':json.loads((plan/'action.json').read_text()),'kind':'target condition only','terminal_rgb':bool(last_frame)},
                [p for p in [first_frame,last_frame] if p]+[segment/'conditions/depth/camera.png',
                 sorted((segment/'conditions/depth').glob('frame_*.png'))[-1]])
            save(segment/'condition_review.json',boundary_review)
            if boundary_review.get('usable') is not True:raise ValueError('Agent rejected the endpoint condition; no generation')
            prompt=('A continuous realistic fantasy game scene matching <Picture 1>. Keep the monkey warrior '
                    'on the near left and tiger guardian on the right, the same staff, armor and stone courtyard. '
                    'Fixed camera, no cut or zoom. Follow this action: '+
                    json.loads((plan/'action.json').read_text())['goal']+'\n'+BOUNDARY_PROMPT)
            (segment/'render_prompt.txt').write_text(prompt,encoding='utf-8')
            video=ComfyH3Renderer(gpu=args.gpu,steps=args.steps,frames=frames,width=1024,height=768,
                                  kernel_backend='triton').render(prompt,None,segment/'render',
                                  first_frame=first_frame,last_frame=last_frame,anchor_first=True,
                                  control_video=segment/'conditions/depth.npz',strength=.6,seed=43+index,
                                  cache_threshold=0)
            review_main(argparse.Namespace(world=root/'world.sqlite',prepare_goal=None,plan=plan,video=video,
                        output=segment/'review',config=args.config,
                        next_goal=args.next_goal))
            status=json.loads((segment/'review/status.json').read_text())
            if status['status']!='COMMITTED':
                save(root/'status.json',{'status':'REGENERATE','segment':index,'history_committed':db.get_meta('whitebox_revision')})
                break
            # Render accepted corrected trajectory for visual acceptance, not the original plan.
            render_proxy(plan,segment/'corrected_preview',timeline=segment/'review/corrected_timeline.json')
            verify_chain(db,root)
            reference=Path(db.get_meta('whitebox_last_preview'))
            pending=segment/'review/next_plan'
        else:save(root/'status.json',{'status':'TWO_SEGMENTS_COMMITTED','revision':db.get_meta('whitebox_revision')})
        save(root/'world_current.json',scene_from_db(db));verify_chain(db,root)
    finally:db.connection.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/default.yaml')
    p.add_argument('--gpu',default='3');p.add_argument('--steps',type=int,default=40)
    p.add_argument('--segments',type=int,default=2)
    p.add_argument('--initial-plan',type=Path)
    p.add_argument('--continue-trial',action='store_true',help='Continue after committed segments without changing their history or artifacts')
    p.add_argument('--goal',help='Initial action intention for a fresh trial')
    p.add_argument('--next-goal',default='Remain at the actual attained position and watch the opponent for about three seconds; keep the staff, no attack and no camera move.')
    p.add_argument('--control',choices=('depth','edited_rgb_target'),default='depth')
    main(p.parse_args())
