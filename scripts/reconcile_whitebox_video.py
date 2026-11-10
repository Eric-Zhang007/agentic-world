"""Review an already generated pending segment, reconcile, then plan continuation.

Independent entry point: does not alter the existing experiment runners.
"""
import argparse
import json
from pathlib import Path
import sys
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from whitebox.video_reconcile import BOUNDARY_PROMPT, build_correction, commit_video,video_alignment


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def align_action(action,fps=24,prefix_frames=1):
    """Extend settling time to H3's frame grid; do not resample accepted history."""
    action=deepcopy(action)
    if action.get('type')=='controls':
        count=len(action.get('inputs',[]))
        if not count or (count+prefix_frames)%17!=5:
            raise ValueError('Control recording must fit 17*k+5 video frames; record an aligned window without appending a stop')
        return action,{'extra_settle_ticks':0,'prefix_frames':prefix_frames,'video_samples':count+prefix_frames}
    counts=[round(p['duration_sec']*fps) for p in action['phases']]
    count=sum(counts)
    if (count+prefix_frames)%17==5:return action,{'extra_settle_ticks':0,'prefix_frames':prefix_frames}
    if any('end_velocity' in p or 'camera_end_motion' in p for p in action['phases']):
        raise ValueError('Kinematic action must fit the exact generation window: ticks + prefix_frames must equal 17*k+5; do not add a stop')
    target=count
    while (target+prefix_frames)%17!=5:target+=1
    if action['phases'][-1]['type']=='wait' and (counts[-1]+target-count)/fps<=8:
        action['phases'][-1]['duration_sec']=(counts[-1]+target-count)/fps
    else:
        while target-count<round(.5*fps):target+=17
        action['phases'].append({'type':'wait','duration_sec':(target-count)/fps,
                                'criterion':'Settle at the attained endpoint; do not return to the start.'})
    return action,{'extra_settle_ticks':target-count,'original_ticks':count,'aligned_ticks':target,
                   'video_samples':target+prefix_frames,'prefix_frames':prefix_frames,
                   'method':'append/extend settling, no trajectory time stretching'}


def prepare_plan(db,config,goal,output,reference=None,feedback=None,prefix_frames=1):
    from whitebox import director
    from whitebox.world import scene_from_db
    from whitebox.motion import simulate,validate_action
    output.mkdir(parents=True,exist_ok=False)
    scene=scene_from_db(db);revision=db.get_meta('whitebox_revision');start=db.get_meta('tick')
    pending=db.get_meta('whitebox_video_action')
    if pending and not pending['action_complete']:
        goal=pending['continuation_goal']
    from whitebox.appearance_rollout import retrieve_for_task
    memory=retrieve_for_task(db,config,output/'appearance_memory',{'goal':goal},reference)
    previous=None;problem={'pending_action':pending,'task_feedback':feedback,
                          'appearance_memory':memory['prompt']}
    for attempt in range(3):
        proposed=director.plan(config,scene,goal,feedback=problem,previous=previous,
                               images=([reference] if reference else [])+memory['images'])
        save(output/f'agent_plan_{attempt:02d}.json',proposed)
        try:
            validate_action(scene,proposed)
            action,alignment=align_action(proposed,prefix_frames=prefix_frames)
            timeline,endpoint,diagnostics=simulate(scene,action,start)
            if not diagnostics['valid']:raise ValueError(str(diagnostics['violations']))
            break
        except (ValueError,KeyError,TypeError) as exc:
            previous=proposed;problem={'error':str(exc),'prior_context':feedback,'appearance_memory':memory['prompt']}
    else:raise ValueError('Agent did not produce an executable continuation plan')
    if db.get_meta('whitebox_revision')!=revision or scene_from_db(db)!=scene:
        raise ValueError('World changed during planning')
    for name,data in [('scene.json',scene),('action.json',action),('timeline.json',timeline),
                      ('candidate_world.json',endpoint),('numerical_report.json',diagnostics),('alignment.json',alignment)]:
        save(output/name,data)
    (output/'render_prompt.txt').write_text(goal+'\n'+BOUNDARY_PROMPT,encoding='utf-8')
    save(output/'status.json',{'status':'PLANNED_NOT_COMMITTED','revision':revision,
                               'world_start':start,'world_unchanged':True,'reference':reference})
    return action


def main(a):
    import yaml
    from agent.action_planner import json_call
    from state.db import WorldDB
    from whitebox import director
    from whitebox.world import scene_from_db
    from whitebox.motion import simulate, validate_action

    if not a.world.is_file():
        raise FileNotFoundError('Use an initialized whitebox world, not an empty database')
    if a.prepare_goal:
        db = WorldDB(a.world)
        try:
            cfg = yaml.safe_load(a.config.read_text())['agent']
            reference = db.get_meta('whitebox_last_preview') or db.get_meta('whitebox_source_image')
            prepare_plan(db,cfg,a.prepare_goal,a.output,reference)
            print('PLANNED_NOT_COMMITTED: '+str(a.output))
            return
        finally:
            db.connection.close()
    import av
    from perception.frame_sampler import sample_frames
    before = json.loads((a.plan/'scene.json').read_text())
    planned = json.loads((a.plan/'timeline.json').read_text())
    action = json.loads((a.plan/'action.json').read_text())
    db = WorldDB(a.world)
    try:
        revision = db.get_meta('whitebox_revision')
        if scene_from_db(db) != before:
            raise ValueError('Video must correspond to a pending segment, not already committed history')
        expected, _, _ = simulate(before, action, db.get_meta('tick'), planned['fps'])
        if expected != planned:
            raise ValueError('Plan does not continue the current world')
        with av.open(str(a.video)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate or planned['fps'])
            frame_times=[float(frame.time) if frame.time is not None else i/fps
                         for i,frame in enumerate(container.decode(video=0))]
        alignment=video_alignment(planned,frame_times)
        duration=len(frame_times)/fps
        saved_review=None
        if getattr(a,'resume_review',False):
            if (a.output/'status.json').exists():
                raise ValueError('Resume is only for an interrupted, uncommitted review')
            if (json.loads((a.output/'planned_timeline.json').read_text())!=planned or
                json.loads((a.output/'alignment.json').read_text())!=alignment):
                raise ValueError('Saved review does not match this pending timeline and video alignment')
            saved_review=json.loads((a.output/'review.json').read_text())
        else:a.output.mkdir(parents=True, exist_ok=False)
        save(a.output/'alignment.json',alignment)
        motion_enabled=any('end_velocity' in p for p in action['phases']) or any('motion' in e for e in before['entities'])
        extra_times=([i/fps for i in (0,3,6,9,12)]+[frame_times[-1]-i/fps for i in (0,3,6,9,12)]) if motion_enabled else []
        frames = sample_frames(a.video, a.output/'frames', duration/16,extra_times=extra_times)
        prior_motion_frames=[]
        if motion_enabled and db.get_meta('whitebox_last_video'):
            prior_motion_frames=sample_frames(db.get_meta('whitebox_last_video'),a.output/'previous_motion',1/8)[-5:]
        contract_path = getattr(a, 'interaction_contract', None)
        contract = json.loads(Path(contract_path).read_text()) if contract_path else None
        camera_evidence_path = getattr(a, 'camera_evidence', None)
        camera_evidence = json.loads(Path(camera_evidence_path).read_text()) if camera_evidence_path else None
        config = yaml.safe_load(a.config.read_text())['agent']
        reference=db.get_meta('whitebox_last_preview') or db.get_meta('whitebox_source_image')
        if reference and not Path(reference).is_file():
            raise FileNotFoundError('Established visual boundary reference is missing: '+reference)
        from whitebox.appearance_rollout import retrieve_for_task
        appearance_memory=retrieve_for_task(db,config,a.output/'appearance_review',
            {'purpose':'Verify actual video against established identities and static landmarks',
             'action':action,'camera_end':planned['frames'][-1]['camera']},reference)
        preceding_count=(int(bool(reference))+len(frames)+len(prior_motion_frames)+
                         len(camera_evidence.get('candidate_preview_images',[]) if camera_evidence else []))
        appearance_review_refs=[dict(ref,picture=preceding_count+i+1)
                                for i,ref in enumerate(appearance_memory['refs'])]
        save(a.output/'appearance_review/review_references.json',appearance_review_refs)
        landmarks = [{'frame_index': i, 'time_sec': f['time']-db.get_meta('tick'),
                      'actors': [{'id': e['id'], 'position': e['position'],
                                  'rotation_y': e.get('rotation_y', 0.)}
                                 for e in f['entities'] if e['kind'] == 'actor']}
                     for i, f in enumerate(planned['frames'])
                     if i in {0, len(planned['frames'])//3, 2*len(planned['frames'])//3, len(planned['frames'])-1}]
        review = saved_review if saved_review is not None else json_call(config,
            'Review this generated segment and reconcile its coarse whitebox. You decide whether the video is '
            'a usable continuation; numerical agreement is not the sole acceptance criterion. Return JSON '
            '{decision:ACCEPT|REGENERATE,history_consistent:boolean,action_complete:boolean,evidence:[string],'
            'actor_assessments:[{id:string,status:CORRECTED|CONFIRMED_PLAN|UNOBSERVED_SIMULATED,evidence:[string]}],'
            'unsupported_state_changes:[string],'
            'continuation_goal:string,state_notes:[string],retry_prompt:string,corrections:[{id:string,'
            'evidence:[string],keyframes:[{frame_index:int,position_offset:[x,y,z],yaw_offset:number}]}],'
            'events:[{time_sec:number,entity_ids:[string],description:string,evidence:[string]}]}. '
            'Also return pose_corrections:[{id:string,evidence:[string],keyframes:[{frame_index:int,'
            'posture:standing|crouched|low_guard,grip:optional both|left|right}]}]. '
            'standing and crouched have two-foot support; low_guard is a low torso supported on feet and '
            'both hands, not an injury, defeat or unconsciousness. This coarse template cannot hold a staff. '
            'Pose corrections require an explicit final keyframe and evidence; use intermediate onset and '
            'completion keyframes to describe actual timing. CORRECTED assessments must cover actors with '
            'root corrections or pose corrections or both. Persistent supported posture is now representable. '
            'One/two-handed grip changes are not weapon ownership changes; record grip when useful but do not '
            'reject natural grip adjustment merely because a planner invented a stricter hand count. '
            'Coordinates are Godot Y-up, yaw in radians. Offsets are corrections to the PLANNED trajectory, '
            'not absolute positions or displacements from the start. Supply an endpoint keyframe for every '
            'corrected actor, and intermediate keyframes if the route differs. If the actor stayed in its '
            'starting place instead of the planned movement, cancel that movement with the offsets. '
            'These are uncertain single-view estimates, not measured 3D. Leave unseen NPC motion as simulated. '
            'Explicitly assess EVERY actor: CORRECTED with a correction track, CONFIRMED_PLAN only with '
            'visible evidence that its intended endpoint occurred, or UNOBSERVED_SIMULATED for unseen actors. '
            'Do not default visible actors to the planned endpoint when uncertain. Assess translation against '
            'the first video frame, using the supplied initial actor coordinates and camera projection; '
            'cancel planned displacement if the video returned to its initial position. No correction means '
            'a consciously confirmed plan, not missing evidence. Keyframe indices are PLAN indices, '
            'with the video-to-plan time mapping supplied in alignment. '
            'Accept natural intermediate acting, torso turns and grip adjustments. Judge causal history, '
            'identity and user intention; a useful partial action may be accepted and continued. Record what '
            'actually happened, not what the prompt requested. An accepted clip MUST be reconciled before '
            'the next plan. Never silently rewrite the previous accepted segment. No new entities or static '
            'edits in this limited tool. If acceptance requires changed weapon ownership, a dropped prop, '
            'joint pose beyond the provided posture vocabulary, or static geometry that this tool cannot represent, REGENERATE and '
            'list it in unsupported_state_changes rather than pretend it was stored. Small secondary motion needs '
            'no fine pose reconstruction. Record consequential events and uncertain residual facts in notes. '
            'When reference_first is true, image zero is the established prior endpoint (or initial source), '
            'NOT a frame of this new video. Compare the first new video frame against that reference: '
            'do not accept a new clip that silently changes established location, held objects, identity, '
            'camera or static scenery at its start. Minor rendering variation and secondary motion are allowed. '
            'Remaining frames are chronological samples; do not claim to have observed hidden intervals. '
            'When interaction_contract is supplied, also return interaction_assessment '
            '{status:SUPPORTED|CONTRADICTED|UNOBSERVED,evidence:[string]}. '
            'For a movement action, SUPPORTED means visible retained displacement into the target area, '
            'not a stance widening or a temporary step that returns. For a wait-only action starting at '
            'an already accepted target, SUPPORTED means retaining that attained location; do not demand '
            'another arrival or any new root travel. Target image coordinates refer to the '
            'original full video resolution, not resized message images. They are approximate projected '
            'coarse geometry, not independently measured landmarks. Cross-check stable floor features, '
            'the initial foot placement, camera stability, and the corrected root. If the target is not '
            'visibly attained, action_complete must be false. Useful partial displacement may be accepted '
            'with a continuation_goal that retains the SAME target; never reset the target relative to '
            'the newly reached position. Do not demand an exact foot pose or exact metric reconstruction. '
            'Camera reconciliation is available: camera_correction may be null or '
            '{evidence:[string],uncertainty:[string],keyframes:[{frame_index:int,position_offset:[x,y,z],'
            'target_offset:[x,y,z],fov_offset:number}]}. Offsets are against each PLANNED camera; '
            'fov_offset is in degrees, and an explicit final keyframe is mandatory. Start with zero offset '
            'at plan index 0 when the opening matches. Use only a coarse continuous camera explanation '
            'supported by multiple STATIC background features; these are uncertain estimates, not recovered '
            'metric extrinsics. Do not explain all image motion by moving actors or changing static geometry. '
            'A background affine fit is evidence of 2D flow only, not a unique 3D camera solution. '
            'If background evidence or actor-world alignment remains insufficient, REGENERATE with the '
            'remaining uncertainty. Camera freedom does not waive a specifically fixed-camera task; '
            'an explicit experiment_policy may relax that intention only for a new experimental branch. '
            'If candidate_preview_images are supplied, the LAST images after all video samples are '
            'hypothetical INITIAL/MIDDLE/FINAL whitebox views, not observations. If its candidate_review '
            'is usable, reuse the exact proposed camera_correction when accepting; if it is inadequate '
            'reject rather than silently invent an unpreviewed camera track. '
            'For every actor with motion in the planned endpoint, also return motion_assessments '
            '[{id:string,velocity:[vx,0,vz],phase:locomotion|stationary,evidence:[string],uncertainty:[string]}]. '
            'Estimate coarse horizontal endpoint velocity in world m/s using the chronological final window '
            'and static landmarks, not the requested velocity. A stopped actor has zero velocity. '
            'Phase locomotion requires nonzero velocity. Explicitly disclose uncertainty. No unsupported '
            'exact measurements. Also return boundary_motion {status:SUPPORTED|CONTRADICTED|UNOBSERVED,'
            'evidence:[string]} comparing incoming motion against prior_motion_images appended AFTER current '
            'video frames and BEFORE any candidate previews. Those prior images are older history, not future frames. '
            'A useful partial action that stopped can be accepted with stationary actual state and '
            'action_complete=false; never invent continuing motion to satisfy a test. '
            'After all other images, appearance_memory_images are historical object crops, not current '
            'video frames or hypothetical previews. Compare stable identity against these when visible; '
            'do not require their old pose, background or location. Do not demand an offscreen object appear. '
            'appearance_identity_anchors preserve earlier identity text independently of the image budget. '
            'Compare identity against these too, while honoring explicitly accepted state changes; '
            'text anchors do not establish an unobserved viewpoint.',
            {'goal': action.get('goal'), 'action': action, 'world_before': before,
             'landmarks': landmarks, 'planned_frames': len(planned['frames']),
             'alignment':alignment,
             'interaction_contract':contract,
             'camera_evidence':camera_evidence,
             'appearance_memory_images':appearance_review_refs,
             'appearance_identity_anchors':appearance_memory.get('identity_anchors',[]),
             'appearance_evidence_limits':{key:appearance_memory.get(key,[]) for key in
                 ('missing_requested_objects','missing_requested_views','selection_missing_evidence')},
             'reference_first':bool(reference),
             'sample_times': [f['timestamp'] for f in frames],
             'planned_endpoint_motion':{e['id']:e['motion'] for e in planned['frames'][-1]['entities'] if 'motion' in e},
             'prior_motion_images':[{'time':f['timestamp'],'path':f['path']} for f in prior_motion_frames],
             'recent_history': [{'planned_intention': h['action'], 'observed_or_simulated_events': h.get('events', []),
                                 'action_complete':h.get('action_complete'),
                                 'provenance':h.get('provenance','ACCEPTED_WHITEBOX_SIMULATION'),
                                 'interpretation':h.get('visual_review',{})}
                                for h in db.get_meta('whitebox_history', [])[-2:]]},
            ([reference] if reference else [])+[f['path'] for f in frames]+[f['path'] for f in prior_motion_frames]+
            (camera_evidence.get('candidate_preview_images',[]) if camera_evidence else [])+appearance_memory['images'])
        if saved_review is None:
            save(a.output/'review.json', review)
            save(a.output/'planned_timeline.json', planned)
        if review.get('decision') != 'ACCEPT':
            save(a.output/'status.json', {'status': 'REGENERATE', 'world_unchanged': True,
                                         'retry_prompt': review.get('retry_prompt')})
            print('REGENERATE: world unchanged; see '+str(a.output/'review.json'))
            return
        if camera_evidence and 'candidate_review' in camera_evidence:
            if (camera_evidence['candidate_review'].get('usable') is not True or
                    review.get('camera_correction') != camera_evidence['proposal'].get('camera_correction')):
                raise ValueError('Accepted camera must match the visually reviewed candidate')
        corrected, after, events, diagnostics = build_correction(before, planned, review)
        if contract:
            from whitebox.interaction import check_arrival_review
            save(a.output/'interaction_check.json', check_arrival_review(after, contract, review))
        save(a.output/'world_before.json',before)
        save(a.output/'history_before.json',db.get_meta('whitebox_history',[]))
        save(a.output/'corrected_timeline.json', corrected)
        save(a.output/'world_after.json', after)
        artifacts = {key: str(path.resolve()) for key, path in {
            'video': a.video, 'last_frame': Path(frames[-1]['path']),
            'planned_timeline': a.output/'planned_timeline.json',
            'corrected_timeline': a.output/'corrected_timeline.json', 'review': a.output/'review.json'}.items()}
        commit_video(db, before, planned, action, review, revision, artifacts)
        save(a.output/'history_after.json',db.get_meta('whitebox_history',[]))
        save(a.output/'status.json', {'status': 'COMMITTED', 'revision': revision+1,
                                     'action_complete': review['action_complete'], 'diagnostics': diagnostics})
        print('COMMITTED: corrected segment, revision '+str(revision+1))
        from whitebox.appearance_rollout import capture_accepted
        try:
            captured=capture_accepted(db,config,a.output/'appearance_capture')
        except Exception as exc:
            save(a.output/'status.json', {'status':'COMMITTED','revision':revision+1,
                'action_complete':review['action_complete'],'diagnostics':diagnostics,
                'memory_status':'FAILED','memory_error':str(exc)})
            print('World committed; memory extraction failed. Recover references before continuing.')
            return
        save(a.output/'status.json', {'status':'COMMITTED','revision':revision+1,
            'action_complete':review['action_complete'],'diagnostics':diagnostics,
            'memory_status':captured['status']})
        goal = a.next_goal if review['action_complete'] else review['continuation_goal']
        if goal and not getattr(a,'defer_next_plan',False):
            # This call happens ONLY after commit; planning reads the corrected DB.
            next_action = prepare_plan(db,config,goal,a.output/'next_plan',str(Path(frames[-1]['path']).resolve()),
                                       feedback={'accepted_events': events,'state_notes':review.get('state_notes',[])})
            save(a.output/'next_action.json', next_action)
            (a.output/'next_render_prompt.txt').write_text(goal+'\n'+BOUNDARY_PROMPT,encoding='utf-8')
    finally:
        db.connection.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--world', type=Path, required=True)
    p.add_argument('--plan', type=Path)
    p.add_argument('--video', type=Path)
    p.add_argument('--prepare-goal', help='Plan without committing; write boundary prompt and whitebox timeline')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--next-goal')
    p.add_argument('--interaction-contract', type=Path, help='Optional coarse arrival task JSON')
    p.add_argument('--camera-evidence', type=Path, help='Optional background evidence and explicit branch policy JSON')
    p.add_argument('--resume-review',action='store_true',help='Reuse the saved GPT review after an interrupted pre-commit step')
    p.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1]/'configs/default.yaml')
    args=p.parse_args()
    if args.prepare_goal and (args.plan or args.video or args.next_goal or args.resume_review):
        p.error('--prepare-goal cannot be combined with review arguments')
    if not args.prepare_goal and (not args.plan or not args.video):
        p.error('Review requires --plan and --video')
    main(args)
