"""Accept video-derived corrections before planning the next whitebox segment.

Corrections are agent estimates in the existing coordinate system, not measured
3D reconstruction. Unseen actors retain their simulated behavior.
"""
from copy import deepcopy
import math

import numpy as np

from whitebox.motion import rotation, simulate, validate_geometry,rig
from whitebox.posture import POSTURES,weights,contacts
from whitebox.world import put_entity, scene_from_db, validate_scene
from whitebox.camera_reconcile import correct_cameras


BOUNDARY_PROMPT = (
    "起始和结束状态应与给定条件一致，中间按动作描述自然丰富；收势是在目标终态稳定下来，不意味着回到起始位置。\n"
    "The supplied start and end conditions describe the SAME persistent world at two times. "
    "Match actor positions, facing, held objects and key poses at those boundaries. "
    "Between them, enrich the described action with natural anticipation, coordinated body motion, "
    "footwork, cloth motion and follow-through. Preserve the specified key events and their order. "
    "Finish at the requested END state. Recovery means settling there, NOT returning to the starting "
    "position. Secondary motion is free; extra hits, dropped props and knockdowns are new events."
)


def build_correction(before, planned, review):
    if review.get('decision') != 'ACCEPT' or review.get('history_consistent') is not True:
        raise ValueError('Video must be accepted as consistent with established history')
    if not review.get('evidence') or type(review.get('action_complete')) is not bool:
        raise ValueError('Review requires evidence and explicit action completion')
    if not review['action_complete'] and not review.get('continuation_goal'):
        raise ValueError('Accepted partial action needs a continuation goal')
    if review.get('unsupported_state_changes'):
        raise ValueError('Accepted video requires unsupported state changes')
    frames = deepcopy(planned['frames'])
    if not frames:
        raise ValueError('Empty planned timeline')
    ids = {e['id']: e for e in before['entities']}
    assessments = review.get('actor_assessments', [])
    actor_ids = {identity for identity,e in ids.items() if e['kind']=='actor'}
    if len(assessments)!=len(actor_ids) or {a.get('id') for a in assessments}!=actor_ids:
        raise ValueError('Every actor needs explicit observed or unobserved state assessment')
    for assessment in assessments:
        if assessment.get('status') not in {'CORRECTED','CONFIRMED_PLAN','UNOBSERVED_SIMULATED'} or not assessment.get('evidence'):
            raise ValueError('Actor assessment requires a supported status and evidence')
    times = np.asarray([f['time'] for f in frames], dtype=float)
    start = times[0] - 1 / planned['fps']
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Invalid timeline times')
    tracks = {}
    for correction in review.get('corrections', []):
        if set(correction)-{'id','keyframes','evidence'}:
            raise ValueError('Unsupported correction fields cannot be silently discarded')
        identity = correction['id']
        if identity not in ids or ids[identity]['kind'] != 'actor' or identity in tracks:
            raise ValueError('Corrections need unique existing actor IDs')
        keys = correction['keyframes']
        if not keys or not correction.get('evidence'):
            raise ValueError('Correction needs evidence and keyframes')
        previous = -1
        track = [(start, np.zeros(3), 0.)]
        for key in keys:
            if set(key)-{'frame_index','position_offset','yaw_offset'}:
                raise ValueError('Only root translation/yaw corrections are supported')
            index = key['frame_index']
            if type(index) is not int or not previous < index < len(frames):
                raise ValueError('Correction keyframes must be chronological planned-frame indices')
            offset = np.asarray(key['position_offset'], dtype=float)
            yaw = key.get('yaw_offset', 0.)
            if offset.shape != (3,) or not np.isfinite(offset).all() or not math.isfinite(yaw):
                raise ValueError('Nonfinite correction')
            if abs(yaw) > math.pi:
                raise ValueError('Yaw offset uses radians in [-pi, pi]')
            track.append((times[index], offset, yaw))
            previous = index
        if previous != len(frames)-1:
            raise ValueError('Every correction must explicitly describe the endpoint')
        tracks[identity] = track
    pose_tracks={}
    for correction in review.get('pose_corrections',[]):
        identity=correction['id']
        if (set(correction)-{'id','evidence','keyframes'} or identity not in actor_ids or
            identity in pose_tracks or not correction.get('evidence') or not correction.get('keyframes')):
            raise ValueError('Pose correction requires a unique actor, evidence and keyframes')
        previous=-1
        for key in correction['keyframes']:
            index=key['frame_index']
            if (set(key)-{'frame_index','posture','grip'} or type(index) is not int or
                not previous<index<len(frames) or key.get('posture') not in POSTURES or
                key.get('grip','both') not in {'both','left','right'}):
                raise ValueError('Invalid coarse pose keyframe')
            previous=index
        if previous!=len(frames)-1:raise ValueError('Pose correction requires explicit endpoint')
        pose_tracks[identity]=correction['keyframes']
    if set(tracks)|set(pose_tracks)!={a['id'] for a in assessments if a['status']=='CORRECTED'}:
        raise ValueError('Correction tracks must match explicit CORRECTED actor assessments')
    for frame in frames:
        if 'phase' in frame:frame['planned_phase']=frame.pop('phase')
        entities = {e['id']: e for e in frame['entities']}
        if set(entities) != set(ids):
            raise ValueError('Plan changes identity set')
        for identity, track in tracks.items():
            actor = entities[identity]
            key_times = [k[0] for k in track]
            offset = np.array([np.interp(frame['time'], key_times, [k[1][axis] for k in track])
                               for axis in range(3)])
            yaw = float(np.interp(frame['time'], key_times, np.unwrap([k[2] for k in track])))
            root = np.asarray(actor['position'])
            matrix = rotation(yaw)
            def transform(point):
                return (root + offset + matrix @ (np.asarray(point) - root)).tolist()
            actor['position'] = (root + offset).tolist()
            actor['rotation_y'] = actor.get('rotation_y', 0.) + yaw
            actor['pose'] = {joint: transform(point) for joint, point in actor.get('pose', {}).items()}
            for staff in entities.values():
                if staff['kind'] == 'staff' and staff.get('holder') == identity:
                    staff['position'] = transform(staff['position'])
                    staff['rotation_y'] = staff.get('rotation_y', 0.) + yaw
                    if 'weapon' in staff:
                        staff['weapon'] = {k: transform(v) for k, v in staff['weapon'].items()}
        for identity,keys in pose_tracks.items():
            actor=entities[identity]
            key_times=[start]+[times[k['frame_index']] for k in keys]
            actor['posture_weights']={name:float(np.interp(frame['time'],key_times,
                [weights(ids[identity]).get(name,0.)]+[float(k['posture']==name) for k in keys])) for name in POSTURES}
            actor['grip']=ids[identity].get('grip','both')
            for key in keys:
                if frame['time']>=times[key['frame_index']] and 'grip' in key:actor['grip']=key['grip']
            actor['support_contacts']=contacts(actor)
            staff=next((e for e in entities.values() if e.get('holder')==identity),None)
            pose,weapon,errors=rig(actor,frame['time'],0.,held=staff,terrain=before['terrain'],scene=before)
            if errors:raise ValueError('Unreachable corrected coarse pose: '+str(errors))
            actor['pose']=pose
            if staff:
                staff['weapon']=weapon;staff['position']=((np.array(weapon['a'])+weapon['b'])/2).tolist()
        for identity, old in ids.items():
            if old['kind'] not in {'actor', 'staff'} and entities[identity] != old:
                raise ValueError('Cannot rewrite static scene')
    camera_correction = correct_cameras(before, frames, planned['fps'], review.get('camera_correction'))
    events = deepcopy(review.get('events', []))
    for event in events:
        if (not event.get('description') or not event.get('evidence') or
            not isinstance(event.get('time_sec'), (int, float)) or
            not math.isfinite(event['time_sec']) or not -1e-8 <= event['time_sec'] <= times[-1]-start+1e-8 or
            not set(event.get('entity_ids', [])).issubset(ids)):
            raise ValueError('Invalid segment event')
        event['time_sec']=min(max(event['time_sec'],0.),float(times[-1]-start))
        event['world_time'] = start + event['time_sec']
        event['provenance'] = 'AGENT_ACCEPTED_VIDEO_INTERPRETATION'
    motion_ids={e['id'] for e in frames[-1]['entities'] if e['kind']=='actor' and 'motion' in e}
    motion_reviews=review.get('motion_assessments',[])
    reviewed_ids={m.get('id') for m in motion_reviews}
    if len(motion_reviews)!=len(reviewed_ids) or not reviewed_ids.issubset(actor_ids):
        raise ValueError('Motion assessments require unique existing actor IDs')
    if not motion_ids.issubset(reviewed_ids):
        raise ValueError('Missing video endpoint motion assessments: '+', '.join(sorted(motion_ids-reviewed_ids)))
    for assessment in motion_reviews:
        identity=assessment['id'];velocity=np.asarray(assessment.get('velocity'),float)
        if (velocity.shape!=(3,) or not np.isfinite(velocity).all() or abs(velocity[1])>1e-6
                or np.linalg.norm(velocity)>4 or assessment.get('phase') not in {'stationary','locomotion'}
                or not assessment.get('evidence') or not assessment.get('uncertainty')):
            raise ValueError('Motion assessment requires velocity, phase, evidence and uncertainty')
        if (np.linalg.norm(velocity)<1e-8)!=(assessment['phase']=='stationary'):
            raise ValueError('Endpoint motion phase contradicts velocity')
        # Corrected positions describe the coarse path; endpoint velocity is a separate
        # uncertain video interpretation, never silently inherited from the plan.
        previous=np.asarray(ids[identity]['position'],float)
        for frame in frames:
            actor=next(e for e in frame['entities'] if e['id']==identity)
            current=np.asarray(actor['position'],float);v=(current-previous)*planned['fps'];v[1]=0
            actor['motion']={'velocity':v.tolist(),'phase':'locomotion' if np.linalg.norm(v)>1e-8 else 'stationary',
                             'provenance':'DERIVED_FROM_COARSE_ACCEPTED_TRAJECTORY'}
            previous=current
        actor['motion']={k:deepcopy(assessment[k]) for k in ('velocity','phase','evidence','uncertainty')}
        actor['motion']['provenance']='AGENT_ACCEPTED_VIDEO_ESTIMATE'
    after = deepcopy(before)
    after['entities'] = deepcopy(frames[-1]['entities'])
    after['camera'] = deepcopy(frames[-1]['camera'])
    validate_scene(after)
    # Return the same executable-state diagnostics used before the next plan.
    diagnostics = validate_geometry(after)
    return {'fps': planned['fps'], 'frames': frames,'accepted_events':events,
            'camera_correction':camera_correction,
            'actor_assessments':deepcopy(assessments),
            'pose_scope':'Procedural posture anchors and rigid root corrections; not recovered video skeletons',
            'pose_corrections':deepcopy(review.get('pose_corrections',[])),
            'support_scope':'Template contact constraints, not measured contact forces or damage',
            'phase_scope':'planned_phase is prediction metadata, not an observed action phase'}, after, events, diagnostics


def commit_video(db, before, planned, action, review, revision, artifact_paths):
    with db.transaction():
        if not db.connection.in_transaction:
            db.connection.execute('BEGIN IMMEDIATE')
        if db.get_meta('whitebox_revision') != revision or scene_from_db(db) != before:
            raise ValueError('World changed since this segment was planned')
        expected, _, _ = simulate(before, action, db.get_meta('tick'), planned['fps'])
        if expected != planned:
            raise ValueError('Input is not the pending plan from the current world')
        corrected, after, events, diagnostics = build_correction(before, planned, review)
        if diagnostics:
            raise ValueError('Accepted endpoint is not executable in this world; revise the evidence-backed correction or regenerate: '+str(diagnostics))
        for entity in after['entities']:
            put_entity(db, entity, 'Agent accepted generated video; corrected this segment before continuing')
        db.set_meta('whitebox_header', {k: v for k, v in after.items() if k != 'entities'})
        start = db.get_meta('tick')
        db.set_meta('tick', corrected['frames'][-1]['time'])
        db.set_meta('whitebox_revision', revision+1)
        record = {'action': action, 'start': start, 'end': db.get_meta('tick'),
                  'action_semantics':'planned intention only; accepted_events and visual_review describe what occurred',
                  'action_complete':review['action_complete'],
                  'revision_before':revision,'revision_after':revision+1,
                  'actor_assessments':deepcopy(review['actor_assessments']),
                  'events': events, 'visual_review': review, 'artifacts': artifact_paths,
                  'provenance': 'AGENT_ACCEPTED_VIDEO_INTERPRETATION',
                  'geometry_diagnostics': diagnostics,
                  'cameras': [f['camera'] for f in corrected['frames']],
                  'actor_samples': [e['position'] for f in corrected['frames']
                                    for e in f['entities'] if e['kind'] == 'actor']}
        history = db.get_meta('whitebox_history', [])
        history.append(record)
        db.set_meta('whitebox_history', history)
        db.set_meta('whitebox_last_video', artifact_paths['video'])
        db.set_meta('whitebox_last_preview', artifact_paths['last_frame'])
        db.set_meta('whitebox_video_action', {
            'goal':action.get('goal'),
            'action_complete': review['action_complete'],
            'continuation_goal': review.get('continuation_goal'),
            'remaining_state_notes': review.get('state_notes', [])})
        return corrected, after, record


def video_alignment(planned, frame_times):
    """Map video samples to simulated state times; never silently stretch either."""
    fps=planned['fps'];frames=planned['frames']
    if not frames or not frame_times or not isinstance(fps,(int,float)) or fps<=0:
        raise ValueError('Nonempty video and positive plan fps required')
    start=frames[0]['time']-1/fps
    expected=np.asarray([f['time']-start for f in frames])
    actual=np.asarray(frame_times,dtype=float);actual-=actual[0]
    if not np.isfinite(actual).all() or np.any(np.diff(actual)<=0):
        raise ValueError('Invalid or repeated video timestamps')
    if len(actual)==len(expected)+1:
        target=np.r_[0,expected];indices=[-1]+list(range(len(expected)))
        convention='initial_state_then_simulated_ticks'
    elif len(actual)==len(expected):
        target=expected-expected[0];indices=list(range(len(expected)))
        convention='simulated_ticks_only_first_tick_at_video_zero'
    else:raise ValueError('Video/plan sample count mismatch; explicit alignment required')
    if not np.allclose(actual,target,rtol=0,atol=1e-4):
        raise ValueError('Video timestamps do not align with planned state samples')
    return {'convention':convention,'video_frame_to_plan_index':indices,
            'world_start':start,'world_end':frames[-1]['time'],
            'video_last_timestamp':float(actual[-1]),'plan_fps':fps}
