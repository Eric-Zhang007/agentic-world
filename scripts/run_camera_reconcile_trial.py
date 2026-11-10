"""Reassess a rejected video in a fresh branch with explicit camera freedom."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.action_planner import json_call
from perception.frame_sampler import sample_frames
from scripts.inspect_background_motion import inspect
from scripts.reconcile_whitebox_video import save, main as review_video
from scripts.run_video_reconcile_trial import render_proxy
from state.db import WorldDB
from whitebox.camera_reconcile import correct_cameras
from whitebox.world import scene_from_db


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    with sqlite3.connect(f'file:{a.parent.resolve()}?mode=ro', uri=True) as source:
        with sqlite3.connect(a.output/'world.sqlite') as target:
            source.backup(target)
    db = WorldDB(a.output/'world.sqlite')
    before = scene_from_db(db)
    history = db.get_meta('whitebox_history', [])
    revision = db.get_meta('whitebox_revision')
    reference = db.get_meta('whitebox_last_preview')
    db.connection.close()
    save(a.output/'parent_history.json', history)
    save(a.output/'world_before.json', before)
    policy = {'scope':'NEW_EXPERIMENT_ONLY', 'parent':str(a.parent.resolve()),
              'source_video':str(a.video.resolve()),
              'original_rejection_preserved':True,
              'intention':'Retain attained actor locations and ownership; allow continuous modest camera motion.',
              'change_from_original':'Original hold requested fixed camera. This branch explicitly relaxes that requirement.',
              'not_a_claim':'A successful reassessment would not mean the original fixed-camera task succeeded.'}
    save(a.output/'branch.json', policy)
    planned = json.loads((a.plan/'timeline.json').read_text())
    if before != json.loads((a.plan/'scene.json').read_text()):
        raise ValueError('Source plan does not start from branch state')
    background = inspect(a.video, a.output/'background', json.loads(a.regions))
    samples = sample_frames(a.video, a.output/'frames', .65)
    config = yaml.safe_load(a.config.read_text())['agent']
    proposal = json_call(config,
        'Propose a COARSE camera explanation for a real video in an existing persistent whitebox. '
        'Images: previous accepted boundary, then chronological new video samples. '
        'Return {representable:boolean,evidence:[string],uncertainty:[string],camera_correction:null|'
        '{evidence:[string],uncertainty:[string],keyframes:[{frame_index:int,position_offset:[x,y,z],'
        'target_offset:[x,y,z],fov_offset:number}]}}. Offsets are relative to planned camera, Godot Y-up. '
        'Index 0 should have zero offsets when opening matches. Explicit final index required. '
        'Infer only a small smooth movement supported by static stairs, plinth, wall and pavement. '
        'Use scene coordinates to reason about right/up/forward directions. Do not move actors or scenery. '
        'Keep field of view unchanged unless visible evidence supports zoom. Single-view offsets are '
        'uncertain coarse estimates, not measured meters. 2D affine flow is not a unique 3D solution. '
        'Return representable=false if changes look like incompatible background morphing or you cannot '
        'offer a plausible same-world camera explanation. No accepted state is modified by this proposal.',
        {'experiment_policy':policy, 'world_before':before, 'background':background,
         'fps':planned['fps'], 'final_plan_index':len(planned['frames'])-1,
         'sample_times':[f['timestamp'] for f in samples],
         'planned_camera_samples':[{'index':i,'camera':planned['frames'][i]['camera']}
                                   for i in [0,len(planned['frames'])//2,len(planned['frames'])-1]]},
        [reference]+[f['path'] for f in samples])
    save(a.output/'proposal.json', proposal)
    if proposal.get('representable') is True and proposal.get('camera_correction'):
        candidate = deepcopy(planned)
        candidate['camera_correction'] = correct_cameras(before,candidate['frames'],candidate['fps'],proposal['camera_correction'])
        save(a.output/'candidate_timeline.json',candidate)
        render_proxy(a.plan,a.output/'candidate_preview',timeline=a.output/'candidate_timeline.json')
        proxy_frames = sorted((a.output/'candidate_preview').glob('frame_*.png'))
        proxy_images = [a.output/'candidate_preview/camera.png',proxy_frames[len(proxy_frames)//2],proxy_frames[-1]]
        check = json_call(config,
            'Check a proposed coarse camera explanation. First images are actual video samples; '
            'last three images are whitebox INITIAL/MIDDLE/FINAL with ONLY proposed camera changes. '
            'Return {usable:boolean,evidence:[string],issues:[string]}. Compare directions and relative '
            'changes of actors and background landmarks across time, not fine appearance. Do not demand '
            'pixel accuracy from coarse geometry, but reject incorrect movement direction, disocclusion, '
            'scale or actor-floor relation. Passing means a plausible uncertain coarse explanation only. '
            'Do not force a pass or compensate by changing established scenery.',
            {'world_before':before,'proposal':proposal,'actual_sample_times':[f['timestamp'] for f in samples]},
            [f['path'] for f in samples]+proxy_images)
        save(a.output/'candidate_review.json',check)
        evidence = {'experiment_policy':policy,'background':background,'proposal':proposal,
                    'candidate_review':check,'candidate_preview_images':[str(p.resolve()) for p in proxy_images]}
        save(a.output/'camera_evidence.json',evidence)
        if check.get('usable') is True:
            review_video(argparse.Namespace(world=a.output/'world.sqlite',plan=a.plan,video=a.video,
                output=a.output/'review',config=a.config,prepare_goal=None,resume_review=False,
                camera_evidence=a.output/'camera_evidence.json',interaction_contract=a.contract,
                next_goal='Hold the actual attained location for a short beat, retain the staff and tiger low guard. '
                          'Continue from the reconciled camera and actual final frame; allow natural secondary motion.'))
    db = WorldDB(a.output/'world.sqlite')
    try:
        now = db.get_meta('whitebox_history', [])
        assert now[:len(history)] == history
        original = WorldDB(a.parent)
        try:
            assert scene_from_db(original) == before
            assert original.get_meta('whitebox_history',[]) == history
        finally:
            original.connection.close()
    finally:
        save(a.output/'result.json',{'revision_before':revision,'revision_after':db.get_meta('whitebox_revision'),
             'history_prefix_preserved':now[:len(history)]==history,
             'committed':len(now)>len(history),
             'scope':'Real existing H3 video reassessed by GPT; no new H3 continuation rendered by this script.'})
        db.connection.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('parent','plan','video','output','contract'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/default.yaml'))
    p.add_argument('--regions',required=True)
    run(p.parse_args())
