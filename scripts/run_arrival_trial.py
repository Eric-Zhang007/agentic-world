"""Branch accepted history and test target arrival then actual-state continuation."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sqlite3
import sys

import numpy as np
from PIL import Image, ImageDraw
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from state.db import WorldDB
from whitebox.world import scene_from_db
from whitebox.interaction import arrival_distance
from whitebox.video_reconcile import BOUNDARY_PROMPT
from scripts.evaluate_motion_ablation import project
from scripts.reconcile_whitebox_video import prepare_plan, main as review, save
from scripts.run_video_reconcile_trial import conditions, render_proxy, verify_chain
from renderer.comfy_h3 import ComfyH3Renderer
from agent.action_planner import json_call


def review_conditions(config, plan, reference, annotation, preview, contract):
    action = json.loads((plan/'action.json').read_text())
    initial = json.loads((plan/'scene.json').read_text())
    final = json.loads((plan/'candidate_world.json').read_text())
    moving = any(p['type'] == 'move' for p in action['phases'])
    return json_call(config,
        'Review WHITEBOX conditions, not observed outcomes. Return '
        '{usable:boolean,evidence:[string],issues:[string]}. Images: current accepted reference; '
        'target annotation on the ORIGINAL scene (its old player position is not the current state); '
        'initial proxy camera; final proxy camera; initial proxy side view; initial proxy overview. '
        'For an ARRIVAL action check readable root travel toward the target. For HOLD_AT_ACCEPTED_ENDPOINT '
        'check preserved placement at the already accepted target, not another translation. '
        'Debug side/overview views inspect weapon geometry and are not requested video camera changes. '
        'A weapon hidden/foreshortened in the main view need not be missing: check alternate views and '
        'the supplied ownership/attachment diagnostics. Coarse body and weapon shape need not match fine '
        'appearance. Natural grip adjustments are allowed. Offscreen NPC continuity is checked in the '
        'supplied simulated state; do not require it to appear in the camera view, and do not call its '
        'simulation observed evidence. Reject real geometry/ownership contradictions; do not require '
        'fine contact reconstruction or proof of future H3 performance from these proxy stills.',
        {'contract': contract, 'action': action,
         'phase_purpose': 'ARRIVAL' if moving else 'HOLD_AT_ACCEPTED_ENDPOINT',
         'initial_target_distance': arrival_distance(initial, contract),
         'final_target_distance': arrival_distance(final, contract),
         'numerical_report': json.loads((plan/'numerical_report.json').read_text()),
         'state_summaries': [[{k:e[k] for k in ('id','kind','position','holder','weapon','behavior') if k in e}
                               for e in state['entities'] if e['kind'] in ('actor','staff')]
                              for state in (initial, final)]},
        [reference, annotation, preview/'camera.png', sorted(preview.glob('frame_*.png'))[-1],
         preview/'side.png', preview/'overview.png'])


def resume_pending(a):
    root = a.output.resolve()
    segment = root/f'segment_{a.resume_segment:02d}'
    plan = segment/'plan'
    if (segment/'review').exists() or (segment/'render').exists():
        raise ValueError('This recovery only resumes a condition-rejected segment before generation')
    check_path = segment/'condition_review_retry.json'
    if check_path.exists():
        raise FileExistsError('Preserve the previous resumed review')
    db = WorldDB(root/'world.sqlite')
    try:
        before = scene_from_db(db)
        status = json.loads((plan/'status.json').read_text())
        if (before != json.loads((plan/'scene.json').read_text()) or
            status['revision'] != db.get_meta('whitebox_revision') or status['world_start'] != db.get_meta('tick')):
            raise ValueError('Pending segment does not start from current accepted state')
        parent_history = json.loads((root/'parent_history.json').read_text())
        assert db.get_meta('whitebox_history')[:len(parent_history)] == parent_history
        config = yaml.safe_load(a.config.read_text())
        contract = json.loads((root/'interaction_contract.json').read_text())
        reference = Path(status['reference'])
        check = review_conditions(config['agent'], plan, reference, root/'target_annotation.jpg',
                                  segment/'planned_preview', contract)
        save(check_path, check)
        if check.get('usable') is not True:
            raise ValueError('Agent rejected resumed whitebox preview; world unchanged')
        action = json.loads((plan/'action.json').read_text())
        count = len(json.loads((plan/'timeline.json').read_text())['frames'])+1
        prompt = ('Continuous realistic fantasy courtyard matching <Picture 1>. Same monkey warrior '
                  'holding the same staff; same alert low tiger guardian and stone paving. '
                  'Fixed camera. No strike, injury, defeat, new weapons or ground markers. '
                  +action['goal']+'\n'+BOUNDARY_PROMPT)
        (segment/'render_prompt.txt').write_text(prompt, encoding='utf-8')
        video = ComfyH3Renderer(gpu=a.gpu, steps=40, frames=count, width=1024, height=768,
                               kernel_backend='triton').render(prompt, None, segment/'render',
            first_frame=segment/'conditions/first.png', anchor_first=True,
            control_video=segment/'conditions/depth.npz', strength=a.control_strength, seed=a.seed+a.resume_segment)
        review(argparse.Namespace(world=root/'world.sqlite', prepare_goal=None, plan=plan, video=video,
            output=segment/'review', config=a.config, next_goal=action['goal'], resume_review=False,
            interaction_contract=root/'interaction_contract.json'))
        result = json.loads((segment/'review/status.json').read_text())
        if result['status'] == 'COMMITTED':
            render_proxy(plan, segment/'corrected_preview', timeline=segment/'review/corrected_timeline.json')
        save(root/'resumption_verification.json', {'segment': a.resume_segment, **result,
            'original_condition_review_preserved': True})
    finally:
        save(root/'world_current.json', scene_from_db(db))
        verify_chain(db, root)
        db.connection.close()


def main(a):
    if a.resume_segment is not None:
        resume_pending(a)
        return
    if not 3 <= a.duration <= 5.125:
        raise ValueError('Arrival trial duration must be between 3 and 5.125 seconds')
    if a.retry_from and a.duration != 3:
        raise ValueError('Same-plan retry preserves duration; use a new length trial instead')
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    with sqlite3.connect(a.parent.resolve().as_uri()+'?mode=ro', uri=True) as source:
        with sqlite3.connect(root/'world.sqlite') as target_db:
            source.backup(target_db)
    db = WorldDB(root/'world.sqlite')
    config = yaml.safe_load(a.config.read_text())
    baseline = deepcopy(db.get_meta('whitebox_history'))
    initial = scene_from_db(db)
    save(root/'parent_history.json', baseline)
    save(root/'world_initial.json', initial)
    save(root/'branch.json', {'parent': str(a.parent.resolve()), 'revision': db.get_meta('whitebox_revision'),
                             'world_time': db.get_meta('tick')})
    reference = Path(db.get_meta('whitebox_last_preview'))
    actor = next(e for e in initial['entities'] if e['id'] == 'player')
    forward = np.asarray(initial['camera']['target'])-initial['camera']['position']
    forward[1] = 0
    forward /= np.linalg.norm(forward)
    target = np.asarray(actor['position']) + np.cross(forward, [0, 1, 0])*.55
    contract = {'kind': 'arrival', 'actor_id': 'player', 'target': target.tolist(), 'radius': .18,
                'description': 'A small existing-floor region to screen-right; not a new object.',
                'coordinate_provenance': 'INFERRED_COARSE_WORLD', 'image_size': [1024, 768]}
    contract['target_image_center'] = project([target], initial['camera'], 1024, 768)[0].tolist()
    contract['initial_image_center'] = project([actor['position']], initial['camera'], 1024, 768)[0].tolist()
    save(root/'interaction_contract.json', contract)
    image = Image.open(reference).convert('RGB').resize((1024, 768))
    draw = ImageDraw.Draw(image)
    ring = [target + [.18*np.cos(t), 0, .18*np.sin(t)] for t in np.linspace(0, 2*np.pi, 40)]
    pixels = project(ring, initial['camera'], 1024, 768)
    draw.line([tuple(p) for p in pixels], fill='yellow', width=3)
    x, y = contract['target_image_center']
    draw.text((x+8, y-30), 'TARGET REGION (annotation only)', fill='yellow')
    image.save(root/'target_annotation.jpg')
    goal = ('Move player approximately 0.55 metres toward screen-right onto the adjacent existing paving '
            'region, then stay there. This is root travel, not merely widening the stance. '
            f'The fixed coarse target is {target.tolist()}, with horizontal tolerance 0.18m. '
            f'Use one move phase of {a.duration-1} seconds and a wait of 1 second. No attack or camera movement. '
            'Keep player_staff; natural grip and intermediate motion are free. Tiger remains alert in '
            'its established low_guard; do not reset its posture. No new props. Retain the new location '
            'at the end instead of returning. The target is an existing floor area, not a visible marker.')
    hold = (f'Hold the actual attained location inside the SAME target region {target.tolist()} for {a.duration} '
            'seconds, retaining the staff and NPC states. No attack, root travel or camera movement. '
            'Use a wait phase. Do not return to the older start position.')
    pending = None
    retry_prompt = None
    if a.retry_from:
        previous = a.retry_from.resolve()
        prior_contract = json.loads((previous/'interaction_contract.json').read_text())
        prior_status = json.loads((previous/'segment_00/review/status.json').read_text())
        if prior_contract != contract or prior_status['status'] != 'REGENERATE':
            raise ValueError('Retry must retain the same rejected target contract')
        if json.loads((previous/'segment_00/plan/scene.json').read_text()) != initial:
            raise ValueError('Retry cannot rewind or replace an accepted state')
        pending = previous/'segment_00/plan'
        retry_prompt = json.loads((previous/'segment_00/review/review.json').read_text())['retry_prompt']
        save(root/'retry_origin.json', {'rejected_trial': str(previous),
            'same_plan': True, 'control_strength': a.control_strength,
            'prompt_source': 'rejected_trial/segment_00/review/review.json:retry_prompt',
            'scope': 'Targeted retry; prompt and control strength change, not a single-factor ablation'})
    records = []
    try:
        for index in range(a.segments):
            segment = root/f'segment_{index:02d}'
            segment.mkdir()
            plan = segment/'plan'
            if pending:
                shutil.copytree(pending, plan)
            else:
                prepare_plan(db, config['agent'], goal, plan, str(reference))
            candidate = json.loads((plan/'candidate_world.json').read_text())
            if not arrival_distance(candidate, contract)['inside']:
                raise ValueError('Planned endpoint misses the fixed target region')
            render_proxy(plan, segment/'planned_preview')
            first, _, count = conditions(plan, reference, segment/'conditions', config, 'depth')
            if count > 124:
                raise ValueError('Trial requires short clips of at most 124 frames')
            check = review_conditions(config['agent'], plan, reference, root/'target_annotation.jpg',
                                      segment/'planned_preview', contract)
            save(segment/'condition_review.json', check)
            if check.get('usable') is not True:
                raise ValueError('Agent rejected the arrival whitebox preview')
            action = json.loads((plan/'action.json').read_text())
            prompt = ('Continuous realistic fantasy courtyard matching <Picture 1>. Same monkey warrior '
                      'holding the same staff; same alert low tiger guardian and stone paving. '
                      'Fixed camera. No strike, injury, defeat, new weapons or ground markers. '
                      +(retry_prompt if index == 0 and retry_prompt else action['goal'])+'\n'+BOUNDARY_PROMPT)
            (segment/'render_prompt.txt').write_text(prompt, encoding='utf-8')
            video = ComfyH3Renderer(gpu=a.gpu, steps=40, frames=count, width=1024, height=768,
                                   kernel_backend='triton').render(
                prompt, None, segment/'render', first_frame=first, anchor_first=True,
                control_video=segment/'conditions/depth.npz', strength=a.control_strength, seed=a.seed+index)
            review(argparse.Namespace(world=root/'world.sqlite', prepare_goal=None, plan=plan, video=video,
                output=segment/'review', config=a.config, next_goal=hold, resume_review=False,
                interaction_contract=root/'interaction_contract.json'))
            status = json.loads((segment/'review/status.json').read_text())
            records.append({'segment': index, **status})
            if status['status'] != 'COMMITTED':
                break
            render_proxy(plan, segment/'corrected_preview', timeline=segment/'review/corrected_timeline.json')
            verify_chain(db, root)
            pending = segment/'review/next_plan'
            reference = Path(db.get_meta('whitebox_last_preview'))
    finally:
        assert db.get_meta('whitebox_history')[:len(baseline)] == baseline
        save(root/'world_current.json', scene_from_db(db))
        save(root/'arrival_verification.json', {'segments': records, 'parent_history_unchanged': True,
             'revision': db.get_meta('whitebox_revision'), 'world_time': db.get_meta('tick'),
             'current_coarse_distance': arrival_distance(scene_from_db(db), contract)})
        verify_chain(db, root)
        db.connection.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--segments', type=int, default=2)
    p.add_argument('--seed', type=int, default=61)
    p.add_argument('--gpu', default='3')
    p.add_argument('--control-strength', type=float, default=.6)
    p.add_argument('--retry-from', type=Path, help='Reuse the same rejected target and pending plan with agent retry prompt')
    p.add_argument('--duration', type=float, default=3., help='Total planned seconds; 5.125 gives 124 native video frames')
    p.add_argument('--resume-segment', type=int, help='Resume a pre-generation condition rejection from its accepted world')
    p.add_argument('--config', type=Path, default=ROOT/'configs/default.yaml')
    main(p.parse_args())
