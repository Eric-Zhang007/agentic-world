"""Agent edits inferred primitives, renders, checks current and archived evidence."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent.action_planner import json_call
from scripts.audit_scene_alignment import audit
from scripts.reconcile_whitebox_video import save
from state.db import WorldDB
from whitebox.scene_refinement import apply_proposal,historical_geometry_checks,commit_refinement
from whitebox.world import scene_from_db


def cached_call(path, *args):
    if path.exists():
        return json.loads(path.read_text())
    result=json_call(*args)
    save(path,result)
    return result


def preview(scene, output):
    if output.exists():
        if (output/'views/camera.png').is_file() and json.loads((output/'scene.json').read_text())==scene:
            return output/'views/camera.png'
        raise ValueError('Existing preview is incomplete or belongs to another scene')
    output.mkdir(parents=True,exist_ok=False)
    save(output/'scene.json',scene)
    with (output/'render.log').open('w') as log:
        subprocess.run([sys.executable,str(ROOT/'whitebox/godot/render.py'),
            '--scene',str(output/'scene.json'),'--output',str(output/'views')],
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    return output/'views/camera.png'


def axes(camera):
    eye=np.asarray(camera['position']);forward=np.asarray(camera['target'])-eye
    forward=forward/np.linalg.norm(forward);right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right)
    return {'eye':eye.tolist(),'forward':forward.tolist(),'right':right.tolist(),
            'up':np.cross(right,forward).tolist(),
            'meaning':'point = eye + forward * depth + right * horizontal + up * vertical; fixed world Y is gravity'}


def run(a):
    a.output.mkdir(parents=True,exist_ok=a.resume)
    if not a.resume:
        with sqlite3.connect(f'file:{a.parent.resolve()}?mode=ro',uri=True) as source:
            with sqlite3.connect(a.output/'world.sqlite') as target:source.backup(target)
    elif not (a.output/'world.sqlite').is_file() or (a.output/'result.json').exists():
        raise ValueError('Resume requires an unfinished, uncommitted branch')
    db=WorldDB(a.output/'world.sqlite')
    try:
        before=scene_from_db(db);history=db.get_meta('whitebox_history',[])
        revision=db.get_meta('whitebox_revision');reference=db.get_meta('whitebox_last_preview')
        config=yaml.safe_load(a.config.read_text())['agent']
        if a.resume and (before!=json.loads((a.output/'world_before.json').read_text()) or
                         history!=json.loads((a.output/'history_before.json').read_text())):
            raise ValueError('Cannot resume refinement after world or history changed')
        save(a.output/'world_before.json',before);save(a.output/'history_before.json',history)
        save(a.output/'branch.json',{'parent':str(a.parent.resolve()),'revision':revision,
             'scope':'Explicit inferred representation revision; prior actions and archived geometry immutable'})
        original_preview=preview(before,a.output/'original')
        feedback=audit(config,before,reference,original_preview,a.output/'original_audit',resume=a.resume)
        previous=None;images=[reference,original_preview]
        accepted=False
        for index in range(a.attempts):
            folder=a.output/f'attempt_{index:02d}';folder.mkdir(exist_ok=a.resume)
            proposal=(json.loads(a.proposal.read_text()) if a.proposal and index==0 and not (folder/'proposal.json').exists()
                      else cached_call(folder/'proposal.json',config,
                'Repair an inferred coarse world representation using visual evidence. Return strict JSON '
                '{edits:[{id:string,position:optional [x,y,z],size:optional [x,y,z],rotation_y:optional number,'
                'evidence:[string]}],additions:optional [{entity:{id:string,kind:box|stairs,category:string,'
                'position:[x,y,z],size:[x,y,z],rotation_y:number,color:[r,g,b],provenance:INFERRED,'
                'reason:string,steps:optional int},evidence:[string]}],'
                'camera:optional {position:[x,y,z],target:[x,y,z],fov:number},'
                'evidence:[string],uncertainty:[string]}. Edit existing INFERRED box/stairs only. '
                'Preserve IDs, actor and held-object state, NPC policies, terrain and causal history. '
                'This is correcting our geometric interpretation, not making buildings move. '
                'If a clearly visible key static mass lacks an existing counterpart, explicitly propose '
                'an INFERRED addition with a unique ID and visual evidence. It remains a hypothesis '
                'until checked against current and historical images; never automatically add extras '
                'from one generated image, and do not duplicate a mass already modeled. '
                'Use camera basis and projected visibility, not the assumption that world +Z is image depth. '
                'A primitive can approximate a statue or stairs without details, but its screen position, '
                'relative scale and occlusion order must match. Box/stairs position is CENTER, actor position '
                'is FOOT ROOT. Dimensions and yaw refer to world axes; a facade may need yaw so its width '
                'lies across the camera right direction. y=size_y/2 puts a box base at ground. '
                'Keep geometry clear of current and prior actor paths. Do not remove inconvenient entities '
                'or invent fine meshes. Preserve unobserved geometry unless needed to resolve a contradiction '
                'with explicit uncertainty. Single-view geometry is underconstrained: disclose assumed depth. '
                'Camera and static layout may be jointly revised, but actor projection must remain consistent. '
                'Prefer retaining a camera that already fits the actor layout. '
                'Each proposal is an absolute patch against world_before, not incremental against the last '
                'candidate. Incorporate prior useful edits. Images: real boundary, original proxy; optionally '
                'the last candidate proxy. Use actual render feedback to repair the candidate.',
                {'world_before':before,'camera_basis':axes(before['camera']),
                 'feedback':feedback,'previous_candidate_patch':previous},images))
            save(folder/'proposal.json',proposal)
            previous=proposal
            try:
                candidate=apply_proposal(before,proposal)
                errors=historical_geometry_checks(candidate,history)
                if errors:raise ValueError('Historical actor path conflicts: '+str(errors[:12]))
            except (ValueError,KeyError,TypeError) as exc:
                feedback={'invalid_candidate':str(exc)};save(folder/'validation.json',feedback);continue
            image=preview(candidate,folder/'candidate')
            feedback=audit(config,candidate,reference,image,folder/'alignment',resume=a.resume)
            # Scene alignment is joint: preserve actors/ownership as well as static anchors.
            joint=cached_call(folder/'joint_revision_review.json',config,
                'Compare images: real boundary, ORIGINAL whitebox, candidate whitebox, candidate side view. '
                'All are the SAME world time. Return '
                '{consistent:boolean,evidence:[string],issues:[string],preexisting_limitations:[string]}. Judge actor left/right ordering, '
                'coarse projected height/root placement, support posture, held prop and relation to static '
                'geometry. Allow crude shapes and appearance differences; do not require mesh reconstruction. '
                'Do not accept a candidate that fixes architecture by losing actor alignment. '
                'Distinguish newly introduced regressions from preexisting procedural pose approximation. '
                'If a held prop is foreshortened in both original and candidate main views, use the side '
                'view and attachment metadata to verify retention; do not demand a specific angle or '
                'exact limb silhouette. Record preexisting limitations explicitly, do not pretend them '
                'fixed. Reject incorrect roots, support class, ownership or substantial new occlusion.',
                {'scene_before':before,'scene_after':candidate},
                [reference,original_preview,image,folder/'candidate/views/side.png'])
            images=[reference,original_preview,image]
            if feedback.get('usable_for_camera_alignment') is not True or joint.get('consistent') is not True:
                feedback={'alignment':feedback,'joint_review':joint};continue
            historical_images=[];descriptions=[]
            for hindex,record in enumerate(history):
                timeline=json.loads(Path(record['artifacts']['corrected_timeline']).read_text())
                endpoint=timeline['frames'][-1]
                state=deepcopy(candidate)
                dynamic={e['id']:e for e in endpoint['entities'] if e['kind'] in {'actor','staff'}}
                state['entities']=[deepcopy(dynamic.get(e['id'],e)) for e in state['entities']]
                state['camera']=deepcopy(endpoint['camera'])
                rendered=preview(state,folder/f'history_{hindex:02d}')
                historical_images += [record['artifacts']['last_frame'],rendered]
                descriptions.append({'history_index':hindex,'time':record['end']})
            past=cached_call(folder/'history_review.json',config,
                'Audit a new INFERRED STATIC GEOMETRY interpretation against archived accepted endpoints. '
                'Image pairs are actual historical boundary then re-rendered whitebox with unchanged '
                'archived actors and camera but revised static geometry. Return '
                '{consistent:boolean,evidence:[string],issues:[string]}. Check coarse static topology, '
                'visibility and actor-scenery relations, not textures or detailed shape. '
                'This is not rewriting history or claiming buildings moved. Reject incompatible past '
                'occlusion, blocked routes or changed identities. State evidence limits: endpoints do not '
                'prove every hidden interaction; all archived actor samples are separately collision checked.',
                {'boundaries':descriptions,'proposal':proposal},historical_images) if history else {
                    'consistent':True,'evidence':['No accepted segments before initialization']}
            save(folder/'history_review.json',past)
            if past.get('consistent') is not True:
                feedback={'alignment':feedback,'history':past};continue
            validation={'current':feedback,'history':past,'joint':joint}
            after=commit_refinement(db,before,proposal,revision,validation,{'experiment':str(folder.resolve())})
            save(a.output/'world_after.json',after);accepted=True;break
        assert db.get_meta('whitebox_history',[])==history
        parent=WorldDB(a.parent)
        try:
            assert scene_from_db(parent)==before and parent.get_meta('whitebox_history',[])==history
        finally:parent.connection.close()
        save(a.output/'result.json',{'committed':accepted,'revision_before':revision,
             'revision_after':db.get_meta('whitebox_revision'),'world_time':db.get_meta('tick'),
             'parent_unchanged':True,'historical_records_unchanged':True,
             'scope':'Coarse visual interpretation revision, not metric reconstruction or new video evidence'})
        print(json.dumps(json.loads((a.output/'result.json').read_text())))
    finally:db.connection.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/default.yaml')
    p.add_argument('--attempts',type=int,default=3)
    p.add_argument('--proposal',type=Path,help='Evaluate an existing candidate in a fresh experiment')
    p.add_argument('--resume',action='store_true',help='Resume an uncommitted branch using saved proposals and reviews')
    run(p.parse_args())
