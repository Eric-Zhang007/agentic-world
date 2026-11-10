"""Create, preview, review and maintain a persistent agent-authored whitebox."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from state.db import WorldDB
from whitebox import director
from whitebox.world import initialize,scene_from_db,validate_scene,terrain_height,reconcile
from whitebox.motion import simulate,commit,validate_geometry,validate_action,prepare_scene
from whitebox.appearance import AppearanceMemory


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def render(scene_path,timeline_path,output,mp4=False):
    command=[sys.executable,str(ROOT/'whitebox/godot/render.py'),'--scene',str(scene_path),
             '--timeline',str(timeline_path),'--output',str(output)]
    if mp4:command.append('--mp4')
    subprocess.run(command,check=True,timeout=360)


def preview_images(output):
    # Prefer multiview sheets plus evenly sampled chronological frames.
    files=sorted(output.glob('*.png'))
    views=[p for p in files if not p.name.startswith('frame_')]
    frames=sorted(output.glob('frame_*.png'))
    if len(frames)>9:
        frames=[frames[round(i*(len(frames)-1)/8)] for i in range(9)]
    return views[:3]+frames


def npc_views(scene,timeline,folder):
    paths=[]
    for npc in scene['entities']:
        if npc['kind']!='actor' or npc['behavior']['type']!='patrol':continue
        copied=deepcopy(timeline);x,y,z=npc['position']
        camera={'position':[x+3,y+2.5,z-4],'target':[x,y+.8,z]}
        for frame in copied['frames']:frame['camera']=camera
        target=folder/('npc_'+npc['id']);target.mkdir()
        save(target/'timeline.json',copied)
        render(folder/'scene.json',target/'timeline.json',target/'preview')
        images=sorted((target/'preview').glob('frame_*.png'))
        paths.extend([images[0],images[len(images)//2],images[-1]])
    return paths


def export_memory(db,path):
    memory=AppearanceMemory(db)
    bundle=memory.prompt_and_images([{'object_id':o.object_id} for o in db.query_world()])
    folder=path/'appearance';folder.mkdir(exist_ok=True)
    images=[]
    for i,image in enumerate(bundle['images']):
        target=folder/f'reference_{i:03d}.png';image.save(target);images.append(str(target))
    save(folder/'request.json',{'prompt':bundle['prompt'],'images':images,'refs':bundle['refs'],
        'scope':'Prepared reference+prompt payload; H3 not invoked by whitebox CLI'})


def run(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    db=WorldDB(out/'world.sqlite')
    try:
        if a.command=='inspect':
            print(json.dumps({'scene':scene_from_db(db),'time':db.get_meta('tick'),
                              'revision':db.get_meta('whitebox_revision'),'history':db.get_meta('whitebox_history')},indent=2));return
        if a.command=='reconcile':
            reconcile(db,json.loads(a.edits.read_text()),a.reason)
            save(out/'world_current.json',scene_from_db(db));print('RECONCILED');return
        cfg=yaml.safe_load(a.config.read_text())['agent']
        creating=a.command=='create'
        if creating:
            if db.query_world():raise ValueError('Use act to continue an existing world')
            if not a.image:raise ValueError('create requires --image')
            source=a.image.resolve()
        else:
            source=Path(db.get_meta('whitebox_source_image'))
        serial=len(list(out.glob('attempt_*')));previous=None;feedback=None;images=[]
        for retry in range(a.attempts):
            folder=out/f'attempt_{serial+retry:03d}';folder.mkdir()
            print(json.dumps({'stage':'agent_plan','attempt':serial+retry,'create':creating}),flush=True)
            try:
                if creating:
                    proposal=json.loads(a.proposal.read_text()) if a.proposal and retry==0 else director.build(cfg,source,a.goal,feedback,previous,images)
                    save(folder/'agent_proposal.json',proposal)
                    if not isinstance(proposal,dict):raise ValueError('Proposal must be an object')
                    scene=proposal['scene'];action=proposal['action'];revision=0;start_time=0.
                else:
                    with db.transaction():
                        if not db.connection.in_transaction:db.connection.execute('BEGIN')
                        scene=scene_from_db(db);revision=db.get_meta('whitebox_revision');start_time=db.get_meta('tick')
                    prior=db.get_meta('whitebox_last_preview')
                    if not prior and (out/'status.json').exists():
                        last_status=json.loads((out/'status.json').read_text())
                        if last_status.get('status')=='COMMITTED':
                            last_frames=sorted((Path(last_status['attempt'])/'preview').glob('frame_*.png'))
                            prior=str(last_frames[-1]) if last_frames else None
                    action=director.plan(cfg,scene,a.goal,feedback,previous,[source]+([prior] if prior else []))
                    proposal={'scene':scene,'action':action}
                    save(folder/'agent_proposal.json',proposal)
                previous=deepcopy(proposal if creating else action)
                validate_scene(scene);validate_action(scene,action)
                # Terrain grounding is a declared compiler operation, never an observed height.
                snapped=[]
                if creating:
                    for e in scene['entities']:
                        if e['kind']=='actor':
                            height=terrain_height(scene['terrain'],e['position'][0],e['position'][2])
                            if abs(e['position'][1]-height)>1e-6:snapped.append({'id':e['id'],'from':e['position'][1],'to':height})
                            e['position'][1]=height
                    scene=prepare_scene(scene)
                errors=validate_geometry(scene)
                if errors:raise ValueError(str(errors))
                timeline,end,numerical=simulate(scene,action,start_time)
                numerical['grounding_adjustments']=snapped
            except (ValueError,KeyError,TypeError,AttributeError) as exc:
                feedback={'numerical_error':str(exc)};save(folder/'failure.json',feedback)
                print(json.dumps({'stage':'repair_required',**feedback}),flush=True);continue
            save(folder/'scene.json',scene);save(folder/'action.json',action)
            save(folder/'timeline.json',timeline);save(folder/'candidate_world.json',end)
            save(folder/'numerical_report.json',numerical)
            print(json.dumps({'stage':'render_whitebox','frames':len(timeline['frames']),'objects':len(scene['entities'])}),flush=True)
            render(folder/'scene.json',folder/'timeline.json',folder/'preview',mp4=True)
            images=preview_images(folder/'preview')
            images.extend(npc_views(scene,timeline,folder))
            if not images:raise RuntimeError('Engine produced no visual review images')
            visual=director.review(cfg,scene,action,numerical,images,source,continuation=not creating)
            save(folder/'visual_review.json',visual)
            alignment_ok=True
            if creating and any(e['kind'] in {'box','stairs'} for e in scene['entities']):
                from scripts.audit_scene_alignment import audit
                alignment=audit(cfg,scene,source,folder/'preview/camera.png',folder/'initial_alignment')
                alignment_ok=alignment.get('usable_for_camera_alignment') is True
            print(json.dumps({'stage':'visual_review','decision':visual.get('decision'),'issues':visual.get('issues'),'numerical_valid':numerical['valid']}),flush=True)
            if visual.get('decision')=='ACCEPT' and numerical['valid'] and visual.get('evidence') and alignment_ok:
                with db.transaction():
                    if creating:
                        initialize(db,scene);db.set_meta('whitebox_source_image',str(source))
                        db.set_meta('whitebox_source_kind',a.source_kind)
                        memory=AppearanceMemory(db)
                        for e in scene['entities']:
                            if e.get('appearance_bbox') or (e['kind'] in {'actor','staff'} and e['provenance']=='INFERRED'):
                                memory.add(e['id'],source,e.get('description',e['id']),original=True,
                                    source=a.source_kind,accepted=True,bbox_normalized=e.get('appearance_bbox'))
                    commit(db,end,timeline,numerical,visual,action,revision)
                    db.set_meta('whitebox_last_preview',str(sorted((folder/'preview').glob('frame_*.png'))[-1]))
                export_memory(db,folder)
                save(out/'world_current.json',scene_from_db(db))
                save(out/'status.json',{'status':'COMMITTED','time':db.get_meta('tick'),'revision':db.get_meta('whitebox_revision'),
                    'attempt':str(folder),'visual_review':visual,'npc_displacements':numerical['npc_displacements'],
                    'scope':'Whitebox scene/action accepted; no H3/video fidelity claim'})
                print(json.dumps({'stage':'committed','time':db.get_meta('tick'),'revision':db.get_meta('whitebox_revision')}),flush=True)
                return
            feedback={'visual':visual,'numerical':numerical}
            if creating and not alignment_ok:feedback['initial_alignment']=alignment
        save(out/'status.json',{'status':'NOT_COMMITTED','reason':'Review or numerical checks failed within attempt budget',
                               'feedback':feedback})
        print(json.dumps({'stage':'finished','status':'NOT_COMMITTED'}),flush=True)
    finally:db.connection.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['create','act','inspect','reconcile'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/default.yaml')
    p.add_argument('--image',type=Path);p.add_argument('--source-kind',default='GENERATED_SEED')
    p.add_argument('--proposal',type=Path,help='Replay an existing agent proposal for executor regression testing')
    p.add_argument('--goal',default='Sidestep, make one horizontal staff sweep, then recover. NPCs act independently.')
    p.add_argument('--attempts',type=int,default=2)
    p.add_argument('--edits',type=Path);p.add_argument('--reason')
    args=p.parse_args()
    if args.attempts<1:p.error('attempts must be positive')
    if args.command=='reconcile' and (args.edits is None or not args.reason):p.error('reconcile requires --edits and --reason')
    run(args)
