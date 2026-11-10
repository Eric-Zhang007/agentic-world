"""Deterministic world operations. All model decisions belong to the Pi session."""
import json
from pathlib import Path
import sqlite3
import sys
import subprocess
from contextlib import closing

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from state.db import WorldDB
from whitebox.world import scene_from_db
from whitebox.motion import simulate
from whitebox.video_reconcile import commit_video, video_alignment


def read(path):
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(value,dict) and 'static_geometry' in value:
        for key in ('path','collision_path'):
            if key in value['static_geometry']:
                asset=Path(value['static_geometry'][key])
                if not asset.is_absolute():value['static_geometry'][key]=str((Path(path).resolve().parent/asset).resolve())
    return value


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def check_plan(plan):
    """Recompute feasibility; a saved valid=true is not an executable check."""
    scene=read(plan/'scene.json')
    saved=read(plan/'timeline.json')
    timeline,_,diagnostics=simulate(scene,read(plan/'action.json'),
        read(plan/'status.json')['world_start'],fps=saved['fps'])
    if not diagnostics['valid']:
        raise ValueError('PLAN_REJECTED: '+json.dumps(diagnostics['violations'],ensure_ascii=False))
    if timeline!=saved:
        raise ValueError('PLAN_CHANGED: regenerate the plan before compiling or rendering')
    return scene,timeline


def review_contract(timeline):
    actors=[e for e in timeline['frames'][-1]['entities'] if e['kind']=='actor']
    return {'actor_assessment_ids':[e['id'] for e in actors],
            'required_motion_assessment_ids':[e['id'] for e in actors if 'motion' in e],
            'optional_motion_assessment_ids':[e['id'] for e in actors if 'motion' not in e],
            'final_plan_frame_index':len(timeline['frames'])-1,
            'correction_offsets_are_relative_to_plan':True,
            'partial_acceptance_requires_continuation_goal':True}


def dispatch(operation, args):
    if operation == 'geometry_propose':
        from whitebox.geometry_pipeline import propose
        return propose(args)
    if operation == 'preview':
        from whitebox.world import validate_scene
        from whitebox.motion import prepare_scene
        scene = validate_scene(read(args['scene']))
        scene = prepare_scene(scene)
        output = Path(args['output']).resolve()
        output.mkdir(parents=True, exist_ok=False)
        save(output/'prepared_scene.json', scene)
        command = [sys.executable, str(ROOT/'whitebox/godot/render.py'),
                   '--scene', str(output/'prepared_scene.json'), '--output', str(output/'views'),
                   '--width', str(args.get('width', 1024)), '--height', str(args.get('height', 576))]
        with (output/'preview.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        return {'camera': str(output/'views/camera.png'), 'overview': str(output/'views/overview.png'),
                'prepared_scene': str(output/'prepared_scene.json')}
    if operation == 'initialize':
        from whitebox.world import initialize, validate_scene
        scene = validate_scene(read(args['scene']))
        target = Path(args['db']).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb'):
            pass
        db = WorldDB(target)
        try:
            initialize(db, scene)
        finally:
            db.connection.close()
        return {'db': str(target), 'revision': 0}
    if operation == 'fork':
        source = Path(args['source']).resolve()
        target = Path(args['db']).resolve()
        if not source.is_file():
            raise ValueError('Source world does not exist')
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb'):
            pass
        with closing(sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)) as src:
            with closing(sqlite3.connect(target)) as dst:
                src.backup(dst)
        return {'db': str(target), 'source': str(source)}
    if operation == 'inspect_video':
        import av
        from perception.frame_sampler import sample_frames
        plan = Path(args['plan']).resolve()
        video = Path(args['video']).resolve()
        evidence = Path(args['output']).resolve()
        evidence.mkdir(parents=True, exist_ok=False)
        with av.open(str(video)) as container:
            times = [float(f.time) for f in container.decode(video=0)]
        alignment = video_alignment(read(plan/'timeline.json'), times)
        frames = sample_frames(video, evidence/'frames', interval=args.get('interval', .5))
        from PIL import Image,ImageDraw
        columns=4;rows=(len(frames)+columns-1)//columns
        sheet=Image.new('RGB',(columns*384,rows*240),'#202020');draw=ImageDraw.Draw(sheet)
        for index,frame in enumerate(frames):
            with Image.open(frame['path']) as opened:
                image=opened.convert('RGB');image.thumbnail((384,216))
            x=index%columns*384;y=index//columns*240
            sheet.paste(image,(x,y));draw.text((x+5,y+219),f"{frame['timestamp']:.3f}s | frame {frame['frame_id']}",fill='white')
        sheet.save(evidence/'contact_sheet.jpg',quality=94)
        result = {'video': str(video), 'plan': str(plan), 'alignment': alignment, 'frames': frames,
                  'review_images':[str(evidence/'contact_sheet.jpg'),frames[0]['path'],frames[-1]['path']],
                  'review_contract':review_contract(read(plan/'timeline.json'))}
        save(evidence/'evidence.json', result)
        return result
    if operation == 'compile':
        plan = Path(args['plan']).resolve()
        checked_scene,checked_timeline=check_plan(plan)
        import av
        from PIL import Image
        plan = Path(args['plan']).resolve()
        output = Path(args['output']).resolve()
        output.mkdir(parents=True, exist_ok=False)
        width, height = args.get('width', 1024), args.get('height', 768)
        command = [sys.executable, str(ROOT/'whitebox/godot/render.py'),
                   '--scene', str(plan/'scene.json'), '--timeline', str(plan/'timeline.json'),
                   '--output', str(output/'proxy'), '--frames', 'all',
                   '--width', str(width), '--height', str(height)]
        with (output/'compile.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        files = [output/'proxy/camera.png', *sorted((output/'proxy').glob('frame_*.png'))]
        timeline = read(plan/'timeline.json')
        if len(files) != len(timeline['frames'])+1:
            raise ValueError('Compiler did not produce every planned frame plus initial frame')
        video = output/'whitebox.mp4'
        with av.open(str(video), 'w') as container:
            stream = container.add_stream('libx264', rate=timeline['fps'])
            stream.width, stream.height, stream.pix_fmt = width, height, 'yuv420p'
            for path in files:
                with Image.open(path) as image:
                    frame = av.VideoFrame.from_image(image.convert('RGB'))
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        import numpy as np
        depth_command=list(command)
        depth_command[depth_command.index('--output')+1]=str(output/'depth')
        depth_command+=['--render-pass','depth']
        with (output/'depth.log').open('w') as log:
            subprocess.run(depth_command,stdout=log,stderr=subprocess.STDOUT,check=True)
        depth_files=[output/'depth/camera.png',*sorted((output/'depth').glob('frame_*.png'))]
        if len(depth_files)!=len(files):raise ValueError('Depth and proxy frame counts differ')
        pixels=[]
        for path in depth_files:
            with Image.open(path) as image:
                if image.size!=(width,height):raise ValueError('Depth canvas mismatch')
                pixels.append(np.asarray(image.convert('RGB')))
        np.savez_compressed(output/'depth.npz',rgb=np.stack(pixels))
        save(output/'compiled_plan.json',{'plan':str(plan),'scene':checked_scene,
            'timeline':checked_timeline,'reference_video':str(video),'depth_control':str(output/'depth.npz')})
        return {'reference_video': str(video), 'depth_control': str(output/'depth.npz'),
                'frames': len(files), 'width': width, 'height': height,
                'initial_proxy': str(files[0]), 'final_proxy': str(files[-1]),
                'depth_encoding': {'near_white_world_units':2, 'far_black_world_units':18}}
    if operation == 'render':
        request = dict(args['request'])
        if not request.get('reference_video') or not request.get('first_frame'):
            raise ValueError('Native whitebox video and RGB first frame are required')
        request.setdefault('controls', [])
        video=Path(request['reference_video']).resolve()
        receipt=video.parent/'compiled_plan.json'
        if not receipt.is_file():
            raise ValueError('COMPILE_REQUIRED: use the compile tool before render')
        compiled=read(receipt)
        if Path(compiled['reference_video']).resolve()!=video:
            raise ValueError('Reference video does not belong to this compilation')
        scene,timeline=check_plan(Path(compiled['plan']))
        if scene!=compiled['scene'] or timeline!=compiled['timeline']:
            raise ValueError('RECOMPILE_REQUIRED: plan changed after compilation')
        if not args.get('db'):
            raise ValueError('WORLD_REQUIRED: render requires the accepted world DB')
        if request.get('prefix_video') or request.get('hud_profile'):
            raise ValueError('FIRST_FRAME_CONFLICT: prefix and HUD rewriting are not supported in the checked world loop')
        if request.get('anchor_first') is False:
            raise ValueError('FIRST_FRAME_REQUIRED: current RGB must be the start guide')
        request['anchor_first']=True
        if args.get('db'):
            from memory.paired_history import select
            if not Path(args['db']).is_file():
                raise FileNotFoundError(args['db'])
            memory_db = WorldDB(Path(args['db']))
            try:
                current_scene=scene_from_db(memory_db)
                if current_scene != scene:
                    raise ValueError('WORLD_CHANGED: render plan is not the accepted current scene')
                planned_status=read(Path(compiled['plan'])/'status.json')
                if (planned_status.get('revision') != memory_db.get_meta('whitebox_revision') or
                        planned_status['world_start'] != memory_db.get_meta('tick')):
                    raise ValueError('WORLD_CHANGED: plan revision or time is stale')
                expected_rgb=memory_db.get_meta('whitebox_last_preview')
                if not expected_rgb:
                    initial=[r for r in memory_db.get_meta('paired_history', []) if r['source']=='INITIAL_RGB']
                    if initial:
                        expected_rgb=initial[0]['paths']['rgb.png']
                if not expected_rgb:
                    raise ValueError('INITIAL_RGB_REQUIRED: capture the approved initial RGB before rendering')
                if expected_rgb:
                    import numpy as np
                    from PIL import Image
                    with Image.open(expected_rgb) as expected, Image.open(request['first_frame']) as actual:
                        if not np.array_equal(np.asarray(expected.convert('RGB')),np.asarray(actual.convert('RGB'))):
                            raise ValueError('FIRST_FRAME_MISMATCH: use the accepted current RGB endpoint')
                candidates=select(memory_db, timeline['frames'][-1]['camera'], limit=1)
            finally:
                memory_db.connection.close()
            if candidates:
                request['history_pair']=candidates[0]['history_path']
            selection=Path(request['output_dir'])
            selection.mkdir(parents=True,exist_ok=True)
            save(selection/'memory_selection.json', {'query_camera':timeline['frames'][-1]['camera'],
                 'candidates':candidates,'selection':'nearest historical camera to planned endpoint'})
        from renderer.comfy_h3 import ComfyH3Renderer
        renderer = ComfyH3Renderer(**args['renderer'])
        return {'video': str(renderer.render(**request))}
    db_path = Path(args['db']).resolve()
    if not db_path.is_file():
        raise ValueError('World must exist; fork an existing world first')
    db = WorldDB(db_path)
    try:
        scene = scene_from_db(db)
        revision = db.get_meta('whitebox_revision')
        if operation == 'memory_capture':
            from memory.paired_history import capture
            return capture(db, Path(args['output']), initial_rgb=args.get('initial_rgb'))
        if operation == 'memory_query':
            from memory.paired_history import select
            return {'candidates':select(db, args.get('camera',scene['camera']), limit=args.get('limit',3)),
                    'scope':'Historical paired frames; not current world facts'}
        if operation == 'status':
            return {'scene': scene, 'revision': revision, 'tick': db.get_meta('tick'),
                    'pending_action': db.get_meta('whitebox_video_action'),
                    'last_frame': db.get_meta('whitebox_last_preview'),
                    'history_length': len(db.get_meta('whitebox_history', []))}
        if operation == 'extend':
            from whitebox.world import reconcile,in_camera
            import numpy as np
            edits=read(args['edits'])
            if not isinstance(edits,list) or not edits:
                raise ValueError('Extension requires nonempty spawn edits')
            if args['revision']!=revision:
                raise ValueError('World changed before extension')
            # Even an initial reference image is evidence; do not insert objects
            # into its established view merely because no video was accepted yet.
            for edit in edits:
                if edit.get('op')!='spawn':raise ValueError('Extension only adds new objects')
                entity=edit['entity'];radius=float(np.linalg.norm(entity['size']))/2
                points=[entity['position']]+[(np.asarray(entity['position'])+offset).tolist()
                    for offset in ([radius,0,0],[-radius,0,0],[0,radius,0],[0,-radius,0],[0,0,radius],[0,0,-radius])]
                if any(in_camera(point,scene['camera']) for point in points):
                    raise ValueError('Extension overlaps the accepted starting view')
            output=Path(args['output']).resolve()
            output.mkdir(parents=True,exist_ok=False)
            save(output/'world_before.json',scene)
            with db.transaction():
                db.connection.execute('BEGIN IMMEDIATE')
                if db.get_meta('whitebox_revision')!=revision:
                    raise ValueError('World changed before extension')
                reconcile(db,edits,args['reason'])
            after=scene_from_db(db)
            result={'revision_before':revision,'revision_after':db.get_meta('whitebox_revision'),
                    'added_ids':[e['entity']['id'] for e in edits], 'reason':args['reason'],
                    'provenance':'AGENT_CREATED','world_time':db.get_meta('tick'),
                    'history_length':len(db.get_meta('whitebox_history',[])),
                    'scope':'Prospective world extension, not a video observation'}
            save(output/'world_after.json',after);save(output/'receipt.json',result)
            return result
        if operation == 'plan':
            from scripts.reconcile_whitebox_video import align_action
            action = read(args['action'])
            pending = db.get_meta('whitebox_video_action')
            if pending and not pending['action_complete'] and action.get('goal') != pending['continuation_goal']:
                raise ValueError('Finish pending continuation_goal before accepting a new goal')
            action, alignment = align_action(action)
            timeline, endpoint, diagnostics = simulate(scene, action, db.get_meta('tick'))
            if not diagnostics['valid']:
                raise ValueError(str(diagnostics['violations']))
            output = Path(args['output']).resolve()
            output.mkdir(parents=True, exist_ok=False)
            for name, value in [('scene', scene), ('action', action), ('timeline', timeline),
                                ('endpoint', endpoint), ('diagnostics', diagnostics),
                                ('status', {'revision': revision, 'world_start': db.get_meta('tick'),
                                            'status': 'PLANNED_NOT_COMMITTED', 'alignment': alignment})]:
                save(output/(name+'.json'), value)
            return {'plan': str(output), 'revision': revision, 'frames': len(timeline['frames']),
                    'video_frames':len(timeline['frames'])+1,
                    'duration_sec':len(timeline['frames'])/timeline['fps'],
                    'alignment':alignment, 'diagnostics':diagnostics,
                    'endpoint':{'camera':endpoint['camera'],
                        'actors':[{k:v for k,v in e.items() if k!='pose'}
                                  for e in endpoint['entities'] if e['kind']=='actor']},
                    'review_contract':review_contract(timeline)}
        if operation == 'commit':
            plan = Path(args['plan']).resolve()
            evidence = read(args['evidence'])
            review = read(args['review'])
            if evidence['plan'] != str(plan):
                raise ValueError('Evidence belongs to another plan')
            # Decode again: an edited manifest cannot bypass timing validation.
            import av
            with av.open(evidence['video']) as container:
                video_alignment(read(plan/'timeline.json'), [float(f.time) for f in container.decode(video=0)])
            history = db.get_meta('whitebox_history', [])
            for record in history:
                if record.get('artifacts', {}).get('planned_timeline') == str(plan/'timeline.json'):
                    if record['visual_review'] != review or record['artifacts']['video'] != evidence['video']:
                        raise ValueError('This plan was already accepted with a different review or video')
                    return {'status': 'ALREADY_COMMITTED', 'revision': record['revision_after']}
            if review.get('decision') == 'REGENERATE':
                return {'status': 'REJECTED', 'revision': revision, 'world_unchanged': True}
            output = Path(args['output']).resolve()
            output.mkdir(parents=True, exist_ok=False)
            artifacts = {'video': evidence['video'], 'last_frame': evidence['frames'][-1]['path'],
                         'planned_timeline': str(plan/'timeline.json'),
                         'corrected_timeline': str(output/'corrected_timeline.json'),
                         'review': str(output/'review.json')}
            save(output/'review.json', review)
            corrected, after, record = commit_video(db, read(plan/'scene.json'),
                read(plan/'timeline.json'), read(plan/'action.json'), review,
                read(plan/'status.json')['revision'], artifacts)
            save(output/'corrected_timeline.json', corrected)
            save(output/'world_after.json', after)
            save(output/'receipt.json', record)
            from memory.paired_history import capture
            try:
                paired_memory=capture(db, output/'paired_memory')
            except Exception as exc:
                paired_memory={'status':'CAPTURE_FAILED','error':str(exc),
                               'retry_operation':'memory_capture','world_commit_succeeded':True}
                save(output/'memory_capture_error.json',paired_memory)
            return {'status': 'COMMITTED', 'revision': record['revision_after'],
                    'paired_memory':paired_memory,
                    'action_complete': review['action_complete'],
                    'continuation_goal': review.get('continuation_goal')}
        raise ValueError('Unknown world operation: '+operation)
    finally:
        db.connection.close()


if __name__ == '__main__':
    request = json.load(sys.stdin)
    print(json.dumps(dispatch(request['operation'], request['args']), ensure_ascii=False))
