"""Read recorded static scene membership without asserting pixel visibility."""
from collections import Counter
from pathlib import Path
import orjson
import sqlite3


def inspect_frame(manifest_path, frame_index):
    manifest_path=Path(manifest_path)
    manifest=orjson.loads(manifest_path.read_bytes())
    frame=None
    with (manifest_path.parent/'frames.jsonl').open('rb') as source:
        for line in source:
            row=orjson.loads(line)
            if row['proxy_frame_index']==frame_index:
                frame=row;break
    if frame is None:raise KeyError(frame_index)
    scene_name=frame.get('scene_index')
    result={'episode_id':manifest['episode_id'],'agent_id':manifest['agent_id'],
            'segment':manifest['segment'],'frame_index':frame_index,
            'time_ns':frame.get('server_time_estimate_ns'),
            'renderer_entity_ids':frame.get('entity_ids',[]),
            'static_candidates':frame.get('static_candidates',0),
            'static_visibility':'UNKNOWN: scene membership does not prove visibility or occlusion',
            'scene_source':scene_name,'placements':[],'surface_component_count':0}
    if not scene_name:return result
    scene=orjson.loads((manifest_path.parent/scene_name).read_bytes())
    suppressed={str(v['id']):'vegetation_suppressed' for v in scene.get('suppressed_vegetation_placements',[])}
    suppressed.update({str(v['id']):v.get('reason','excluded') for v in scene.get('excluded_ground_clutter_placements',[])})
    for obj in scene.get('static_placement_ids',[]):
        oid=str(obj['id'])
        result['placements'].append({'placement_id':oid,'kind':obj.get('kind'),
            'memory_key':manifest['episode_id']+':placement:'+oid,
            'membership':'scene_candidate','visibility':'unknown','exclusion':suppressed.get(oid)})
    result['surface_component_count']=len(scene.get('surface_components',[]))
    result['surface_kind_counts']=dict(Counter(v.get('kind','unknown') for v in scene.get('surface_components',[])))
    result['reported_collision_triangles']=scene.get('collision_triangles')
    result['geometry_policy']=scene.get('geometry_policy')
    result['scope']='Frame-linked candidate identities; episode-scoped keys, no cross-episode identity assertion; not masks or visible-object labels'
    return result


def index_segment(manifest_path, output):
    """Store frame->scene->placement membership, never inferred visibility."""
    manifest_path=Path(manifest_path);output=Path(output)
    if output.exists():raise ValueError('Existing scene history is not overwritten')
    manifest=orjson.loads(manifest_path.read_bytes())
    db=sqlite3.connect(output)
    try:
        db.executescript('''CREATE TABLE frame_scene(frame INTEGER PRIMARY KEY,time_ns,scene);
            CREATE TABLE scene_placement(scene,placement_id,kind,PRIMARY KEY(scene,placement_id));
            CREATE TABLE scene_metadata(scene PRIMARY KEY,data);
            CREATE TABLE metadata(key PRIMARY KEY,value);''')
        names=set();frame_count=0
        with (manifest_path.parent/'frames.jsonl').open('rb') as source:
            for line in source:
                row=orjson.loads(line);name=row.get('scene_index')
                db.execute('INSERT INTO frame_scene VALUES (?,?,?)',(row['proxy_frame_index'],row.get('server_time_estimate_ns'),name))
                if name:names.add(name)
                frame_count+=1
        for name in sorted(names):
            scene=orjson.loads((manifest_path.parent/name).read_bytes())
            db.executemany('INSERT OR IGNORE INTO scene_placement VALUES (?,?,?)',
                           [(name,str(v['id']),v.get('kind')) for v in scene.get('static_placement_ids',[])])
            db.execute('INSERT INTO scene_metadata VALUES (?,?)',(name,orjson.dumps({
                'collision_mesh_ids':scene.get('collision_mesh_ids',[]),
                'surface_component_count':len(scene.get('surface_components',[])),
                'geometry_policy':scene.get('geometry_policy')})))
        db.executescript('CREATE INDEX placement_lookup ON scene_placement(placement_id); CREATE INDEX frame_scene_lookup ON frame_scene(scene);')
        unique=db.execute('SELECT count(DISTINCT placement_id) FROM scene_placement').fetchone()[0]
        report={'manifest':str(manifest_path),'episode_id':manifest['episode_id'],
                'frames':frame_count,'scenes':len(names),'unique_candidate_placements':unique,
                'visibility':'UNKNOWN','scope':'Candidate membership by exact recorded frame; not visible pixel or occlusion labels'}
        db.execute('INSERT INTO metadata VALUES (?,?)',('summary',orjson.dumps(report).decode()))
        db.commit();return report
    finally:db.close()
