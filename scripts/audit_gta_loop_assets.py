"""Compare recorded GTA scene resources with executable whitebox resources."""
import json
from pathlib import Path
from collections import Counter
import sqlite3
from contextlib import closing

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/gta_pi_loop_20261102'


def main():
    reports={}
    for name in ['rural','urban','harbor']:
        source=json.loads((OUT/name/'source.json').read_text())
        state=json.loads(Path(source['state']).read_text())
        manifest=Path(state['source_manifest'])
        with (manifest.parent/'frames.jsonl').open() as f:
            row=next(r for line in f if (r:=json.loads(line))['proxy_frame_index']==state['frame_index'])
        candidates=[manifest.parent/row['scene_index'],Path(state['media_paths']['proxy_video']).parent/row['scene_index']]
        scene_path=next((p for p in candidates if p.is_file()),None)
        scene=json.loads(scene_path.read_text()) if scene_path else {}
        reports[name]={
            'source_frame':state['frame_index'],'scene_path':str(scene_path),
            'static_candidates_in_region':row.get('static_candidates'),
            'dynamic_scope_entities':row.get('visible_scope_entities'),
            'dynamic_primitive_parts':row.get('dynamic_primitive_parts'),
            'pose_sources':row.get('pose_sources'),
            'scene_keys':list(scene),
            'placement_kinds':dict(Counter(x.get('kind','unknown') for x in scene.get('static_placement_ids',[]))),
            'surface_kinds':dict(Counter(x.get('kind','unknown') for x in scene.get('surface_components',[]))),
            'collision_triangles':scene.get('collision_triangles'),
            'geometry_policy':scene.get('geometry_policy'),
            'building_proxy':scene.get('building_proxy'),
            'vegetation_shape':scene.get('vegetation_shape'),
            'placement_examples':scene.get('static_placement_ids',[])[:3],
            'surface_examples':scene.get('surface_components',[])[:2],
            'scope':'Region candidates are NOT per-frame visible object counts.'}
    report={'cases':reports,'current_resources':['actor','box','staff','stairs'],
        'conflicts':[
            'Source game uses Z-up world positions; executable whitebox uses Y-up; explicit basis and local origin needed.',
            'Recorded buildings use original collision surfaces, not closed solid bounding boxes; replacing them with boxes loses openings and walkable spaces.',
            'Current terrain is a procedural height field; overhangs, bridges and multi-level walkable surfaces are unsupported.',
            'Current vehicle category can only be static box composition; no vehicle controller, seat association or wheel articulation.',
            'Current reconciliation can correct actor root/yaw, coarse posture and camera, not arbitrary object transforms or ownership.',
            'Source dynamic pose source may be rigid humanoid bounds, not measured detailed body joints.',
            'GTA screen-ray camera calibration and Pi perspective fov are different interfaces; matching screen composition is not recovering metric calibration.'
        ]}
    world=Path('E:/GTA_555_complete_rebuild_20261025/source/data/worlds/gta5_scene_v3_20260908')
    manifest=json.loads((world/'manifest.json').read_text())
    report['available_map_assets']={'path':str(world),'counts':manifest.get('counts'),
        'coordinate_system':manifest.get('coordinate_system'),
        'files':['static.sqlite3','placements.jsonl.gz','heightmap.json','collision/collision.sqlite3'],
        'reuse_candidates':['wm_engine/world/static_store.py','wm_engine/world/collision.py',
                           'wm_engine/simulator/road_vehicle.py','wm_engine/simulator/vehicle_physics.py']}
    report['map_sqlite_schemas']={}
    for relative in ['static.sqlite3','collision/collision.sqlite3']:
        path=world/relative
        with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as db:
            report['map_sqlite_schemas'][relative]=[{'table':name,'sql':sql} for name,sql in
                db.execute("SELECT name,sql FROM sqlite_master WHERE type='table'")]
    (OUT/'asset_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for name,r in reports.items():
        print(name, {k:r[k] for k in ['static_candidates_in_region','dynamic_scope_entities','collision_triangles','placement_kinds','surface_kinds']})


if __name__=='__main__':main()
