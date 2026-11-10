"""Read-only audit of committed whitebox artifacts and their persistent state."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from state.db import WorldDB
from whitebox.world import scene_from_db


def audit(root):
    db=WorldDB(root/'world.sqlite')
    try:
        history=db.get_meta('whitebox_history',[]);scene=scene_from_db(db)
        records=[];initial=None
        for folder in sorted(root.glob('attempt_*')):
            if not (folder/'visual_review.json').exists():continue
            visual=json.loads((folder/'visual_review.json').read_text())
            before=json.loads((folder/'scene.json').read_text());after=json.loads((folder/'candidate_world.json').read_text())
            numerical=json.loads((folder/'numerical_report.json').read_text())
            timeline=json.loads((folder/'timeline.json').read_text())
            if initial is None:initial=before
            old={e['id']:e for e in before['entities']};new={e['id']:e for e in after['entities']}
            records.append({'attempt':folder.name,'decision':visual['decision'],'numerical_valid':numerical['valid'],
                'frames':len(timeline['frames']),'fps':timeline['fps'],
                'start_time':timeline['frames'][0]['time']-1/timeline['fps'],'end_time':timeline['frames'][-1]['time'],
                'ids_preserved':set(old)==set(new),'terrain_preserved':before['terrain']==after['terrain'],
                'static_preserved':all(new.get(k)==v for k,v in old.items() if v['kind'] not in {'actor','staff'}),
                'npc_displacements':numerical['npc_displacements'],'proxy_contacts':numerical['events'],
                'grip_error':numerical.get('max_grip_axis_error'),'sole_clearance':numerical.get('support_sole_clearance_range')})
        original={e['id']:e for e in initial['entities']};current={e['id']:e for e in scene['entities']}
        result={'revision':db.get_meta('whitebox_revision'),'time':db.get_meta('tick'),'history_entries':len(history),
            'entity_count':len(current),'ids_persisted':set(original)==set(current),
            'static_scene_persisted':all(current.get(k)==e for k,e in original.items() if e['kind'] not in {'actor','staff'}),
            'terrain_persisted':initial['terrain']==scene['terrain'],'appearance_references':len(db.get_meta('appearance_reference_memory',[])),
            'actions':records,'limitations':['Single image geometry is inferred','Sparse visual review','Approximate proxy contacts, no damage simulation']}
        (root/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    finally:db.connection.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);audit(p.parse_args().root)
