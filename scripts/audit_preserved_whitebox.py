"""Read-only comparison of the original world with its existing scene snapshot."""
import argparse
import json
from pathlib import Path
import sqlite3


def audit(root):
    snapshot=json.loads((root/'world_current.json').read_text())
    with sqlite3.connect((root/'world.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
        meta={k:json.loads(v) for k,v in db.execute('SELECT key,value FROM world_meta')}
        entities={}
        for identity,attributes,position,rotation,provenance in db.execute(
            'SELECT object_id,attributes,position,rotation,provenance FROM objects'
        ):
            x,y,z=json.loads(position)
            entities[identity]={**json.loads(attributes)['whitebox'],'id':identity,
                                'position':[x,z,-y],'rotation_y':json.loads(rotation)[2],
                                'provenance':provenance}
    assert entities=={e['id']:e for e in snapshot['entities']}
    assert meta['whitebox_header']=={k:v for k,v in snapshot.items() if k!='entities'}
    return {'original_database_matches_existing_snapshot':True,
            'revision':meta['whitebox_revision'],'world_time':meta['tick'],
            'history_entries':len(meta['whitebox_history']),'entity_count':len(entities)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    print(json.dumps(audit(p.parse_args().root),indent=2))
