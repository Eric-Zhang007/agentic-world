"""Read-only task-directed memory retrieval using the configured agent."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys

import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from state.db import WorldDB
from whitebox.appearance_rollout import retrieve_for_task


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    # Use an in-memory snapshot so a diagnostic cannot migrate or edit its source.
    db=WorldDB(':memory:')
    try:
        with sqlite3.connect(a.world.resolve().as_uri()+'?mode=ro',uri=True) as source:
            source.backup(db.connection)
        before=list(db.connection.iterdump())
        config=yaml.safe_load(a.config.read_text())['agent']
        result=retrieve_for_task(db,config,a.output,{'goal':a.goal},db.get_meta('whitebox_last_preview'))
        assert list(db.connection.iterdump())==before,'Memory query changed its snapshot'
        summary={'selected_objects':[r['object_id'] for r in result['refs']],
                 'images':len(result['images']),'source_world_read_only':True,
                 'missing_requested_objects':result['missing_requested_objects'],
                 'missing_requested_views':result['missing_requested_views'],
                 'selection_missing_evidence':result.get('selection_missing_evidence',[])}
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary))
    finally:
        db.connection.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--goal',required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/default.yaml')
    main(parser.parse_args())
