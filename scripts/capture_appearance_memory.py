"""Recover or initialize references from the latest already accepted video."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from state.db import WorldDB
from whitebox.appearance_rollout import capture_accepted,retrieve

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--world',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/default.yaml')
    a=p.parse_args()
    if not a.world.is_file():raise FileNotFoundError(a.world)
    db=WorldDB(a.world)
    try:
        result=capture_accepted(db,yaml.safe_load(a.config.read_text())['agent'],a.output)
        bundle=retrieve(db,a.output/'retrieved')
        print(json.dumps({'capture':result,'retrieved':len(bundle['refs'])}))
    finally:db.connection.close()
