"""Check accepted-state continuation and preservation without interpreting pixels."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from state.db import WorldDB
from whitebox.world import scene_from_db


def verify(root):
    before=json.loads((root/'world_before.json').read_text())
    baseline=json.loads((root/'history_before.json').read_text())
    db=WorldDB(root/'world.sqlite')
    try:
        history=db.get_meta('whitebox_history',[])
        if history[:len(baseline)]!=baseline:raise ValueError('Archived history changed')
        expected=before;records=[];accepted=0
        for segment in sorted(root.glob('segment_*')):
            plan=segment/'plan'
            if not (plan/'scene.json').exists():continue
            if json.loads((plan/'scene.json').read_text())!=expected:
                raise ValueError('Next plan did not start from actual accepted state')
            status=segment/'review/status.json'
            if not status.exists():
                records.append({'segment':segment.name,'status':'NOT_REVIEWED'});continue
            decision=json.loads(status.read_text())
            if decision['status']=='COMMITTED':
                corrected=json.loads((segment/'review/corrected_timeline.json').read_text())
                after=json.loads((segment/'review/world_after.json').read_text())
                if after['entities']!=corrected['frames'][-1]['entities'] or after['camera']!=corrected['frames'][-1]['camera']:
                    raise ValueError('Committed endpoint differs from actual corrected state')
                expected=after;accepted+=1
            records.append({'segment':segment.name,**decision})
        if scene_from_db(db)!=expected:raise ValueError('Database does not match latest accepted endpoint')
        if len(history)!=len(baseline)+accepted:raise ValueError('Unexpected history records')
        result={'prior_history_preserved':True,'accepted_segments':accepted,'state_chain_valid':True,
                'segments':records,'scope':'State/ledger checks only; not independent visual validation'}
        (root/'chain_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        return result
    finally:db.connection.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    print(json.dumps(verify(p.parse_args().root),indent=2))
