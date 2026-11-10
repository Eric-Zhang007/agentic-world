"""Report motion evidence separately from persistence and timestamp correctness."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.verify_evidence_rollouts import verify


def inspect(root):
    chain=verify(root);records=[]
    for segment in sorted(root.glob('segment_*')):
        review_path=segment/'review/review.json'
        if not review_path.exists():continue
        review=json.loads(review_path.read_text())
        action=json.loads((segment/'plan/action.json').read_text())
        before=json.loads((segment/'plan/scene.json').read_text())
        actor=next(e for e in before['entities'] if e['id']==action['actor_id'])
        record={'segment':segment.name,'decision':review['decision'],
                'incoming_motion':actor.get('motion'),
                'endpoint_motion':next((m for m in review.get('motion_assessments',[]) if m['id']==actor['id']),None),
                'boundary_motion':review.get('boundary_motion'),
                'action_complete':review.get('action_complete')}
        alignment=segment/'continuation.alignment.json'
        if alignment.exists():record['overlap']=json.loads(alignment.read_text())
        records.append(record)
    first=records[0].get('endpoint_motion') if records else None
    supported=(len(records)>=2 and chain['accepted_segments']>=2 and first is not None
               and np.linalg.norm(first['velocity'])>1e-8
               and records[1].get('boundary_motion',{}).get('status')=='SUPPORTED')
    result={'state_chain':chain,'segments':records,'continuous_boundary_supported_by_agent':bool(supported),
            'scope':'Agent estimates from sampled images, not independent measured 3D velocities. Review actual seam video.'}
    (root/'motion_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    print(json.dumps(inspect(p.parse_args().root),indent=2))
