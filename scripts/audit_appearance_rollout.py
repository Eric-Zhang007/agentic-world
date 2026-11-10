"""Verify saved memory consumers against native renderer metadata and history."""
import argparse
import json
from pathlib import Path
import sqlite3


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit(episode):
    with sqlite3.connect(f'file:{(episode/"world.sqlite").resolve()}?mode=ro',uri=True) as db:
        meta={key:json.loads(value) for key,value in db.execute('SELECT key,value FROM world_meta')}
        ids={row[0] for row in db.execute('SELECT object_id FROM objects')}
    history=meta['whitebox_history'];baseline=read(episode/'history_before.json')
    assert history[:len(baseline)]==baseline,'Prior history changed'
    stored={ref['id']:ref for ref in meta.get('appearance_reference_memory',[])}
    assert all(ref['object_id'] in ids for ref in stored.values())
    segments=[]
    for segment in sorted(episode.glob('segment_*')):
        if not (segment/'render/metadata.json').exists():continue
        request=read(segment/'appearance_memory/request.json')
        contract=read(segment/'render_contract.json')
        native=read(segment/'render/metadata.json')
        assert contract['appearance_reference_ids']==[r['id'] for r in request['refs']]
        if 'renderer_references' in contract:
            selected=read(segment/'renderer_references.json')
            assert selected==contract['renderer_references'],'Renderer selection changed after prompt compilation'
            by_id={ref['id']:ref for ref in request['refs']}
            assert len(selected['reference_ids'])==len(set(selected['reference_ids']))
            assert all(identity in by_id for identity in selected['reference_ids'])
            assert selected['images']==[by_id[identity]['crop_path'] for identity in selected['reference_ids']]
            renderer_images=selected['images']
        else:
            # Historical runs predate the separation and intentionally passed every crop.
            renderer_images=request['images']
        assert native['appearance_images']==renderer_images
        assert native['reference_blocks']==len(renderer_images)+1
        assert len(request['refs'])==len(request['images'])
        assert all(ref['id'] in stored and Path(path).is_file()
                   for ref,path in zip(request['refs'],request['images']))
        assert native['first_frame_guide'] or native['prefix_frame_count']>1
        status=read(segment/'review/status.json') if (segment/'review/status.json').exists() else {'status':'NOT_REVIEWED'}
        segments.append({'segment':segment.name,'reference_blocks':native['reference_blocks'],
            'agent_memory_images':len(request['images']),
            'renderer_supplementary_images':len(renderer_images),
            'identity_objects':[r['object_id'] for r in request['refs']],
            'review':status,'native_reference_paths_verified':True})
    result={'prior_history_preserved':True,'accepted_history_before':len(baseline),
            'accepted_history_after':len(history),'stored_reference_count':len(stored),
            'capture_count':len(meta.get('appearance_captures',{})),
            'segments':segments,'scope':'Wiring and ledger verification; not a visual quality score.'}
    if (episode/'next_plan/status.json').exists():
        status=read(episode/'next_plan/status.json')
        request=read(episode/'next_plan/appearance_memory/request.json')
        assert status['status']=='PLANNED_NOT_COMMITTED'
        assert status['revision']==meta['whitebox_revision']==request['world_revision']
        assert read(episode/'next_plan/scene.json')==read(episode/'world_after.json')
        latest={ref['id'] for ref in stored.values()
                if ref.get('evidence',{}).get('revision')==history[-1]['revision_after']}
        retrieved={ref['id'] for ref in request['refs']}
        assert retrieved.issubset(stored) and retrieved.intersection(latest)
        result['next_plan']={'revision':status['revision'],'reference_count':len(retrieved),
            'newly_accepted_reference_count':len(retrieved.intersection(latest)),
            'starts_from_actual_committed_world':True,'status':status['status']}
    (episode/'memory_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode',type=Path)
    print(json.dumps(audit(parser.parse_args().episode),indent=2))
