"""Qualify reference candidates against recorded entity lifetimes, not visibility."""
import json
from pathlib import Path
from collections import Counter, defaultdict
import pyarrow.parquet as pq


def main():
    root=Path('E:/GTA_555/0921')
    output=Path('E:/GTA_memory_prepared/reference_candidates_0921_v1')
    registry=defaultdict(list)
    for batch in pq.ParquetFile(root/'server/entities.parquet').iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            e=json.loads(row['payload_json']);registry[str(e['dataset_entity_id'])].append(e)
    counts=Counter();cached_path=None;frames={}
    with (output/'lifecycle_qualified.jsonl').open('w',encoding='utf-8') as dest:
        for line in (output/'candidates.jsonl').read_text(encoding='utf-8').splitlines():
            record=json.loads(line);path=Path(record['source_manifest']).parent/'frames.jsonl'
            if path!=cached_path:
                with path.open(encoding='utf-8') as source:
                    frames={r['proxy_frame_index']:r for r in (json.loads(s) for s in source)}
                cached_path=path
            indices=[record['reference_frame'],*range(record['target_start_frame'],record['target_end_frame_exclusive'])]
            ticks=[frames[i]['world_capture_tick'] for i in indices]
            eligible=[];rejected=[]
            for oid in record['candidate_entity_ids']:
                matches=registry[oid]
                if len(matches)!=1:
                    rejected.append({'id':oid,'reason':'missing_or_ambiguous_registry'});continue
                e=matches[0];start=e.get('spawn_sample_seq');end=e.get('despawn_sample_seq')
                if start is None or min(ticks)<start or (end is not None and max(ticks)>=end):
                    rejected.append({'id':oid,'reason':'outside_recorded_lifetime'});continue
                eligible.append({'object_id':record['episode_id']+':entity:'+oid,
                    'dataset_entity_id':oid,'entity_type':e.get('entity_type'),'model_hash':e.get('model_hash'),
                    'spawn_sample_seq':start,'despawn_sample_seq':end,
                    'lifecycle_basis':e.get('lifecycle_basis'),'visibility_verified':False,'state_compatibility':'UNKNOWN'})
            counts['windows']+=1;counts['eligible_entity_edges']+=len(eligible);counts['rejected_entity_edges']+=len(rejected)
            counts['windows_with_eligible_entities']+=bool(eligible)
            record['lifecycle_qualified_objects']=eligible;record['lifecycle_rejections']=rejected
            dest.write(json.dumps(record)+'\n')
    (output/'lifecycle_summary.json').write_text(json.dumps(dict(counts),indent=2),encoding='utf-8')
    print(dict(counts))


if __name__=='__main__':main()
