"""Index historical object references; candidate co-occurrence is not visibility."""
import argparse
import json
from collections import Counter
from pathlib import Path
from memory.gta_validation import camera_exclusion


def candidates(rows, target_frames=120, reference_gap_frames=120):
    by_index = {r['proxy_frame_index']: r for r in rows}
    results=[]
    for start in range(0, max(by_index, default=-1)+1, target_frames):
        target=[by_index.get(i) for i in range(start,start+target_frames)]
        reference=by_index.get(start-reference_gap_frames)
        if reference is None or camera_exclusion(reference):continue
        if any(r is None or camera_exclusion(r) for r in target):continue
        all_rows=[reference,*target]
        if len({r.get('server_clock_domain') for r in all_rows})!=1:continue
        if len({r.get('client_clock_domain') for r in all_rows})!=1:continue
        times=[r['server_time_estimate_ns'] for r in all_rows]
        if any(a>=b for a,b in zip(times,times[1:])):continue
        if any(b['rgb_decode_index']!=a['rgb_decode_index']+1 for a,b in zip(target,target[1:])):continue
        # Appearance is scoped to the observed agent, not every entity in scene.
        appearance=[r.get('appearance_id') for r in all_rows]
        if any(a is None for a in appearance) or len(set(appearance))!=1:continue
        common=set(str(x) for x in reference.get('entity_ids',[]))
        for r in target:common.intersection_update(str(x) for x in r.get('entity_ids',[]))
        if not common:continue
        results.append({'reference_frame':reference['proxy_frame_index'],
            'target_start_frame':start,'target_end_frame_exclusive':start+target_frames,
            'observed_at_ns':times[0],'used_at_ns':times[1],
            'reference_age_sec':(times[1]-times[0])/1e9,
            'candidate_entity_ids':sorted(common),'observed_agent_appearance_id':appearance[0],
            'visibility_verified':False,'state_compatibility':'UNKNOWN','training_ready':False})
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--episode',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    counts=Counter(); exclusions=[]
    with (args.output/'candidates.jsonl').open('w',encoding='utf-8') as dest:
        for path in sorted(args.episode.glob('derived/segments/*/*/manifest.json')):
            try:
                m=json.loads(path.read_text(encoding='utf-8'))
                if not (args.episode/'agents'/m['agent_id']/'camera.parquet').is_file():
                    counts['missing_camera_files']+=1;continue
                if not all((args.episode/m[k]).is_file() for k in ['rgb_video','proxy_video']):
                    counts['missing_video_files']+=1;continue
                with (path.parent/'frames.jsonl').open(encoding='utf-8') as source:
                    rows=[json.loads(s) for s in source if s.strip()]
                counts['segments_scanned']+=1
                for row in rows:
                    reason=camera_exclusion(row)
                    if reason:counts['excluded_'+reason]+=1
                for gap in [120,480]:
                    for record in candidates(rows,120,gap):
                        record.update({'episode_id':m['episode_id'],'agent_id':m['agent_id'],
                            'segment':m['segment'],'source_manifest':str(path),
                            'remaining_checks':['actual camera/world joins','lifecycle','object visibility',
                                                'alignment','object state','HUD','media decode']})
                        dest.write(json.dumps(record)+'\n');counts['candidate_windows']+=1
                        counts['candidate_object_edges']+=len(record['candidate_entity_ids'])
            except (OSError,ValueError,KeyError) as error:
                exclusions.append({'path':str(path),'error':str(error)})
            if counts['segments_scanned']%25==0:print(dict(counts),flush=True)
    (args.output/'summary.json').write_text(json.dumps({'counts':dict(counts),'errors':exclusions,
        'status':'CANDIDATES_ONLY','reference_policy':'historical same-segment frames; 5s target; nominal 5s/20s reference gap',
        'not_proven':'visibility, same entity state, revisit, decode and spatial alignment'},indent=2),encoding='utf-8')
    print(dict(counts))


if __name__=='__main__':main()
