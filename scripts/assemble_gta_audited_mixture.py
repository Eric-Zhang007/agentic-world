"""Build unified manifests only after every eligible scene has an audit decision."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from memory.gta_mixed_dataset import balanced_round, MixedMemoryDataset
from scripts.finalize_gta_passage_training import merge_unique_rows
from scripts.audit_gta_scene_memory import validate_results


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')


def resolve_review(key, sample_path, audit, direct_audit=None):
    paths=([direct_audit/(key+'.json')] if direct_audit else [])+[audit/(key+'.json')]
    for path in paths:
        if not path.exists():continue
        decision=json.loads(path.read_bytes())
        validate_results({'reviews':[decision]},[key])
        if Path(decision['sample_path']).resolve()!=sample_path.resolve():
            raise ValueError('Audit path differs')
        origin=('direct_local_visual' if decision.get('method','').startswith('Direct Codex')
                else 'prior_api_visual' if decision.get('model') else 'other_recorded_review')
        decision['review_origin']=origin
        return path,decision
    return None,None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots',type=Path,nargs='+',required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--direct-audit',type=Path,help='Local direct decisions take precedence over earlier audit records')
    p.add_argument('--old-mixture',type=Path,required=True)
    p.add_argument('--split-map',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    old=sum((read_rows(a.old_mixture/(s+'.jsonl')) for s in ('train','validation','test')),[])
    old=[r for r in old if r['dataset']=='objects']
    splitmap=json.loads(a.split_map.read_text())
    for row in old:
        if row['episode_id'] in splitmap and splitmap[row['episode_id']]!=row['split']:
            raise ValueError('Existing episode split conflict')
        splitmap[row['episode_id']]=row['split']
    roots={'objects':json.loads((a.old_mixture/'roots.example.json').read_text())['objects']}
    rows=[];ledger=[];missing=[];decisions=[]
    for source in a.roots:
        dataset=source.name
        roots[dataset]=str(source/'training')
        for path in sorted((source/'training/samples').glob('*/sample.json')):
            sample=json.loads(path.read_bytes());key=sample['sample_id']
            if sample['training_status'].startswith('EXCLUDED'):
                ledger.append({'sample_id':key,'accepted':False,'reason':sample['training_status']});continue
            decision_path,decision=resolve_review(key,path,a.audit,a.direct_audit)
            if decision is None:missing.append(key);continue
            ledger.append(decision)
            if decision['accepted'] is not True:continue
            episode=sample['episode_id']
            if episode not in splitmap:raise ValueError('Missing explicit episode split: '+episode)
            ref=json.loads((path.parent/'reference_state.json').read_bytes())
            target=json.loads((path.parent/'target_state.json').read_bytes())
            if ref['episode_id']!=episode or target['episode_id']!=episode or ref['server_clock_domain']!=target['server_clock_domain']:
                raise ValueError('Reference world/clock mismatch')
            if sample['reference_time_ns']>=sample['target_time_ns']:raise ValueError('Nonhistorical reference')
            evidence=sample['reference'];age=evidence['age_sec'];angle=evidence['view_angle_deg']
            x,y,z=target['camera_position'];site=f'{int(x//40)}_{int(y//40)}_{int(z//8)}'
            rows.append(dict(dataset=dataset,sample_id=key,sample_root='samples/'+key,
                split=splitmap[episode],episode_id=episode,task='scene_background',
                balance_group=episode+':'+site,kind=evidence['kind'],
                age_bin='under10s' if age<10 else '10to30s' if age<30 else '30to120s' if age<120 else 'over120s',
                angle_bin='under15deg' if angle<15 else '15to45deg' if angle<45 else 'over45deg',
                historical_reference_pair_reviewed=True,annotation_level='sampled_background_memory_review',
                review_origin=decision['review_origin']))
            decisions.append(decision_path)
    if missing:raise ValueError(f'{len(missing)} scene decisions still missing; no final manifests written')
    combined,duplicates=merge_unique_rows(old+rows)
    ids=[r['sample_id'] for r in combined]
    if len(ids)!=len(set(ids)):raise ValueError('Global duplicate IDs')
    # Same source segment cannot cross splits, regardless of task or window ID.
    segments={}
    for row in combined:
        segment=row['sample_id'].rsplit('_f' if row['task']=='scene_background' else '_w',1)[0]
        if segments.setdefault(segment,row['split'])!=row['split']:raise ValueError('Source window split leakage')
    a.output.mkdir(parents=True,exist_ok=False)
    for split in ('train','validation','test'):
        write_rows(a.output/(split+'.jsonl'),[r for r in combined if r['split']==split])
    fractions={'scene_background':.5,'object_person':.2,'object_vehicle':.2,'object_mixed':.1}
    train=[r for r in combined if r['split']=='train']
    plans=[]
    for seed in range(10):
        nearby=[r for r in train if r['task']=='scene_background' and r['kind']=='nearby_history']
        passages=[r for r in train if r['task']=='scene_background' and r['kind']!='nearby_history']
        if not passages:raise ValueError('No return/cross-camera training pool')
        subfractions={k:v for k,v in fractions.items() if k!='scene_background'}
        subfractions['scene_passage']=.25 if nearby else .5
        if nearby:subfractions['scene_nearby']=.25
        pool=[r for r in train if r['task']!='scene_background']
        pool += [{**r,'task':'scene_passage'} for r in passages]
        pool += [{**r,'task':'scene_nearby'} for r in nearby]
        plan=balanced_round(pool,subfractions,seed=seed)
        for row in plan:
            if row['task'] in ('scene_passage','scene_nearby'):row['task']='scene_background'
        write_rows(a.output/f'train_round_{seed:02d}.jsonl',plan)
        plans.append({'seed':seed,'count':len(plan),'tasks':dict(Counter(r['task'] for r in plan)),
            'scene_kinds':dict(Counter(r.get('kind') for r in plan if r['task']=='scene_background'))})
    (a.output/'roots.example.json').write_text(json.dumps(roots,indent=2))
    (a.output/'episode_splits.json').write_text(json.dumps(splitmap,indent=2))
    write_rows(a.output/'exclusion_ledger.jsonl',ledger)
    provenance=a.output/'scene_audit';provenance.mkdir()
    for path in decisions:shutil.copy2(path,provenance/path.name)
    report={'counts':dict(Counter(r['task'] for r in combined)),
        'scene_kinds':dict(Counter(r['kind'] for r in rows)),
        'scene_age_bins':dict(Counter(r['age_bin'] for r in rows)),
        'scene_angle_bins':dict(Counter(r['angle_bin'] for r in rows)),
        'splits':dict(Counter(r['split'] for r in combined)),
        'duplicates_removed':duplicates,'rounds':plans,'mixture_fractions':fractions,
        'accepted_scene_review_origins':dict(Counter(r['review_origin'] for r in rows)),
        'max_scene_per_site_per_round':2,
        'training_run':False,'scope':'Scene background memory audited at five target times; old object QA retained. Episode isolated, not unseen map-location split.'}
    (a.output/'distribution_report.json').write_text(json.dumps(report,indent=2))
    exercised=[]
    for split in ('train','validation','test'):
        ds=MixedMemoryDataset(a.output/(split+'.jsonl'),roots,temporal_indices=[0,30,60,90,119])
        groups=set()
        for i,row in enumerate(ds.rows):
            folder=Path(roots[row['dataset']])/row['sample_root']
            sample=json.loads((folder/'sample.json').read_bytes())
            for category in ('conditions','supervision'):
                for name in ('target_whitebox','target_rgb','prompt','valid_pixels'):
                    path=sample[category].get(name)
                    if path and not (folder/path).is_file():raise ValueError('Missing sample dependency')
            group=(row['dataset'],row['task'],row.get('kind'))
            if group in groups:continue
            item=ds[i]
            if item['conditions']['target_whitebox'].shape!=item['supervision']['target_rgb'].shape:raise ValueError('Loader shape mismatch')
            groups.add(group);exercised.append(row['sample_id'])
    (a.output/'verification.json').write_text(json.dumps({'passed':True,'records':len(combined),
        'episode_and_source_segment_split_isolation':True,'loader_examples':exercised,
        'scope':'Manifest and dependency checks plus representative loader checks; prior export full decode retained'},indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
