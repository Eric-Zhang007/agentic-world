"""Coverage reporting and mixed manifests for expanded passage references."""
import argparse
from collections import Counter,defaultdict
import html
from pathlib import Path
import random
import shutil
import zipfile

import orjson
from scripts.validate_gta_passage_training import validate


def write_rows(path,rows):path.write_bytes(b''.join(orjson.dumps(r)+b'\n' for r in rows))


def require_pair_review(sample):
    review = sample.get('supplemental_target_review', {})
    if (sample['training_status'] != 'AUTO_REFERENCE_WITH_REVIEWED_TARGET_QA'
            or review.get('accepted') is not True
            or review.get('historical_reference_pair_reviewed') is not True):
        raise ValueError('Explicit current reference/target pair review required: ' + sample['sample_id'])


def merge_unique_rows(rows):
    unique = {}
    episode_splits = {}
    duplicates = 0
    for row in rows:
        episode = row['episode_id']
        if episode in episode_splits and episode_splits[episode] != row['split']:
            raise ValueError('Episode crosses dataset splits: ' + episode)
        episode_splits[episode] = row['split']
        family = 'scenes' if row['task'] == 'scene_background' else row['dataset']
        key = (family, row['sample_id'])
        previous = unique.get(key)
        if previous:
            if any(previous[k] != row[k] for k in ('episode_id', 'split', 'task')):
                raise ValueError('Conflicting duplicate sample: ' + row['sample_id'])
            duplicates += 1
        # Current scene records follow previous releases and replace the same ID.
        unique[key] = row
    return list(unique.values()), duplicates


def stratified_take(rows,count,rng):
    groups=defaultdict(list)
    for row in rows:groups[row['balance_group']].append(row)
    for group in groups.values():rng.shuffle(group)
    selected=[]
    while len(selected)<count:
        keys=[k for k,v in groups.items() if v];rng.shuffle(keys)
        if not keys:break
        for k in keys:
            selected.append(groups[k].pop())
            if len(selected)==count:break
    return selected


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('D:/GTA_passage_pairs_20261020'))
    p.add_argument('--delivery',type=Path,default=Path('E:/GTA_passage_training_20261020'))
    p.add_argument('--episode-splits',type=Path)
    p.add_argument('--previous-scenes',type=Path)
    args=p.parse_args();root=args.root;training=root/'training';mix=root/'mixture';mix.mkdir(exist_ok=True)
    points={r['point_id']:r for r in map(orjson.loads,(root/'points.jsonl').read_bytes().splitlines())}
    splitmap=orjson.loads(Path('D:/GTA_memory_release_20261018/DELIVERY.json').read_bytes())['report']['episode_split_map']
    if args.episode_splits:splitmap.update(orjson.loads(args.episode_splits.read_bytes()))
    rows=[];all_rows=[];gallery=[]
    for path in sorted((training/'samples').glob('*/sample.json')):
        sample=orjson.loads(path.read_bytes());key=sample['sample_id'];status=sample['training_status']
        all_rows.append({'sample_id':key,'status':status,'reason':sample.get('reason'),
                         'supplemental_target_review':sample.get('supplemental_target_review')})
        if status not in ('AUTO_REFERENCE_WITH_INHERITED_TARGET_QA','AUTO_REFERENCE_WITH_REVIEWED_TARGET_QA'):continue
        require_pair_review(sample)
        evidence=sample['reference'];pt=points[key];x,y,z=pt['camera_position'];age=evidence['age_sec'];angle=evidence['view_angle_deg']
        # Site keys control within-task sampling; they do not assert instance IDs.
        site=f'{int(x//40)}_{int(y//40)}_{int(z//8)}'
        row={'dataset':'scenes','sample_id':key,'sample_root':'samples/'+key,'task':'scene_background',
             'split':splitmap[sample['episode_id']],'episode_id':sample['episode_id'],
             'balance_group':sample['episode_id']+':'+site,'spatial_bin_40_units':site,
             'kind':evidence['kind'],'age_bin':'under10s' if age<10 else '10to30s' if age<30 else '30to120s' if age<120 else 'over120s',
             'angle_bin':'under15deg' if angle<15 else '15to45deg' if angle<45 else 'over45deg',
             'annotation_level':'automatic_reference_background_check_plus_target_review',
             'historical_reference_pair_reviewed':sample.get('supplemental_target_review',{}).get('historical_reference_pair_reviewed',False),
             'target_review_source':'supplemental_sampled_visual_review' if status=='AUTO_REFERENCE_WITH_REVIEWED_TARGET_QA' else 'inherited_review'}
        if row['kind']=='gap_unknown':continue
        rows.append(row)
        relative='samples/'+key
        gallery.append(f'<article><h2>{html.escape(key)}</h2><p>{html.escape(row["kind"])} | {age:.1f}s | '
            f'{evidence["camera_distance"]:.1f} units | {angle:.1f} degrees</p><a href="{relative}/comparison.jpg">'
            f'<img loading="lazy" src="{relative}/comparison.jpg"></a><details><summary>Target video</summary>'
            f'<video controls preload="none" src="{relative}/target_rgb.mp4"></video>'
            f'<video controls preload="none" src="{relative}/target_whitebox.mp4"></video></details></article>')
    (training/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Passage scene references</title>'
        '<style>body{font:16px system-ui;background:#fafafa;margin:24px}article{border-bottom:1px solid #bbb;padding:20px 0}'
        'img{max-width:100%;width:1000px}video{width:48%}h2{font-size:16px;overflow-wrap:anywhere}</style>'
        '<h1>Scene references from earlier passages</h1><p>Automatic background overlap evidence with sampled target QA. '
        'Per-sample review metadata records whether the historical reference pair was also visually inspected.</p>'+''.join(gallery),encoding='utf-8')
    write_rows(training/'samples.jsonl',rows);write_rows(training/'exclusion_ledger.jsonl',all_rows)
    (training/'splits').mkdir(exist_ok=True)
    for split in ('train','validation','test'):write_rows(training/'splits'/(split+'.jsonl'),[r for r in rows if r['split']==split])
    old=[]
    for split in ('train','validation','test'):
        old.extend(r for r in map(orjson.loads,Path('D:/GTA_memory_mixture_20261019',split+'.jsonl').read_bytes().splitlines()) if r['dataset']=='objects')
    previous=[]
    if args.previous_scenes:
        previous=[{**orjson.loads(line),'dataset':'previous_scenes'} for line in (args.previous_scenes/'samples.jsonl').read_bytes().splitlines()]
    combined,duplicate_count=merge_unique_rows(old+previous+rows)
    for split in ('train','validation','test'):write_rows(mix/(split+'.jsonl'),[r for r in combined if r['split']==split])
    fractions={'scene_background':.5,'object_person':.2,'object_vehicle':.2,'object_mixed':.1}
    pools={task:[r for r in combined if r['split']=='train' and r['task']==task] for task in fractions}
    scene_passages=[r for r in pools['scene_background'] if r['kind']!='nearby_history']
    scene_nearby=[r for r in pools['scene_background'] if r['kind']=='nearby_history']
    scene_capacity=len(scene_passages)+min(len(scene_nearby),len(scene_passages))
    size=int(min((scene_capacity if k=='scene_background' else len(pools[k]))/v for k,v in fractions.items()))
    plans=[]
    for seed in range(10):
        rng=random.Random(seed);plan=[]
        for task,fraction in fractions.items():
            quota=int(size*fraction)
            if task=='scene_background':
                nearby_quota=min(quota//2,len(scene_nearby))
                plan.extend(stratified_take(scene_passages,quota-nearby_quota,rng))
                plan.extend(stratified_take(scene_nearby,nearby_quota,rng))
            else:plan.extend(stratified_take(pools[task],quota,rng))
        rng.shuffle(plan);write_rows(mix/f'train_round_{seed:02d}.jsonl',plan)
        plans.append({'seed':seed,'count':len(plan),'tasks':dict(Counter(r['task'] for r in plan)),
                      'scene_kinds':dict(Counter(r['kind'] for r in plan if r['task']=='scene_background'))})
    report={'scene_count':len(rows),'old_object_count':len(old),'sample_status_counts':dict(Counter(r['status'] for r in all_rows)),
        'splits':{s:{key:dict(Counter(r[key] for r in rows if r['split']==s)) for key in ('kind','age_bin','angle_bin','episode_id','spatial_bin_40_units')}
                  for s in ('train','validation','test')},'mixture_fractions':fractions,'rounds':plans,
        'paid_api_calls':0,'annotation_limit':'Automatic new-reference QA; inherited or supplemental sampled target video QA. Not all new pair images visually reviewed.',
        'split_limit':'Episode-isolated, not guaranteed unseen-map-location split.',
        'quantity_limit':'Five-second trajectory sampling can miss nearby passages. Same passage provides correlated samples; use spatial groups.',
        'ratios_validated_by_training':False}
    report['previous_scene_count']=len(previous)
    report['combined_scene_count']=len(previous)+len(rows)
    report['combined_sample_count']=len(combined)
    report['duplicate_sample_ids_removed']=duplicate_count
    report['combined_task_counts']=dict(Counter(r['task'] for r in combined))
    report['combined_scene_count']=sum(r['task']=='scene_background' for r in combined)
    report['combined_scene_kinds']=dict(Counter(r['kind'] for r in combined if r['task']=='scene_background'))
    report['nearby_history_max_fraction_within_scene_round']=.5
    survey_path=root/'survey.json'
    if survey_path.exists():report['quantity_survey']=orjson.loads(survey_path.read_bytes())
    matches=[orjson.loads(line) for line in (root/'background_matches.jsonl').read_bytes().splitlines()]
    if {r['sample_id'] for r in all_rows}!={r['target_point'] for r in matches}:
        raise ValueError('Not every matched target has a completed export or exclusion decision')
    if any(r['status']=='VIDEO_QA_PENDING' for r in all_rows):
        raise ValueError('Finish pending target reviews before final packaging')
    (root/'training_progress.json').write_bytes(orjson.dumps(report['sample_status_counts'],option=orjson.OPT_INDENT_2))
    report['historical_candidate_rank_counts']=dict(Counter(str(r['reference_rank']) for r in matches))
    report['background_match_count']=len(matches)
    report['training_unique_spatial_bins']=len({r['balance_group'] for r in rows if r['split']=='train'})
    report['validation_scene_count']=sum(r['split']=='validation' for r in rows)
    report['test_scene_count']=sum(r['split']=='test' for r in rows)
    report['scene_kinds']=dict(Counter(r['kind'] for r in rows))
    report['scene_time_gaps']=dict(Counter(r['age_bin'] for r in rows))
    report['scene_view_angles']=dict(Counter(r['angle_bin'] for r in rows))
    report['target_review_sources']=dict(Counter(r['target_review_source'] for r in rows))
    report['historical_reference_pairs_visually_reviewed']=sum(r['historical_reference_pair_reviewed'] for r in rows)
    report['annotation_limit']='Automatic background matching plus sampled visual QA. Historical reference visual scope is recorded per sample. Not exhaustive temporal inspection.'
    (mix/'distribution_report.json').write_bytes(orjson.dumps(report,option=orjson.OPT_INDENT_2))
    shutil.copy2('memory/gta_mixed_dataset.py',mix/'dataset.py')
    roots={'objects':'D:/GTA_memory_release_20261018/dataset','scenes':str(training)}
    if args.previous_scenes:roots['previous_scenes']=str(args.previous_scenes)
    (mix/'roots.example.json').write_bytes(orjson.dumps(roots,option=orjson.OPT_INDENT_2))
    (mix/'episode_splits.json').write_bytes(orjson.dumps(splitmap,option=orjson.OPT_INDENT_2))
    code=mix/'processing_code';code.mkdir(exist_ok=True)
    for name in ('scripts/survey_gta_passage_pairs.py','scripts/match_gta_passage_pairs.py',
                 'scripts/export_gta_passage_training.py','scripts/finalize_gta_passage_training.py',
                 'scripts/test_gta_passage_pairs.py','scripts/review_gta_passage_targets.py',
                 'scripts/validate_gta_passage_training.py'):
        path=code/Path(name).name;shutil.copy2(name,path)
    shutil.copy2(root/'survey.json',mix/'quantity_survey.json')
    shutil.copy2('docs/gta_passage_expansion.md',mix/'processing_notes.md')
    if (root/'preparation_report.json').exists():
        shutil.copy2(root/'preparation_report.json',mix/'source_preparation_report.json')
        source_audit=mix/'source_qualification';source_audit.mkdir(exist_ok=True)
        for source in (root/'qualification').iterdir():
            if not source.is_dir():continue
            destination=source_audit/source.name;destination.mkdir(exist_ok=True)
            for name in ('summary.json','frame_errors.json'):
                shutil.copy2(source/name,destination/name)
        shutil.copy2('docs/gta_spool_passage_cleaning.md',mix/'incremental_processing_notes.md')
        for name in ('prepare_gta_spool_passages.py','run_gta_spool_cleaning.py','test_gta_spool_passages.py','index_gta_spool.py'):
            shutil.copy2(Path('scripts')/name,code/name)
        shutil.copy2('configs/gta_spool_episode_splits.json',mix/'incremental_episode_splits.json')
    for batch in root.glob('pending_audit*'):
        if (batch/'decisions.jsonl').exists():shutil.copy2(batch/'decisions.jsonl',mix/(batch.name+'_decisions.jsonl'))
    examples=mix/'reference_spot_checks';examples.mkdir(exist_ok=True)
    for image in (root/'audit').glob('sample_*.jpg'):shutil.copy2(image,examples/image.name)
    readme='''# Expanded scene-passage training data

Historical RGB/whitebox reference frame -> subsequent target whitebox video ->
aligned target RGB supervision. No object identity annotation is required for
scene samples. Existing 4,984 object samples remain in their previous release.

Candidate search tries 20 world units / 5 seconds / 90 degrees, then expands
unmatched targets to 30 units / 2 seconds / 110 degrees. Up to eight historical
references per target are tested. Return passages are preferred; within type,
recent past observations are tried first. Exact trajectory intersection and a
30-second minimum are not required. Continuous nearby history is labeled
separately from leave/return and cross-camera passage. Unknown gaps are excluded
from the training manifest.

Automatic SIFT background checks exclude projected dynamic actors and HUD
margins. This does not prove perfect correspondence. Target footage previously
rejected for RGB/whitebox quality is excluded. Supplemental target review checks
five paired video frames per target. Unreviewed targets remain pending.
All exported videos decode to 120 frames and undergo first/last-frame pixel
alignment checks where exact-index cached endpoints exist. Per-sample review
metadata records whether the historical reference pair was inspected alongside
the target. Temporal visual review is sampled, not exhaustive.

Extract scene archive as scenes root. Extract previous object archives as objects
root. MixedMemoryDataset from mixture/dataset.py consumes train_round_XX.jsonl
with those two roots. It separates conditions from target RGB supervision.
Use distribution_report.json for actual type, time, angle, location and split
coverage. Approximate 50/20/20/10 task fractions are an unvalidated starting
configuration. Nearby continuous-history examples are at most half of the scene
portion of a round; the full eligible pool is retained, not deleted to enforce
sampling proportions. Rounds sample without replacement and cycle spatial/instance
groups; all eligible data remains available. Repeated rounds reuse observations.
No model training was run, and shared GTA map sites can appear across episodes.
'''
    if previous:readme+='\nThis is an incremental scene archive. Previous scene samples remain in their original archive and use the previous_scenes root. Pass all three roots from roots.example.json to the loader. Mixed manifests include both scene releases. See distribution_report.json for counts. processing_notes.md describes the earlier release; incremental_processing_notes.md and source_preparation_report.json describe this increment.\n'
    (training/'README.md').write_text(readme,encoding='utf-8');(mix/'README.md').write_text(readme,encoding='utf-8')
    verification=validate(root)
    print('VALIDATED',verification['scene_samples_checked'],flush=True)
    args.delivery.mkdir(parents=True,exist_ok=True)
    eligible={r['sample_id'] for r in rows}
    for folder,name in [(training,'scene_passage_training.zip'),(mix,'mixed_manifests.zip')]:
        archive=args.delivery/name
        if archive.exists():raise ValueError('Existing delivered archive is not overwritten')
        partial=archive.with_suffix('.partial')
        with zipfile.ZipFile(partial,'w',zipfile.ZIP_STORED,allowZip64=True) as z:
            for path in folder.rglob('*'):
                if not path.is_file():continue
                relative=path.relative_to(folder)
                if folder==training and relative.parts[0]=='samples' and relative.parts[1] not in eligible:continue
                z.write(path,relative.as_posix())
        with zipfile.ZipFile(partial) as z:
            if z.testzip():raise ValueError('Corrupt archive')
        partial.rename(archive)
    shutil.copy2(mix/'distribution_report.json',args.delivery/'distribution_report.json')
    shutil.copy2(training/'README.md',args.delivery/'README.md')
    shutil.copy2(mix/'verification.json',args.delivery/'verification.json')
    (args.delivery/'DELIVERY.json').write_bytes(orjson.dumps({'status':'PACKAGED_AND_VERIFIED',
        'scene_count':len(rows),'old_object_count':len(old),'old_objects_shipped_separately':True,
        'previous_scene_count':len(previous),'combined_scene_count':len(rows)+len(previous),
        'archives':[{ 'name':p.name,'bytes':p.stat().st_size} for p in args.delivery.glob('*.zip')],
        'paid_api_calls':0,'model_training_run':False},option=orjson.OPT_INDENT_2))
    print(orjson.dumps({'scene_count':len(rows),'round0':plans[0],'delivery':str(args.delivery)},option=orjson.OPT_INDENT_2).decode())


if __name__=='__main__':main()
