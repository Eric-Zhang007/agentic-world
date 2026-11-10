"""Assemble the accepted cumulative pool into one portable, verified archive."""
import argparse
import json
from collections import Counter
from pathlib import Path
import shutil
import tempfile
import zipfile

from memory.gta_mixed_dataset import MixedMemoryDataset


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--auxiliary-roots', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    mixture = args.mixture
    output = args.output
    stage = output / 'package_metadata'
    archive = output / 'GTA_memory_training_unified.zip'
    if archive.exists():
        raise FileExistsError(archive)
    stage.mkdir(parents=True, exist_ok=False)
    roots = {k: Path(v) for k, v in json.loads((mixture/'roots.example.json').read_text()).items()}
    source_rows = sum((read_rows(mixture/(split+'.jsonl')) for split in ('train','validation','test')), [])
    ids = [r['sample_id'] for r in source_rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Expected nonempty manifests with globally unique sample IDs')
    sources = {r['sample_id']: roots[r['dataset']]/r['sample_root'] for r in source_rows}
    for folder in sources.values():
        assert (folder/'sample.json').is_file(), folder
    manifests = stage/'manifests'
    manifests.mkdir(exist_ok=True)
    round_names = [p.stem for p in sorted(mixture.glob('train_round_*.jsonl'))]
    for name in ['train','validation','test'] + round_names:
        rows = read_rows(mixture/(name+'.jsonl'))
        for row in rows:
            row['source_release'] = row['dataset']
            row['dataset'] = 'data'
        (manifests/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    shutil.copy2('memory/gta_mixed_dataset.py',stage/'dataset_core.py')
    for name in ('dataset.py','DATA_FORMAT.md','requirements.txt','example.py'):
        shutil.copy2(Path('docs/gta_unified_delivery')/name,stage/name)
    shutil.copy2('docs/gta_audited_upload_run.md', stage/'REVIEW_AND_SAMPLING.md')
    shutil.copy2('docs/gta_training_acceptance.md', stage/'ACCEPTANCE_SCOPE.md')
    provenance=stage/'provenance'
    provenance.mkdir(exist_ok=True)
    for key, root in roots.items():
        dest=provenance/key
        dest.mkdir(exist_ok=True)
        for name in ('acceptance_report.json','delivery_verification.json','exclusion_ledger.jsonl','source_ledger.jsonl'):
            if (root/name).exists():shutil.copy2(root/name,dest/name)
    audit=provenance/'cumulative_audit'
    shutil.copytree(mixture,audit,dirs_exist_ok=True,ignore=shutil.ignore_patterns('*.jsonl','dataset.py','roots.example.json','README.md'))
    for path in mixture.glob('pending_audit_*_decisions.jsonl'):
        shutil.copy2(path,audit/path.name)
    if (mixture/'exclusion_ledger.jsonl').exists():
        shutil.copy2(mixture/'exclusion_ledger.jsonl',audit/'exclusion_ledger.jsonl')
    windows = set()
    unidentified_windows = 0
    for row in source_rows:
        # Match the existing exporter/validator's window naming contract.
        try:
            if row['task'] == 'scene_background':
                segment, frame = row['sample_id'].rsplit('_f', 1)
                start = int(frame)
            else:
                segment, window = row['sample_id'].rsplit('_w', 1)
                start = (int(window) + 1) * 120
            windows.add((segment, start))
        except (ValueError, KeyError):
            unidentified_windows += 1
    report={
        'records':len(source_rows), 'tasks':dict(Counter(r['task'] for r in source_rows)),
        'splits':dict(Counter(r['split'] for r in source_rows)),
        'unique_target_windows':None if unidentified_windows else len(windows),
        'reused_target_windows':None if unidentified_windows else len(source_rows)-len(windows),
        'records_without_explicit_target_window':unidentified_windows,
        'source_release_counts':dict(Counter(r['dataset'] for r in source_rows)),
        'format':'Single data root, relative sample paths; no previous archives required',
        'training_run':False,
        'packaging_external_review_calls':0,
        'upstream_review_api_usage':'See per-sample review provenance; not measured by packaging',
        'scene_review_origins':dict(Counter(r.get('review_origin','legacy_recorded_qa')
            for r in source_rows if r['task']=='scene_background')),
    }
    write_json(stage/'DATASET_REPORT.json',report)
    (stage/'README.md').write_text(
        '# GTA reference-memory dataset\n\n'
        f'This archive contains {len(source_rows)} task records in one samples directory. '
        'No previous archive is required. Exact task and split counts are in DATASET_REPORT.json.\n\n'
        'Scene tasks use an earlier RGB/whitebox frame pair as reference, a later whitebox '
        'video as condition, and its synchronized RGB video as supervision. Object tasks '
        'also provide object-reference associations. Scene tasks do not claim object identities.\n\n'
        'Install requirements.txt and run example.py. DATA_FORMAT.md describes the loader. '
        'manifests/train.jsonl, validation.jsonl and test.jsonl contain the full pools. '
        'Optional train_round files are alternative sampling plans, not additional examples.\n\n'
        'Use the balanced rounds for the supplied 50% background, 20% person, '
        '20% vehicle and 10% mixed-object sampling recipe. The full pool is not '
        'balanced to those ratios. Each round caps scene samples at two per site; '
        'different rounds can reuse records. REVIEW_AND_SAMPLING.md gives counts '
        'and coverage limitations.\n\n'
        'VERIFICATION.json covers archive integrity and representative loader checks. '
        'Visual review scope remains the scope recorded in sample metadata and provenance; '
        'packaging does not approve new samples or validate model training. '
        'Coarse geometry and estimated masks are not exact reconstructions.\n\n'
        'Final inclusion is defined by manifests and provenance/cumulative_audit, '
        'including its per-sample scene_audit decisions and exclusion ledger. '
        'Original sample.json training_status values are preserved source metadata '
        'and can still say VIDEO_QA_PENDING. Earlier API and direct local reviews '
        'are distinguished in the manifest review_origin field.\n\n'
        'Optional auxiliary_observations/<sample_id>.json preserves recorded action, '
        'camera, world and sampled state metadata for scenes where available. '
        'Target-time outcomes are audit/sampling annotations, not generation inputs. '
        'Missing auxiliary files mean unavailable annotation, not an empty world. '
        'The realized mixture and scene-memory audit evidence are under provenance.\n', encoding='utf-8')
    # Exercise the remapped manifests against the real source folders without copying media.
    examples={}
    for row in source_rows:
        examples.setdefault((row['dataset'],row['task'],row['split'],row.get('kind')),row)
    partial=archive.with_suffix('.partial')
    print(f'PACKING {len(sources)} samples',flush=True)
    def add_file(z, path, name):
        compression = zipfile.ZIP_DEFLATED if path.suffix.lower() in ('.json','.jsonl','.txt','.md','.py','.html') else zipfile.ZIP_STORED
        z.write(path,name,compress_type=compression,compresslevel=1 if compression==zipfile.ZIP_DEFLATED else None)

    with zipfile.ZipFile(partial,'w',zipfile.ZIP_STORED,allowZip64=True) as z:
        for path in sorted(stage.rglob('*')):
            if path.is_file():add_file(z,path,path.relative_to(stage).as_posix())
        for i,(sample_id,folder) in enumerate(sources.items(),1):
            for path in sorted(folder.rglob('*')):
                if path.is_file():add_file(z,path,('samples/'+sample_id+'/'+path.relative_to(folder).as_posix()))
            auxiliary=[root/(sample_id+'.json') for root in args.auxiliary_roots if (root/(sample_id+'.json')).is_file()]
            if len(auxiliary)>1:raise ValueError('Ambiguous auxiliary metadata: '+sample_id)
            if auxiliary:add_file(z,auxiliary[0],'auxiliary_observations/'+sample_id+'.json')
            if i%250==0:print(f'PACKED {i}/{len(sources)}',flush=True)
    print('VERIFYING archive CRC and portable loader',flush=True)
    with zipfile.ZipFile(partial) as z:
        names=z.namelist()
        assert len(names)==len(set(names)), 'Duplicate archive paths'
        assert z.testzip() is None
        assert sum(n.startswith('samples/') and n.endswith('/sample.json') for n in names)==len(source_rows)
        with tempfile.TemporaryDirectory(prefix='gta_unified_verify_',dir=output) as tmp:
            tmp=Path(tmp)
            for name in names:
                if not name.startswith('samples/') or name.split('/')[1] in {r['sample_id'] for r in examples.values()}:
                    z.extract(name,tmp)
            checked=[]
            for split in ('train','validation','test'):
                ds=MixedMemoryDataset(tmp/'manifests'/(split+'.jsonl'),{'data':tmp},temporal_indices=[0,30,60,90,119])
                chosen={r['sample_id'] for r in examples.values() if r['split']==split}
                for i,row in enumerate(ds.rows):
                    if row['sample_id'] not in chosen:continue
                    sample=ds[i]
                    assert sample['conditions']['target_whitebox'].shape==sample['supervision']['target_rgb'].shape
                    assert 'target_rgb' not in sample['conditions']
                    checked.append(row['sample_id'])
    verification={'passed':True,'archive_crc_all_entries':True,'unique_archive_paths':True,
                  'sample_records':len(source_rows),'portable_loader_examples':checked,
                  'scope':'All archived bytes checked; representative loader examples across release/task/split/kind. Prior visual QA scope unchanged.'}
    write_json(stage/'VERIFICATION.json',verification)
    with zipfile.ZipFile(partial,'a',zipfile.ZIP_STORED) as z:
        z.write(stage/'VERIFICATION.json','VERIFICATION.json')
    with zipfile.ZipFile(partial) as z:
        assert json.loads(z.read('VERIFICATION.json'))['passed']
    partial.rename(archive)
    write_json(output/'DELIVERY.json',{'status':'PACKAGED_AND_VERIFIED','archive':archive.name,'bytes':archive.stat().st_size,**report})
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'records':len(source_rows),'loader_examples':len(checked)}),flush=True)


if __name__=='__main__':main()
