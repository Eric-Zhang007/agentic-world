"""Exercise the patch on copies of actual delivered records and video files."""
import argparse
import json
import shutil
import zipfile
from pathlib import Path
from apply_delivery_patch import apply, rows

ARCHIVE=Path('E:/GTA_555_complete_rebuild_20261025/unified_delivery/GTA_memory_training_unified.zip')
PACKAGE=Path('D:/GTA_memory_patch')
ROOT=Path('D:/GTA_memory_patch_test/dataset')

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    if a.prepare:
        audit=Path(__file__).resolve().parents[1]/'outputs/dataset_reaudit/review_ledger.jsonl'
        ids=[r['sample_id'] for r in rows(audit) if r['reasons']==['encoded_media_timing_or_decode_issue']]
        with zipfile.ZipFile(ARCHIVE) as z:
            names=[n for n in z.namelist() if n.startswith('manifests/') and n.endswith('.jsonl')]+['DATASET_REPORT.json']
            names += [f'samples/{sid}/{file}' for sid in ids for file in ('sample.json','target_rgb.mp4','target_whitebox.mp4')]
            for name in names:
                dest=ROOT/name
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True,exist_ok=True)
                with z.open(name) as src,dest.open('wb') as out:
                    shutil.copyfileobj(src,out)
        print(json.dumps({'prepared_actual_sample_copies':len(ids),'files':len(names)}))
        return
    report_path=PACKAGE/'payload/DATASET_REPORT.json'
    report=json.loads(report_path.read_text(encoding='utf-8'))
    if 'scene_review_origins' in report:
        report['original_package_scene_review_origins']=report.pop('scene_review_origins')
        report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        spec=json.loads((PACKAGE/'patch.json').read_text(encoding='utf-8'))
        spec['summary']=report
        for f in spec['files']:
            if f['path']=='DATASET_REPORT.json':f['bytes']=report_path.stat().st_size
        (PACKAGE/'patch.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
    before=ARCHIVE.stat()
    checks=[apply(ROOT,PACKAGE),apply(ROOT,PACKAGE,True),apply(ROOT,PACKAGE,True)]
    spec=json.loads((PACKAGE/'patch.json').read_text(encoding='utf-8'))
    excluded=set(spec['excluded_ids'])
    for name in spec['original_manifest_records']:
        original=rows(ROOT/'patch_backup'/name)
        assert original==spec['original_manifest_records'][name]
        assert rows(ROOT/name)==[r for r in original if r['sample_id'] not in excluded]
    with zipfile.ZipFile(ARCHIVE) as z:
        for f in spec['files']:
            name=f['path']
            if name.endswith('.mp4'):
                assert (ROOT/'patch_backup'/name).read_bytes()==z.read(name)
                assert (ROOT/name).read_bytes()==(PACKAGE/'payload'/name).read_bytes()
    after=ARCHIVE.stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    result={'checks':checks,'all_manifest_filters_exact':True,'all_original_video_backups_identical':True,'all_replacements_identical_to_verified_payload':True,'source_archive_unchanged':True,'source_archive_bytes':after.st_size}
    (PACKAGE/'integration_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
