"""Finish the release after concurrent review/export workers have drained."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time
import shutil
import zipfile


def run(module,root,*extra):
    print('RUN',module,flush=True)
    subprocess.run([sys.executable,'-u','-m',module,'--release',str(root),*extra],check=True)


def copy_archive(source,target):
    if target.exists():
        with zipfile.ZipFile(source) as original,zipfile.ZipFile(target) as copied:
            signature=lambda archive:[(i.filename,i.file_size,i.CRC) for i in archive.infolist()]
            if source.stat().st_size!=target.stat().st_size or signature(original)!=signature(copied) or copied.testzip():
                raise ValueError('Existing delivery archive differs: '+str(target))
        return
    temporary=target.with_name(target.name+'.copying')
    shutil.copyfile(source,temporary)
    if temporary.stat().st_size!=source.stat().st_size:raise ValueError('Archive copy size mismatch')
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip():raise ValueError('Delivery archive CRC failure')
    temporary.rename(target)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',type=Path,required=True)
    args=parser.parse_args();root=args.release
    while not all((root/name).exists() for name in ('qualification_summary.json','visual_review_complete.json','export_complete.json','sample_validation_complete.json')):
        time.sleep(20)
    # The first-pass workers have exited. Retrying decisions cannot race with them.
    for attempt in range(3):
        failed=[p for p in (root/'visual_review').glob('*/decision.json') if json.loads(p.read_text()).get('status')=='REVIEW_FAILED']
        if not failed:break
        print('RETRY_ROUND',attempt+1,'failed_requests',len(failed),flush=True)
        run('scripts.retry_gta_release_reviews',root)
        run('scripts.export_gta_release',root,'--workers','3')
    failed=[p.parent.name for p in (root/'visual_review').glob('*/decision.json')
            if json.loads(p.read_text()).get('status')=='REVIEW_FAILED']
    if failed:
        (root/'unresolved_reviews.json').write_text(json.dumps(failed,indent=2),encoding='utf-8')
        raise ValueError(f'{len(failed)} reviews remain unavailable; formal delivery is blocked')
    run('scripts.repair_gta_release_timing',root)
    run('scripts.repair_gta_conditioning',root)
    run('scripts.validate_gta_release_samples',root,'--workers','3')
    counts=Counter(json.loads(p.read_text())['status'] for p in (root/'sample_validation').glob('*.json'))
    print('FINAL_SAMPLE_VALIDATION',dict(counts),flush=True)
    if counts.get('FAILED'):raise ValueError('Samples require repair before packaging')
    run('scripts.finalize_gta_release',root)
    delivery=Path('E:/GTA_memory_delivery_20261019')
    delivery.mkdir(exist_ok=True)
    metadata=json.loads((root/'DELIVERY.json').read_text())
    archives=[row['file'] for row in metadata['archives']]+['release_manifest.zip']
    for name in archives:
        target=delivery/name
        copy_archive(root/name,target)
        print('DELIVERED_ARCHIVE',name,target.stat().st_size,flush=True)
    for name in ('README_zh.md','acceptance_report.json'):
        shutil.copyfile(root/'dataset'/name,delivery/name)
    shutil.copyfile(root/'DELIVERY.json',delivery/'DELIVERY.json')
    print('COMPLETE_DELIVERY',str(delivery),flush=True)


if __name__=='__main__':main()
