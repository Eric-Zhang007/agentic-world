"""Validate committed exports while the remaining samples are being processed."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from scripts.finalize_gta_release import sample_files,validate_release_sample,VALIDATOR_VERSION


def check(path,release):
    out=release/'sample_validation'/(path.parent.name+'.json')
    review=release/'visual_review'/path.parent.name/'decision.json'
    files=sample_files(path.parent)
    review_stat=[review.stat().st_size,review.stat().st_mtime_ns]
    prior=json.loads(out.read_text()) if out.exists() else {}
    if prior.get('validator_version')==VALIDATOR_VERSION and prior.get('files')==files and prior.get('review_stat')==review_stat:return
    try:
        validate_release_sample(path,release)
        result={'status':'PASSED'}
    except Exception as error:
        result={'status':'FAILED','error':str(error)}
    if files!=sample_files(path.parent):raise RuntimeError('Export changed during validation')
    result.update(files=files,review_stat=review_stat,validator_version=VALIDATOR_VERSION)
    out.write_text(json.dumps(result),encoding='utf-8')
    print(path.parent.name,result['status'],result.get('error',''),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--wait-export',action='store_true')
    args=parser.parse_args();root=args.release
    (root/'sample_validation').mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while True:
            paths=[]
            for path in (root/'dataset/samples').glob('*/sample.json'):
                verdict=root/'export_verdicts'/(path.parent.name+'.json')
                if verdict.exists() and json.loads(verdict.read_text()).get('status')=='EXPORTED':paths.append(path)
            list(pool.map(lambda path:check(path,root),paths))
            if not args.wait_export or (root/'export_complete.json').exists():break
            time.sleep(30)
    (root/'sample_validation_complete.json').write_text(json.dumps({'validator_version':VALIDATOR_VERSION}),encoding='utf-8')


if __name__=='__main__':main()
