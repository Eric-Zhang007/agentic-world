"""Retry failed requests without relabeling them as visual rejections."""
import argparse
import json
from pathlib import Path
import time
import yaml
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from scripts.review_gta_release import review_batch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',type=Path,required=True)
    args=parser.parse_args();root=args.release
    cfg=yaml.safe_load(Path('configs/default.yaml').read_text())['agent']
    failed_keys={p.parent.name for p in (root/'visual_review').glob('*/decision.json') if json.loads(p.read_text())['status']=='REVIEW_FAILED'}
    failed_keys.update(p.name for p in (root/'visual_review').iterdir()
                       if p.is_dir() and not (p/'decision.json').exists() and any(p.glob('failed_decision_*.json')))
    def pending():
        for p in sorted((root/'qualification').glob('*/windows.jsonl')):
            with p.open() as source:
                for line in source:
                    w=json.loads(line)
                    if w['sample_id'] in failed_keys:yield w
    def retry(window):
        key=window['sample_id'];path=root/'visual_review'/key/'decision.json'
        if path.exists():
            if json.loads(path.read_text())['status']!='REVIEW_FAILED':return
            path.rename(path.parent/('failed_decision_'+str(time.time_ns())+'.json'))
        verdict=root/'export_verdicts'/(key+'.json')
        if verdict.exists():
            history=root/'export_verdict_history';history.mkdir(exist_ok=True)
            verdict.rename(history/(key+'_'+str(time.time_ns())+'.json'))
        review_batch([window],root,cfg)
    windows=iter(pending())
    with ThreadPoolExecutor(max_workers=4) as pool:
        active=set()
        for _ in range(4):
            window=next(windows,None)
            if window is not None:active.add(pool.submit(retry,window))
        while active:
            finished,active=wait(active,return_when=FIRST_COMPLETED)
            for future in finished:
                future.result()
                window=next(windows,None)
                if window is not None:active.add(pool.submit(retry,window))
    print('Retries finished; rerun exporter and finalizer after reviewing failures')


if __name__=='__main__':main()
