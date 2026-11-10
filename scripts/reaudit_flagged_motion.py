"""Independent source-video timing diagnostics for metadata-flagged segments."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.reaudit_motion_timing import motion,affine_fit


def sources():
    result={}
    for base in (Path('E:/GTA_555_complete_rebuild_20261025/source/data'),Path('E:/GTA_555')):
        folders=list(base.iterdir())
        for parent in list(folders):
            if parent.is_dir() and not (parent/'manifest.json').exists():
                folders.extend(p for p in parent.glob('ep*') if p.is_dir())
        for folder in folders:
            path=folder/'manifest.json'
            if not path.is_file():
                continue
            try:
                episode=json.loads(path.read_text(encoding='utf-8')).get('episode_id')
            except (ValueError,OSError):
                continue
            if episode:
                result.setdefault(episode,[]).append(folder)
    return result


def run_one(item):
    out=Path(item['output'])
    out.mkdir(parents=True,exist_ok=True)
    report=out/'motion_timing.json'
    if report.exists():
        return json.loads(report.read_text())
    started=time.monotonic()
    result={k:v for k,v in item.items() if k!='output'}
    try:
        signals={kind:motion(item[kind],1440) for kind in ('rgb','proxy')}
        a,b=[np.array(signals[k],dtype=float) for k in ('rgb','proxy')]
        if min(len(a),len(b))<120:
            raise ValueError('Insufficient decoded frames')
        result.update(status='MEASURED_NOT_VISUALLY_ACCEPTED',affine_timing=affine_fit(a,b),
                      decoded_frames={k:len(v) for k,v in signals.items()})
        (out/'signals.json').write_text(json.dumps(signals,allow_nan=False))
    except Exception as exc:
        result.update(status='DIAGNOSTIC_FAILED',error=f'{type(exc).__name__}: {exc}')
    result['elapsed_seconds']=time.monotonic()-started
    report.write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--controls',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/dataset_reaudit/flagged_motion')
    args=p.parse_args()
    if args.controls:
        args.output=ROOT/'outputs/dataset_reaudit/control_motion'
    args.output.mkdir(parents=True,exist_ok=True)
    rates=[json.loads(line) for line in (ROOT/'outputs/dataset_reaudit/metadata/source_clock_rates.jsonl').read_text().splitlines()]
    lookup=sources()
    pending=[]
    missing=[]
    control_sources={('ep_20260921_124218_450d','agent_000','segment_00000'),
                     ('ep_20261006_123152_abb1','agent_000','segment_00000'),
                     ('ep_20261006_132302_f26f','agent_000','segment_00000')}
    for row in rates:
        if args.controls and tuple(row['source']) not in control_sources:
            continue
        if not args.controls and not row['flagged']:
            continue
        episode,agent,segment=row['source']
        paths={}
        for folder in lookup.get(episode,[]):
            manifest=folder/'derived/segments'/agent/segment/'manifest.json'
            if not manifest.exists():
                continue
            info=json.loads(manifest.read_text())
            for kind,key in (('rgb','rgb_video'),('proxy','proxy_video')):
                path=folder/info[key]
                if path.exists():
                    paths[kind]=str(path)
        item={'source':row['source'],'sample_ids':row['sample_ids'],
              'metadata_rate':row['fitted_server_over_rgb_pts_slope'],
              'output':str(args.output/'_'.join(row['source'])),**paths}
        if len(paths)!=2:
            missing.append({**item,'status':'SOURCE_MEDIA_NOT_LOCATED'})
        else:
            pending.append(item)
    (args.output/'inputs.json').write_text(json.dumps({'pending':pending,'missing':missing},indent=2))
    results=list(missing)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_one,item) for item in pending]
        for future in as_completed(futures):
            result=future.result()
            results.append(result)
            (args.output/'progress.json').write_text(json.dumps({'total':len(pending)+len(missing),
                'completed':len(results),'results':results},indent=2))
            print(json.dumps(result),flush=True)
    (args.output/'summary.json').write_text(json.dumps({'completed':len(results),'results':results,
        'scope':'Source motion diagnostics only; no automatic repair or visual acceptance.'},indent=2))


if __name__=='__main__':
    main()
