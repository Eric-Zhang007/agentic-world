"""Collect remote geometry audits and fully decode local videos before review."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

from memory.gta_release_media import open_video


def commit(args):
    row,root=args;root=Path(root)
    key=row['episode_id']+'_'+row['agent_id']+'_'+row['segment'];out=root/'qualification'/key
    if (out/'summary.json').exists():return
    source=Path(row['episode_root']);manifest=json.loads(Path(row['source_manifest']).read_text())
    counts={}
    existing=root/'media_decode'/(key+'.json')
    if existing.exists():counts=json.loads(existing.read_text())['decoded_frames']
    for kind in ('rgb_video','proxy_video'):
        if kind in counts:continue
        cap=open_video(source/manifest[kind]);count=0
        try:
            while cap.read()[0]:count+=1
        finally:cap.release()
        counts[kind]=count
    frames=[json.loads(s) for s in (Path(row['source_manifest']).parent/'frames.jsonl').read_text().splitlines()]
    summary={'source':row,'segment_key':key,'decoded_frames':counts}
    if counts['proxy_video']!=len(frames) or any(f.get('rgb_decode_index',-1)>=counts['rgb_video'] for f in frames):
        summary['status']='EXCLUDED_MEDIA_COUNT_MISMATCH'
        # Do not feed geometry-only candidates from a failed media pair into review.
        (out/'excluded_geometry_windows.jsonl').write_bytes((out/'windows.jsonl').read_bytes())
        (out/'windows.jsonl').write_text('')
    else:
        windows=[json.loads(s) for s in (out/'windows.jsonl').read_text().splitlines()]
        summary.update(status='QUALIFIED',windows=dict(Counter(w['status'] for w in windows)),
                       invalid_frames=len(json.loads((out/'frame_errors.json').read_text())))
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('LOCAL_DECODE_VERIFIED',key,counts,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',type=Path,required=True)
    args=parser.parse_args();root=args.release
    rows=[json.loads(s) for s in (root/'source_ledger.jsonl').read_text().splitlines()]
    lookup={r['episode_id']+'_'+r['agent_id']+'_'+r['segment']:r for r in rows if r['status']=='PENDING'}
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]));from server_exec import connect
    remote='/mnt/hdd/gta_release_20261018';client=connect()
    try:
        with client.open_sftp() as sftp,ThreadPoolExecutor(max_workers=3) as pool:
            futures={};downloaded=set()
            while True:
                try:names=sftp.listdir(remote+'/metadata_results')
                except FileNotFoundError:names=[]
                for key in names:
                    if key in downloaded:continue
                    local=root/'qualification'/key
                    if (local/'summary.json').exists():downloaded.add(key);continue
                    if not (root/'media_decode'/(key+'.json')).exists():continue
                    try:sftp.stat(remote+'/metadata_results/'+key+'/complete.json')
                    except FileNotFoundError:continue
                    local.mkdir(parents=True,exist_ok=True)
                    for name in ('windows.jsonl','frame_errors.json','complete.json'):
                        sftp.get(remote+'/metadata_results/'+key+'/'+name,str(local/name))
                    downloaded.add(key);futures[key]=pool.submit(commit,(lookup[key],str(root)))
                for key,future in list(futures.items()):
                    if future.done():future.result();del futures[key]
                try:sftp.stat(remote+'/metadata_complete.json');complete=True
                except FileNotFoundError:complete=False
                if complete and not futures:
                    missing=[key for key in lookup if not (root/'qualification'/key/'summary.json').exists()]
                    if not missing:break
                time.sleep(10)
        summaries=[json.loads(p.read_text()) for p in (root/'qualification').glob('*/summary.json')]
        (root/'qualification_summary.json').write_text(json.dumps(summaries),encoding='utf-8')
    finally:client.close()


if __name__=='__main__':main()
