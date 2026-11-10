"""Qualify every available source segment, with a terminal ledger for each."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import cv2

from memory.gta_release import Records, qualify_windows
from memory.gta_release_media import open_video


def process(args):
    row, root = args
    root = Path(root)
    key = row['episode_id']+'_'+row['agent_id']+'_'+row['segment']
    out = root/'qualification'/key
    if (out/'summary.json').exists():
        return json.loads((out/'summary.json').read_text())
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(row['source_manifest'])
    m = json.loads(manifest_path.read_text(encoding='utf-8'))
    source = Path(row['episode_root'])
    counts = {}
    # Decode every frame of both streams; a metadata-only count is not validation.
    for kind in ('rgb_video','proxy_video'):
        cap = open_video(source/m[kind])
        count = 0
        shape = None
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if shape is not None and shape != image.shape:
                cap.release()
                raise ValueError('Video resolution changes within segment')
            shape = image.shape
            count += 1
        cap.release()
        counts[kind] = count
    with (manifest_path.parent/'frames.jsonl').open(encoding='utf-8') as f:
        frames = [json.loads(s) for s in f if s.strip()]
    summary = {'source':row,'decoded_frames':counts,'segment_key':key}
    if counts['proxy_video'] != len(frames) or any(f.get('rgb_decode_index',-1)>=counts['rgb_video'] for f in frames):
        summary.update(status='EXCLUDED_MEDIA_COUNT_MISMATCH')
    else:
        records = Records(root/row['record_index'])
        try:
            windows, errors = qualify_windows(records,m,frames)
        finally:
            records.close()
        for i, window in enumerate(windows):
            window.update(sample_id=key+f'_w{i:03d}',source_manifest=row['source_manifest'],
                          episode_id=m['episode_id'],agent_id=m['agent_id'],segment=m['segment'],
                          episode_root=str(source),record_index=row['record_index'])
        (out/'windows.jsonl').write_text(''.join(json.dumps(w)+'\n' for w in windows),encoding='utf-8')
        (out/'frame_errors.json').write_text(json.dumps(errors),encoding='utf-8')
        summary.update(status='QUALIFIED', windows=dict(Counter(w['status'] for w in windows)), invalid_frames=len(errors))
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(key, summary.get('windows',summary['status']),flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--wait-indexes',action='store_true')
    args = parser.parse_args()
    rows = [json.loads(s) for s in (args.release/'source_ledger.jsonl').read_text().splitlines()]
    pending = [r for r in rows if r['status']=='PENDING']
    import sqlite3
    done = set()
    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while pending:
            available = []
            for index in {row['record_index'] for row in pending}-done:
                path = args.release/index
                if path.exists():
                    try:
                        with sqlite3.connect(f'file:{path.as_posix()}?mode=ro',uri=True,timeout=1) as db:
                            complete = db.execute("SELECT value FROM metadata WHERE key='complete'").fetchone()
                        if complete:
                            done.add(index)
                    except sqlite3.Error:
                        pass
            available=[row for row in pending if row['record_index'] in done]
            if not available:
                if not args.wait_indexes:
                    raise RuntimeError('Required indexes not complete')
                print('Waiting for source indexes',flush=True)
                time.sleep(20)
                continue
            for result in pool.map(process,[(r,str(args.release)) for r in available]):
                summaries.append(result)
            keys = {r['source_manifest'] for r in available}
            pending = [r for r in pending if r['source_manifest'] not in keys]
    (args.release/'qualification_summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
