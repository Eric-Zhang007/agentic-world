"""Decode every locally eligible source video; preserve measured media counts."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from memory.gta_release_media import decode_count


def work(row):
    root=Path('E:/GTA_memory_release_20261018')
    key=row['episode_id']+'_'+row['agent_id']+'_'+row['segment']
    out=root/'media_decode'/(key+'.json')
    if out.exists():return
    summary=root/'qualification'/key/'summary.json'
    if summary.exists():
        data=json.loads(summary.read_text())
        counts=data['decoded_frames']
    else:
        manifest=json.loads(Path(row['source_manifest']).read_text());source=Path(row['episode_root'])
        counts={}
        for kind in ('rgb_video','proxy_video'):
            path=source/manifest[kind];before=path.stat()
            count=decode_count(path)
            if (before.st_size,before.st_mtime_ns)!=(path.stat().st_size,path.stat().st_mtime_ns):raise ValueError('Source changed during decode')
            counts[kind]=count
    out.write_text(json.dumps({'decoded_frames':counts,'source_manifest':row['source_manifest']}))
    print('DECODED',key,counts,flush=True)


def main():
    root=Path('E:/GTA_memory_release_20261018');(root/'media_decode').mkdir(exist_ok=True)
    rows=[r for r in map(json.loads,(root/'source_ledger.jsonl').read_text().splitlines()) if r['status']=='PENDING']
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(work,rows))
    (root/'media_decode_complete.json').write_text(json.dumps({'segments':len(rows)}))


if __name__=='__main__':main()
