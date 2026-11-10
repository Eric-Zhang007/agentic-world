"""Remote CPU-only geometry qualification; source videos stay on the local host."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import time

from memory.gta_release import Records,qualify_windows
from scripts.build_gta_release_index import unpack
import orjson


class MappedRecords(Records):
    def __init__(self,index,local_root,remote_root):
        super().__init__(index)
        self.local_root=local_root.replace('\\','/')
        self.remote_root=Path(remote_root)

    def payload(self,value):
        pointer=unpack(value)
        if '_parquet' not in pointer:return pointer
        path=pointer['_parquet'].replace('\\','/')
        if not path.startswith(self.local_root+'/'):raise ValueError('Source path outside mapped episode')
        source=self.remote_root/path[len(self.local_root)+1:]
        return orjson.loads(self.row_group(str(source),pointer['group'])[pointer['row']])


def work(args):
    row,root=args;root=Path(root)
    key=row['episode_id']+'_'+row['agent_id']+'_'+row['segment']
    out=root/'metadata_results'/key
    if (out/'complete.json').exists():return key
    out.mkdir(parents=True,exist_ok=True)
    episode=root/'sources'/Path(row['record_index']).stem
    segment=episode/'derived/segments'/row['agent_id']/row['segment']
    manifest=json.loads((segment/'manifest.json').read_text())
    frames=[json.loads(s) for s in (segment/'frames.jsonl').read_text().splitlines()]
    records=MappedRecords(root/'indexes'/row['record_index'],row['episode_root'],episode)
    start=time.monotonic()
    try:windows,errors=qualify_windows(records,manifest,frames)
    finally:records.close()
    for i,w in enumerate(windows):
        w.update(sample_id=key+f'_w{i:03d}',source_manifest=row['source_manifest'],
                 episode_id=row['episode_id'],agent_id=row['agent_id'],segment=row['segment'],
                 episode_root=row['episode_root'],record_index=row['record_index'])
    (out/'windows.jsonl').write_text(''.join(json.dumps(w)+'\n' for w in windows))
    (out/'frame_errors.json').write_text(json.dumps(errors))
    (out/'complete.json').write_text(json.dumps({'seconds':time.monotonic()-start,'invalid_frames':len(errors)}))
    return key


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args()
    rows=[json.loads(s) for s in (args.root/'source_ledger.jsonl').read_text().splitlines() if s.strip()]
    completed=set(json.loads((args.root/'already_qualified.json').read_text()))
    pending=[r for r in rows if r['status']=='PENDING' and r['source_manifest'] not in completed]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(work,(r,str(args.root))) for r in pending]
        for future in as_completed(jobs):print('QUALIFIED',future.result(),flush=True)
    (args.root/'metadata_complete.json').write_text(json.dumps({'segments':len(pending)}))


if __name__=='__main__':main()
