"""Index source row locations; avoid duplicating multi-gigabyte JSON payloads."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3

import orjson
import pyarrow.parquet as pq

from scripts.build_gta_release_index import pack


def build(episode,release):
    root=Path(episode['root']);destination=release/episode['index']
    if destination.exists():
        db=sqlite3.connect(destination)
        try:
            if db.execute("SELECT value FROM metadata WHERE key='complete'").fetchone():
                print(root.name,'already complete',flush=True);return
        finally:db.close()
        raise ValueError('Incomplete location index: '+str(destination))
    db=sqlite3.connect(destination)
    db.executescript('CREATE TABLE camera(agent,domain,seq,data); CREATE TABLE world(domain,seq,data); CREATE TABLE entity(id,data); CREATE TABLE metadata(key PRIMARY KEY,value);')
    counts={}
    for path in sorted((root/'agents').glob('*/camera.parquet')):
        pf=pq.ParquetFile(path);before=path.stat();count=0
        for group in range(pf.num_row_groups):
            rows=pf.read_row_group(group,columns=['agent_id','client_clock_domain','client_sample_seq']).to_pylist()
            db.executemany('INSERT INTO camera VALUES (?,?,?,?)',[(r['agent_id'],r['client_clock_domain'],r['client_sample_seq'],
                pack({'_parquet':str(path),'group':group,'row':i})) for i,r in enumerate(rows)])
            count+=len(rows)
        if (before.st_size,before.st_mtime_ns)!=(path.stat().st_size,path.stat().st_mtime_ns):raise ValueError('Source changed')
        counts[path.parent.name]=count;db.commit();print(root.name,path.parent.name,count,flush=True)
    path=root/'server/world_state.parquet';pf=pq.ParquetFile(path);before=path.stat()
    for group in range(pf.num_row_groups):
        rows=pf.read_row_group(group,columns=['sample_seq','payload_json']).to_pylist()
        values=[]
        for i,row in enumerate(rows):
            p=orjson.loads(row['payload_json'])
            values.append((p.get('server_clock_domain'),row['sample_seq'],pack({'_parquet':str(path),'group':group,'row':i})))
        db.executemany('INSERT INTO world VALUES (?,?,?)',values)
    if (before.st_size,before.st_mtime_ns)!=(path.stat().st_size,path.stat().st_mtime_ns):raise ValueError('Source changed')
    counts['world']=pf.metadata.num_rows;db.commit()
    path=root/'server/entities.parquet';pf=pq.ParquetFile(path);before=path.stat()
    for batch in pf.iter_batches(batch_size=1024,columns=['payload_json']):
        rows=[orjson.loads(r['payload_json']) for r in batch.to_pylist()]
        db.executemany('INSERT INTO entity VALUES (?,?)',[(str(r['dataset_entity_id']),pack(r)) for r in rows])
    if (before.st_size,before.st_mtime_ns)!=(path.stat().st_size,path.stat().st_mtime_ns):raise ValueError('Source changed')
    counts['entities']=pf.metadata.num_rows
    db.executescript('CREATE INDEX camera_key ON camera(agent,domain,seq); CREATE INDEX world_key ON world(domain,seq); CREATE INDEX entity_key ON entity(id);')
    db.execute('INSERT INTO metadata VALUES (?,?)',('complete',json.dumps(counts)));db.commit();db.close()
    print(root.name,'COMPLETE',counts,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',type=Path,required=True)
    args=parser.parse_args();root=args.release
    episodes=json.loads((root/'episodes.json').read_text())
    ledger=[json.loads(s) for s in (root/'source_ledger.jsonl').read_text().splitlines()]
    for episode in episodes:episode['index']=episode['index'].replace('episode_','locations_')
    for row in ledger:row['record_index']=row['record_index'].replace('episode_','locations_')
    (root/'episodes.json').write_text(json.dumps(episodes,indent=2),encoding='utf-8')
    (root/'source_ledger.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ledger),encoding='utf-8')
    eligible={r['record_index'] for r in ledger if r['status']=='PENDING'}
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda e:build(e,root),[e for e in episodes if e['index'] in eligible]))


if __name__=='__main__':main()
