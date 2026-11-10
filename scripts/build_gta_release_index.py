"""Snapshot every local segment and index complete source records for release QA."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import zlib
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
import orjson


def pack(value):
    return zlib.compress(orjson.dumps(value), 1)


def unpack(value):
    return orjson.loads(zlib.decompress(value))


def index_episode(root, output):
    db = sqlite3.connect(output)
    db.executescript('''CREATE TABLE IF NOT EXISTS camera(agent TEXT, domain TEXT, seq INTEGER, data BLOB);
        CREATE TABLE IF NOT EXISTS world(domain TEXT, seq INTEGER, data BLOB);
        CREATE TABLE IF NOT EXISTS entity(id TEXT, data BLOB);
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT);''')
    counts = Counter()
    for path in sorted((root/'agents').glob('*/camera.parquet')):
        before = path.stat()
        parquet=pq.ParquetFile(path)
        existing=db.execute('SELECT count(*) FROM camera WHERE agent=?',(path.parent.name,)).fetchone()[0]
        if existing:
            if existing!=parquet.metadata.num_rows:raise ValueError('Partial committed camera file')
            counts['camera']+=existing
            print(root.name,path.parent.name,'resume committed file',existing,flush=True)
            continue
        for batch in parquet.iter_batches(batch_size=256):
            values = []
            for row in batch.to_pylist():
                # Retain the source payload without parsing/re-encoding all entities.
                values.append((row['agent_id'], row['client_clock_domain'], row['client_sample_seq'],
                               zlib.compress(row['payload_json'].encode(),1)))
            db.executemany('INSERT INTO camera VALUES (?,?,?,?)', values)
            counts['camera'] += len(values)
        if (before.st_size, before.st_mtime_ns) != (path.stat().st_size, path.stat().st_mtime_ns):
            raise ValueError(f'Source changed during scan: {path}')
        db.commit()
        print(root.name, path.parent.name, dict(counts), flush=True)
    for kind, filename in [('world','world_state.parquet'), ('entity','entities.parquet')]:
        path = root/'server'/filename
        before = path.stat()
        parquet=pq.ParquetFile(path)
        existing=db.execute('SELECT count(*) FROM '+kind).fetchone()[0]
        if existing:
            if existing!=parquet.metadata.num_rows:raise ValueError('Partial committed '+kind+' file')
            counts[kind]+=existing
            continue
        for batch in parquet.iter_batches(batch_size=256):
            values = []
            for row in batch.to_pylist():
                p = orjson.loads(row['payload_json'])
                if kind == 'world':
                    values.append((p.get('server_clock_domain'), row['sample_seq'], pack(p)))
                else:
                    values.append((str(p['dataset_entity_id']), pack(p)))
            db.executemany('INSERT INTO '+kind+' VALUES ('+','.join('?' for _ in values[0])+')', values)
            counts[kind] += len(values)
        if (before.st_size, before.st_mtime_ns) != (path.stat().st_size, path.stat().st_mtime_ns):
            raise ValueError(f'Source changed during scan: {path}')
        db.commit()
    db.executescript('CREATE INDEX IF NOT EXISTS camera_key ON camera(agent,domain,seq); CREATE INDEX IF NOT EXISTS world_key ON world(domain,seq); CREATE INDEX IF NOT EXISTS entity_key ON entity(id);')
    db.execute('INSERT INTO metadata VALUES (?,?)', ('complete', json.dumps(dict(counts))))
    db.commit()
    db.close()
    print(root.name, 'COMPLETE', dict(counts), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--index-episode', type=int)
    parser.add_argument('--index-all', action='store_true')
    args = parser.parse_args()
    if args.index_all:
        episodes = json.loads((args.output/'episodes.json').read_text(encoding='utf-8'))
        ledger = [json.loads(s) for s in (args.output/'source_ledger.jsonl').read_text(encoding='utf-8').splitlines()]
        eligible = {r['record_index'] for r in ledger if r['status'] == 'PENDING'}
        def work(episode):
            if episode['index'] not in eligible:
                return
            destination = args.output/episode['index']
            if destination.exists():
                with sqlite3.connect(destination) as db:
                    done = db.execute("SELECT value FROM metadata WHERE key='complete'").fetchone()
                if done:
                    print(episode['root'], 'already indexed', flush=True)
                    return
                print(episode['root'],'resuming committed source files',flush=True)
            index_episode(Path(episode['root']), destination)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(work, episodes))
        return
    if args.index_episode is not None:
        episodes = json.loads((args.output/'episodes.json').read_text(encoding='utf-8'))
        episode = episodes[args.index_episode]
        index_episode(Path(episode['root']), args.output/episode['index'])
        return
    args.output.mkdir(parents=True, exist_ok=False)
    manifests = sorted(args.root.glob('**/derived/segments/*/*/manifest.json'))
    roots = sorted({p.parents[4] for p in manifests})
    episodes = []
    ledger = []
    for root in roots:
        files = [p for p in manifests if p.parents[4] == root]
        index = f'episode_{len(episodes):02d}.sqlite3'
        episode = {'root':str(root), 'index':index, 'segments':len(files)}
        episodes.append(episode)
        for path in files:
            record = {'source_manifest':str(path), 'episode_root':str(root), 'record_index':index}
            try:
                m = json.loads(path.read_text(encoding='utf-8'))
                record.update({k:m[k] for k in ('episode_id','agent_id','segment','frame_count')})
                required = [root/m[k] for k in ('rgb_video','proxy_video')]
                required += [root/'agents'/m['agent_id']/'camera.parquet', root/'server/world_state.parquet', root/'server/entities.parquet', path.parent/'frames.jsonl']
                record['missing_dependencies'] = [str(p) for p in required if not p.is_file()]
                record['status'] = 'EXCLUDED_MISSING_DEPENDENCIES' if record['missing_dependencies'] else 'PENDING'
                record['source_sizes'] = {str(p.relative_to(root)):p.stat().st_size for p in required if p.is_file()}
            except (OSError, KeyError, ValueError) as error:
                record.update(status='EXCLUDED_INVALID_MANIFEST', error=str(error))
            ledger.append(record)
    (args.output/'episodes.json').write_text(json.dumps(episodes, indent=2), encoding='utf-8')
    (args.output/'source_ledger.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ledger), encoding='utf-8')
    counts = dict(Counter(r['status'] for r in ledger))
    (args.output/'inventory_summary.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    print(json.dumps({'counts':counts,'indexable_episode_numbers':[i for i,e in enumerate(episodes) if any(r['record_index']==e['index'] and r['status']=='PENDING' for r in ledger)]}))


if __name__ == '__main__':
    main()
