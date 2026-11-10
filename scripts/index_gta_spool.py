"""Index raw observations without copying payloads or inventing lifecycle facts."""
import argparse
from collections import Counter
from contextlib import closing
from pathlib import Path
import sqlite3
import orjson
from scripts.build_gta_release_index import pack
from memory.gta_release import Records


def build(root, destination):
    if destination.exists():
        with closing(sqlite3.connect(destination)) as db:
            done = db.execute("SELECT value FROM metadata WHERE key='complete'").fetchone()
        if done:
            return orjson.loads(done[0])
        raise ValueError('Incomplete index requires investigation: ' + str(destination))
    db = sqlite3.connect(destination)
    db.executescript('''CREATE TABLE camera(agent,domain,seq,data);
        CREATE TABLE world(domain,seq,data); CREATE TABLE entity(id,data);
        CREATE TABLE metadata(key PRIMARY KEY,value);
        CREATE TABLE source_files(path PRIMARY KEY,bytes,mtime_ns,records);
        CREATE TABLE observed_entities(id,kind,samples,first_time,last_time,PRIMARY KEY(id,kind));''')
    counts = Counter(); entities = {}; examples = []
    paths = [(p, 'camera') for p in sorted((root/'.spool/agents').glob('*/client.jsonl'))]
    paths.append((root/'.spool/world_state.jsonl', 'world'))
    for path, kind in paths:
        before = path.stat(); batch = []; n = 0
        with path.open('rb') as source:
            while True:
                offset = source.tell(); line = source.readline()
                if not line: break
                row = orjson.loads(line)
                pointer = pack({'_jsonl':str(path), 'offset':offset, 'length':len(line)})
                if kind == 'camera':
                    key = (row['agent_id'],row['client_clock_domain'],row['client_sample_seq'])
                    observed = (row.get('scene_observation') or {}).get('entities', [])
                    timestamp = row.get('server_time_estimate_ns')
                    counts['camera_partial_observations'] += (row.get('scene_observation') or {}).get('completeness') == 'partial'
                else:
                    key = (row['server_clock_domain'],row['sample_seq'])
                    observed = [v for category in ('agents','vehicles','peds','objects') for v in row.get(category,[])]
                    timestamp = row.get('server_monotonic_ns')
                batch.append((*key,pointer)); n += 1
                for obj in observed:
                    oid = obj.get('dataset_entity_id')
                    if oid is None:
                        counts[kind+'_unidentified_observations'] += 1; continue
                    ekey = (str(oid),kind)
                    value = entities.setdefault(ekey,[0,None,None]); value[0] += 1
                    if timestamp is not None:
                        value[1] = timestamp if value[1] is None else min(value[1],timestamp)
                        value[2] = timestamp if value[2] is None else max(value[2],timestamp)
                if n == 1: examples.append((kind,key,row['episode_id']))
                if len(batch) >= 512:
                    db.executemany('INSERT INTO '+kind+' VALUES ('+','.join('?' for _ in batch[0])+')',batch); batch.clear()
                if n % 20000 == 0: print(root.name,kind,path.parent.name,n,flush=True)
        if batch: db.executemany('INSERT INTO '+kind+' VALUES ('+','.join('?' for _ in batch[0])+')',batch)
        if (before.st_size,before.st_mtime_ns)!=(path.stat().st_size,path.stat().st_mtime_ns):raise ValueError('Source changed')
        db.execute('INSERT INTO source_files VALUES (?,?,?,?)',(str(path),before.st_size,before.st_mtime_ns,n))
        counts[kind] += n; db.commit(); print(root.name,kind,'indexed',n,flush=True)
    db.executemany('INSERT INTO observed_entities VALUES (?,?,?,?,?)',[(oid,kind,*v) for (oid,kind),v in entities.items()])
    db.executescript('CREATE INDEX camera_key ON camera(agent,domain,seq); CREATE INDEX world_key ON world(domain,seq); CREATE INDEX entity_key ON entity(id);')
    counts['distinct_observed_entity_ids'] = len({k[0] for k in entities})
    counts['authoritative_lifecycle_records'] = 0
    db.commit(); db.close()
    records = Records(destination)
    for kind,key,episode in examples:
        payload = getattr(records,kind)(*key)
        assert payload['episode_id'] == episode
    records.close()
    with closing(sqlite3.connect(destination)) as db:
        db.execute('INSERT INTO metadata VALUES (?,?)',('complete',orjson.dumps(dict(counts)).decode()))
        db.commit()
    return dict(counts)


def audit_frames(root, index, output):
    records = Records(index); summary = Counter(); ledger = []
    for manifest in sorted(root.glob('derived/segments/*/*/manifest.json')):
        m = orjson.loads(manifest.read_bytes()); stats = Counter(); tested = 0
        with (manifest.parent/'frames.jsonl').open('rb') as source:
            for line in source:
                r = orjson.loads(line); stats['frames'] += 1
                stats['frames_with_static_candidates'] += r.get('static_candidates',0)>0
                stats['static_candidate_frame_occurrences'] += r.get('static_candidates',0)
                stats['renderer_entity_frame_occurrences'] += len(r.get('entity_ids',[]))
                interp = r.get('camera_interpolation')
                if not interp:
                    stats['frames_without_camera_alignment'] += 1; continue
                if r['proxy_frame_index'] % 120: continue
                try:
                    cameras = [records.camera(m['agent_id'],r['client_clock_domain'],interp[k]) for k in ('lower_client_sample_seq','upper_client_sample_seq')]
                    world = records.world(r['server_clock_domain'],r['world_capture_tick'])
                    assert all(c['episode_id']==m['episode_id'] and c['server_clock_domain']==r['server_clock_domain'] for c in cameras)
                    assert world['episode_id']==m['episode_id']
                    stats['sampled_frame_joins_passed'] += 1
                except (ValueError,KeyError,AssertionError):stats['sampled_frame_joins_failed'] += 1
                tested += 1
        ledger.append({'manifest':str(manifest),**stats}); summary.update(stats)
    records.close()
    output.write_bytes(b''.join(orjson.dumps(v)+b'\n' for v in ledger))
    return {'segments':len(ledger),**summary,'scope':'All frame metadata counted; camera/world joins tested every 120 frames. Candidate counts are not visibility labels.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('D:/GTA_spool_integration_20261020'));args=p.parse_args()
    args.output.mkdir(exist_ok=True)
    root=Path('E:/GTA_555'); roots=[root/name for name in ('0916','0917','09181','09182')]
    roots += list((root/'1007').glob('*bundle*'))
    report=[]
    for source in roots:
        index=args.output/(source.name+'.sqlite3')
        stats=build(source,index)
        audit=audit_frames(source,index,args.output/(source.name+'_frames.jsonl'))
        report.append({'root':str(source),'index':str(index),'records':stats,'frame_audit':audit})
        (args.output/'report.json').write_bytes(orjson.dumps(report,option=orjson.OPT_INDENT_2))
        print('EPISODE_COMPLETE',source.name,audit,flush=True)
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
