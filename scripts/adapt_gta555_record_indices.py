"""Reuse frozen spool offsets with the existing training pipeline Records API.

Validate spool bytes against the source package before adapting index records.
No JSON payload duplication and no inferred entity lifecycle labels.
"""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import zlib


def adapt_episode(source, old_source, episode, reports, output):
    eid = episode['episode_id']
    final = output / (eid + '.sqlite3')
    receipt = output / (eid + '.json')
    if receipt.exists() and final.exists():
        return json.loads(receipt.read_text())
    chosen = {}
    for kind, spec in episode['spools'].items():
        expected = reports[eid, kind]
        candidates = [source / spec['source']]
        if old_source:
            candidates.append(old_source / Path(spec['source']).relative_to('data'))
        candidate = next((p for p in candidates if p.is_file() and p.stat().st_size == expected['signature'][1]), None)
        if candidate is None or not (source / spec['index']).is_file():
            return None
        chosen[kind] = candidate
    temp = output / (eid + '.sqlite3.partial')
    if temp.exists():
        temp.unlink()
    counts, sources = {}, {}
    with closing(sqlite3.connect(temp)) as db, db:
        db.executescript('CREATE TABLE camera(agent,domain,seq,data); CREATE TABLE world(domain,seq,data); '
                         'CREATE TABLE entity(id,data); CREATE TABLE metadata(key PRIMARY KEY,value);')
        for kind, raw in chosen.items():
            expected = reports[eid, kind]
            before = raw.stat()
            hasher = hashlib.sha256()
            checked = 0
            last_progress = time.monotonic()
            with raw.open('rb') as stream:
                for block in iter(lambda: stream.read(8 << 20), b''):
                    hasher.update(block)
                    checked += len(block)
                    if time.monotonic() - last_progress >= 10:
                        print('HASH', eid, kind, checked, '/', before.st_size, flush=True)
                        last_progress = time.monotonic()
            if hasher.hexdigest() != expected['sha256']:
                raise ValueError('Original byte-offset index cannot be used with changed spool: ' + str(raw))
            if (before.st_size, before.st_mtime_ns) != (raw.stat().st_size, raw.stat().st_mtime_ns):
                raise ValueError('Spool changed during verification')
            idx = source / episode['spools'][kind]['index']
            with closing(sqlite3.connect(idx.resolve().as_uri() + '?mode=ro', uri=True)) as frozen:
                if frozen.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                    raise ValueError('Corrupt frozen index: ' + str(idx))
                count = frozen.execute('SELECT COUNT(*) FROM records').fetchone()[0]
                if count != expected['records']:
                    raise ValueError('Record count differs from source audit')
                with raw.open('rb') as stream:
                    for ordinal in sorted({0, count // 2, count - 1}):
                        seq, domain, client_domain, offset, length = frozen.execute(
                            'SELECT seq,domain,client_domain,offset,length FROM records WHERE ordinal=?', (ordinal,)).fetchone()
                        stream.seek(offset)
                        row = json.loads(stream.read(length))
                        if row['episode_id'] != eid or row['server_clock_domain'] != domain:
                            raise ValueError('Indexed identity/domain mismatch')
                        sequence = row.get('sample_seq' if kind == 'world' else 'client_sample_seq',
                                           row.get('sample_seq_estimate'))
                        if kind != 'world' and (row['client_clock_domain'] != client_domain or sequence != seq):
                            raise ValueError('Indexed client sequence mismatch')
                        if kind == 'world' and sequence != seq:
                            raise ValueError('Indexed world sequence mismatch')
                cursor = frozen.execute('SELECT seq,domain,client_domain,offset,length FROM records ORDER BY ordinal')
                while batch := cursor.fetchmany(4096):
                    rows = []
                    for seq, domain, client_domain, offset, length in batch:
                        pointer = zlib.compress(json.dumps({'_jsonl': str(raw), 'offset': offset, 'length': length},
                                                          separators=(',', ':')).encode(), 1)
                        rows.append((domain, seq, pointer) if kind == 'world' else (kind, client_domain, seq, pointer))
                    db.executemany('INSERT INTO world VALUES (?,?,?)' if kind == 'world' else 'INSERT INTO camera VALUES (?,?,?,?)', rows)
                counts[kind] = count
                sources[kind] = {'path': str(raw), 'sha256': expected['sha256'], 'records': count}
        db.executescript('CREATE INDEX camera_key ON camera(agent,domain,seq); '
                         'CREATE INDEX world_key ON world(domain,seq); CREATE INDEX entity_key ON entity(id);')
        db.execute('INSERT INTO metadata VALUES (?,?)', ('complete', json.dumps(counts)))
        db.commit()
    temp.rename(final)
    result = {'episode_id': eid, 'root': str(source / episode['path']), 'index': str(final),
              'sources': sources, 'counts': counts, 'status': 'BYTE_EQUIVALENT_INDEX_ADAPTED',
              'authoritative_lifecycle_records': 0, 'training_status': 'NOT_YET_QUALIFIED'}
    receipt.write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--old-source', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--watch', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    episodes = json.loads((a.source / 'data/episodes.json').read_text())
    reports = {(r['episode_id'], r['agent'] or 'world'): r for r in
               json.loads((a.source / 'audit/original/input_indices.json').read_text())}
    completed, failed = {}, {}
    while True:
        for episode in episodes:
            eid = episode['episode_id']
            if eid in completed or eid in failed:
                continue
            print('CHECK', eid, flush=True)
            (a.output / 'INDEX_STATUS.json').write_text(json.dumps({
                'status': 'checking_episode', 'current_episode': eid, 'completed': len(completed),
                'total': len(episodes), 'failed': failed, 'updated_at': time.time()}, indent=2))
            try:
                result = adapt_episode(a.source, a.old_source, episode, reports, a.output)
                if result:
                    completed[eid] = result
                    print('ADAPTED', eid, result['counts'], flush=True)
            except Exception as exc:
                failed[eid] = str(exc)
                print('FAILED', eid, str(exc), flush=True)
            state = {'status': 'indexing', 'total': len(episodes), 'completed': len(completed),
                     'failed': failed, 'updated_at': time.time(),
                     'waiting_for_spools': [e['episode_id'] for e in episodes if e['episode_id'] not in completed and e['episode_id'] not in failed]}
            (a.output / 'INDEX_STATUS.json').write_text(json.dumps(state, indent=2))
        (a.output / 'report.json').write_text(json.dumps(list(completed.values()), indent=2))
        if len(completed) + len(failed) == len(episodes) or not a.watch:
            state['status'] = 'complete' if len(completed) == len(episodes) else 'incomplete'
            (a.output / 'INDEX_STATUS.json').write_text(json.dumps(state, indent=2))
            break
        time.sleep(20)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
