"""Feed verified source/index/media joins into existing passage preparation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time
from scripts.prepare_gta_spool_passages import work


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--old-source', type=Path, required=True)
    p.add_argument('--indices', type=Path, required=True)
    p.add_argument('--proxies', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache-root', type=Path)
    p.add_argument('--limit', type=int)
    p.add_argument('--watch', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    jobs = json.loads((a.source / 'data/segments.json').read_text())['segments']
    episodes = {e['path']: e for e in json.loads((a.source / 'data/index.json').read_text())['episodes']}
    results = {}
    for path in a.output.glob('qualification/*/summary.json'):
        r = json.loads(path.read_text())
        results[r['segment_key']] = r
    failures = {}
    state = {'status': 'preparing', 'total': len(jobs), 'started_at': time.time()}

    def save():
        state.update(completed=len(results), failed=len(failures),
                     points=sum(r['indexed_points'] for r in results.values()), updated_at=time.time())
        (a.output / 'PREPARATION_STATUS.json').write_text(json.dumps(state, indent=2))

    save()
    while True:
        advanced = False
        for job in jobs:
            e = episodes[job['episode_relative']]
            agent, segment = Path(job['output_relative']).parts[-2:]
            key = '_'.join((e['episode_id'], agent, segment))
            if key in results or key in failures:
                continue
            idx_receipt = a.indices / (e['episode_id'] + '.json')
            proxy_receipt = a.proxies / job['output_relative'] / 'local_verified.json'
            folder = a.source / 'data' / job['output_relative']
            required = [idx_receipt, proxy_receipt, folder / 'manifest.json',
                        folder / 'frames.jsonl', folder / 'validation.json']
            if not all(p.is_file() for p in required):
                continue
            rgb = a.source / 'data' / job['rgb_relative']
            if not rgb.is_file():
                rgb = a.old_source / job['rgb_relative']
            if not rgb.is_file():
                continue
            state['active_segment'] = key
            save()
            try:
                proxy = json.loads(proxy_receipt.read_text())
                record_index = json.loads(idx_receipt.read_text())
                index_path = Path(record_index['index'])
                if a.cache_root:
                    cached_index = a.cache_root / 'indices' / index_path.name
                    cached_index.parent.mkdir(parents=True, exist_ok=True)
                    if not cached_index.exists():
                        shutil.copy2(index_path, cached_index)
                    index_path = cached_index
                if not proxy['local_hash_matches'] or not Path(proxy['proxy']).is_file():
                    raise ValueError('Verified proxy missing')
                if sha(folder / 'validation.json') != job['validation_sha256']:
                    raise ValueError('Segment validation differs from source index')
                validation = json.loads((folder / 'validation.json').read_text())
                for name in ('manifest.json', 'frames.jsonl'):
                    expected = validation['published_sha256'].get(name)
                    if expected and sha(folder / name) != expected:
                        raise ValueError('Changed alignment metadata: ' + name)
                if sha(rgb) != job['rgb_sha256']:
                    raise ValueError('RGB differs from frozen publication')
                src = {'episode_id': e['episode_id'], 'agent_id': agent, 'segment': segment,
                       'source_manifest': str(folder / 'manifest.json'),
                       'episode_root': str(a.source / 'data' / e['path']),
                       'record_index': str(index_path),
                       'media_paths': {'rgb_video': str(rgb), 'proxy_video': proxy['proxy']},
                       'frame_count': job['frames'], 'source_hashes_verified': True}
                if a.cache_root:
                    src['frame_cache_root'] = str(a.cache_root / 'review_frames')
                result = work((src, str(a.indices), str(a.output)))
                results[key] = result
                advanced = True
                print('PREPARED', key, result['status'], result['indexed_points'], flush=True)
            except Exception as exc:
                failures[key] = str(exc)
                print('FAILED', key, str(exc), flush=True)
                (a.output / 'preparation_failures.json').write_text(json.dumps(failures, indent=2))
            save()
            if a.limit and len(results) >= a.limit:
                break
        if (a.limit and len(results) >= a.limit) or not a.watch or len(results) + len(failures) == len(jobs):
            break
        state['status'] = 'waiting_for_verified_inputs' if not advanced else 'preparing'
        save()
        time.sleep(15)
    with (a.output / 'trajectory_points.jsonl').open('wb') as out:
        for key in results:
            out.write((a.output / 'qualification' / key / 'points.jsonl').read_bytes())
    state['status'] = 'prepared' if len(results) == len(jobs) else 'partial'
    state['statuses'] = dict(Counter(r['status'] for r in results.values()))
    save()
    (a.output / 'source_ledger.jsonl').write_text(''.join(json.dumps(r['source']) + '\n' for r in results.values()))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
