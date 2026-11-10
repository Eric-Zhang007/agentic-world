"""Prepare one verified real episode on Linux for frozen-render integration.

Downloads only the static world, frozen inputs and selected episode inputs.
The episode routing table is reconstructed from original source metadata;
it is not presented as the missing original assembly audit.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import oss2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--episode', default='ep_20260909_132205_d2da')
    p.add_argument('--agent', default='agent_000')
    p.add_argument('--segment', default='segment_00006')
    a = p.parse_args()
    source = a.root / 'source'
    index = json.loads((source / 'data/index.json').read_text())
    by_id = {e['episode_id']: e for e in index['episodes']}
    original = json.loads((source / 'method/original_sources.json').read_text())
    episodes = []
    for entry in original:
        rel = by_id[entry['episode_id']]['path']
        episodes.append({'episode_id': entry['episode_id'], 'group': entry['group'],
                         'path': 'data/' + rel, 'spools': {
            kind: {'source': 'data/' + rel + ('/.spool/world_state.jsonl' if kind == 'world'
                    else '/.spool/agents/' + kind + '/client.jsonl'),
                   'index': 'data/record_indices/' + entry['episode_id'] + '/' + kind + '.sqlite3'}
            for kind in entry['spools']}})
    route = source / 'data/episodes.json'
    if route.exists():
        if json.loads(route.read_text()) != episodes:
            raise ValueError('Existing episode routing conflicts with original sources')
    else:
        route.write_text(json.dumps(episodes, indent=2))
    (a.root / 'routing_provenance.json').write_text(json.dumps({
        'derived_from': ['method/original_sources.json', 'data/index.json'],
        'note': 'Reconstructed routing only, not original audit/files.jsonl'}, indent=2))
    e = next(e for e in episodes if e['episode_id'] == a.episode)
    rel = by_id[a.episode]['path']
    needed = {
        'GTA_555/' + rel + '/manifest.json': e['path'] + '/manifest.json',
        'GTA_555/evidence/frozen_jobs_and_spool_indices.zip': 'method/frozen_inputs.zip',
    }
    for kind in ('world', a.agent):
        needed['GTA_555/' + e['spools'][kind]['source'].removeprefix('data/')] = e['spools'][kind]['source']
    segment_rel = rel + '/derived/segments/' + a.agent + '/' + a.segment
    for name in ('compiler_manifest.json', 'validation.json', 'frames.jsonl'):
        needed['GTA_555/' + segment_rel + '/' + name] = 'data/' + segment_rel + '/' + name
    rgb = rel + '/agents/' + a.agent + '/video_' + a.segment.removeprefix('segment_') + '.mkv'
    needed['GTA_555/' + rgb] = 'data/' + rgb
    objects = [json.loads(line) for line in a.inventory.open()]
    tasks = []
    for row in objects:
        if row['Key'].startswith('GTA_555/worlds/') and not row['Key'].endswith('/'):
            needed[row['Key']] = 'data/' + row['Key'].removeprefix('GTA_555/')
        if row['Key'] in needed:
            tasks.append((row, source / needed[row['Key']]))
    if set(needed) != {r['Key'] for r, _ in tasks}:
        raise ValueError('Some required input keys are absent from OSS inventory')
    auth = oss2.Auth(os.environ['OSS_ACCESS_KEY_ID'], os.environ['OSS_ACCESS_KEY_SECRET'])
    state = {'status': 'downloading', 'files': len(tasks), 'completed': 0, 'failed': 0,
             'episode': a.episode, 'agent': a.agent, 'segment': a.segment, 'started_at': time.time()}

    def save():
        state['updated_at'] = time.time()
        temp = a.root / 'BOOTSTRAP_STATUS.tmp'
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(a.root / 'BOOTSTRAP_STATUS.json')

    def fetch(task):
        row, dst = task
        if dst.exists() and dst.stat().st_size == int(row['Size']):
            return
        if dst.exists():
            raise ValueError('Conflicting existing file: ' + str(dst))
        dst.parent.mkdir(parents=True, exist_ok=True)
        bucket = oss2.Bucket(auth, 'https://oss-cn-beijing.aliyuncs.com', 'gta-data-lqr')
        temporary = dst.with_name(dst.name + '.download')
        for attempt in range(3):
            try:
                bucket.get_object_to_file(row['Key'], str(temporary))
                if temporary.stat().st_size != int(row['Size']):
                    raise ValueError('Downloaded size mismatch')
                temporary.rename(dst)
                return
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)

    save()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch, t): t for t in tasks}
        for future in as_completed(futures):
            future.result()
            state['completed'] += 1
            save()
    state['status'] = 'verifying_world_and_spools'
    save()
    world = source / 'data/worlds/gta5_scene_v3_20260908'
    for name, expected in json.loads((world / 'manifest.json').read_text())['sha256'].items():
        digest = hashlib.sha256()
        with (world / name).open('rb') as stream:
            for block in iter(lambda: stream.read(8 << 20), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError('World artifact differs from original manifest: ' + name)
    reports = json.loads((source / 'audit/original/input_indices.json').read_text())
    for kind in ('world', a.agent):
        report = next(r for r in reports if r['episode_id'] == a.episode and (r['agent'] or 'world') == kind)
        path = source / e['spools'][kind]['source']
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 << 20), b''):
                digest.update(block)
        if digest.hexdigest() != report['sha256']:
            raise ValueError('Spool does not match original byte index: ' + kind)
    segments = json.loads((source / 'data/segments.json').read_text())['segments']
    job = next(r for r in segments if r['output_relative'] == segment_rel)
    jobs = [{'episode': a.episode, 'agent': a.agent, 'segment': a.segment,
             'expected_frames': job['frames'], 'usable_frames': job['usable_frames'],
             'ready': True, 'rgb_relative': 'data/' + rgb}]
    (a.root / 'smoke_jobs.json').write_text(json.dumps(jobs, indent=2))
    state['status'] = 'ready'
    state['spool_hashes_match'] = True
    save()
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
