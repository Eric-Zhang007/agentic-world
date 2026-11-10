"""Audit actual GTA camera/world joins; this does not certify RGB visibility."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3

import numpy as np
import pyarrow.parquet as pq

from scripts.prepare_gta_object_memory import camera_geometry


def model_id(value):
    return None if value is None else int(value) & 0xffffffff


def compact_camera(payload):
    entities = {}
    duplicates = set()
    for entity in (payload.get('scene_observation') or {}).get('entities', []):
        oid = entity.get('dataset_entity_id')
        if oid is None:
            continue
        oid = str(oid)
        if oid in entities:
            duplicates.add(oid)
        entities[oid] = model_id(entity.get('model_hash'))
    for oid in duplicates:
        entities.pop(oid)
    return {
        'rendered_camera_position': payload.get('rendered_camera_position'),
        'projection_rays': payload.get('projection_rays'),
        'entities': entities,
        'server_clock_domain': payload.get('server_clock_domain'),
    }


def build_index(root, connection):
    connection.execute('CREATE TABLE camera (episode TEXT, agent TEXT, domain TEXT, seq INTEGER, data TEXT)')
    connection.execute('CREATE TABLE world (episode TEXT, domain TEXT, seq INTEGER, time_ns INTEGER)')
    counts = Counter()
    for path in sorted((root/'agents').glob('*/camera.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            values = []
            for row in batch.to_pylist():
                p = json.loads(row['payload_json'])
                values.append((row['episode_id'], row['agent_id'], row['client_clock_domain'],
                               row['client_sample_seq'], json.dumps(compact_camera(p))))
            connection.executemany('INSERT INTO camera VALUES (?,?,?,?,?)', values)
            counts['camera_records'] += len(values)
        connection.commit()
        print(dict(counts), flush=True)
    for batch in pq.ParquetFile(root/'server/world_state.parquet').iter_batches(batch_size=256):
        values = []
        for row in batch.to_pylist():
            p = json.loads(row['payload_json'])
            values.append((row['episode_id'], p.get('server_clock_domain'), row['sample_seq'],
                           p.get('server_monotonic_ns')))
        connection.executemany('INSERT INTO world VALUES (?,?,?,?)', values)
        counts['world_records'] += len(values)
    connection.execute('CREATE INDEX camera_key ON camera(episode,agent,domain,seq)')
    connection.execute('CREATE INDEX world_key ON world(episode,domain,seq)')
    connection.commit()
    return counts


def audit_frame(connection, record, frame, cache):
    episode, agent = record['episode_id'], record['agent_id']
    domain, server = frame['client_clock_domain'], frame['server_clock_domain']
    cameras = []
    for field in ('lower_client_sample_seq', 'upper_client_sample_seq'):
        key = (episode, agent, domain, frame['camera_interpolation'][field])
        if key not in cache:
            rows = connection.execute('SELECT data FROM camera WHERE episode=? AND agent=? AND domain=? AND seq=?', key).fetchall()
            cache[key] = json.loads(rows[0][0]) if len(rows) == 1 else None
        p = cache[key]
        if p is None:
            return 'missing_or_ambiguous_camera', {}
        if p['server_clock_domain'] != server:
            return 'camera_server_clock_mismatch', {}
        cameras.append(p)
    rows = connection.execute('SELECT time_ns FROM world WHERE episode=? AND domain=? AND seq=?',
                              (episode, server, frame['world_capture_tick'])).fetchall()
    if len(rows) != 1:
        return 'missing_or_ambiguous_world', {}
    try:
        with np.errstate(all='raise'):
            origin, basis = camera_geometry(*cameras, frame['camera_interpolation']['alpha'])
        if origin.shape != (3,) or not np.isfinite(origin).all():
            return 'invalid_camera_origin', {}
    except (TypeError, ValueError, KeyError, AssertionError, FloatingPointError):
        return 'invalid_interpolated_camera', {}
    lower, upper = (p['entities'] for p in cameras)
    common = {oid: model for oid, model in lower.items()
              if model is not None and oid in upper and upper[oid] == model}
    return None, common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    connection = sqlite3.connect(args.output/'record_index.sqlite3')
    counts = build_index(args.episode, connection)
    cached_path = None
    frame_results = {}
    cache = {}
    with args.candidates.open(encoding='utf-8') as source, (args.output/'audited.jsonl').open('w', encoding='utf-8') as dest:
        for line in source:
            record = json.loads(line)
            path = Path(record['source_manifest']).parent/'frames.jsonl'
            if path != cached_path:
                with path.open(encoding='utf-8') as f:
                    frames = {r['proxy_frame_index']: r for r in map(json.loads, f)}
                cached_path = path
                frame_results = {}
                cache = {}
            indices = [record['reference_frame'], *range(record['target_start_frame'], record['target_end_frame_exclusive'])]
            common = {e['dataset_entity_id']: model_id(e['model_hash']) for e in record['lifecycle_qualified_objects']}
            errors = []
            for index in indices:
                if index not in frame_results:
                    frame_results[index] = audit_frame(connection, record, frames[index], cache)
                reason, entities = frame_results[index]
                if reason:
                    errors.append({'frame': index, 'reason': reason})
                common = {oid: model for oid, model in common.items() if oid in entities and entities[oid] == model}
            record['record_join_errors'] = errors
            record['camera_supported_entity_ids'] = sorted(common)
            record['record_join_passed'] = not errors
            record['training_ready'] = False
            record['remaining_checks'] = ['RGB visibility and occlusion', 'spatial alignment', 'object state', 'HUD', 'media decode']
            dest.write(json.dumps(record)+'\n')
            counts['windows'] += 1
            counts['windows_with_valid_joins'] += not errors
            counts['windows_with_supported_entities'] += bool(common) and not errors
            counts['camera_supported_entity_edges'] += len(common)
            for reason in set(e['reason'] for e in errors):
                counts['windows_'+reason] += 1
            if counts['windows'] % 250 == 0:
                print(dict(counts), flush=True)
    connection.close()
    summary = {'status': 'RECORD_JOIN_AUDIT_ONLY', 'counts': dict(counts),
               'limitations': 'Scene entity presence is not RGB visibility. World joins verify record existence, not object state compatibility. No static geometry or segmentation validation.'}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
