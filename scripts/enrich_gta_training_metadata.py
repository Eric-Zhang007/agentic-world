"""Export indexed action/world observations as audit-only training sidecars."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from memory.gta_release import Records


def extract(records, agent, frame):
    interpolation = frame.get('camera_interpolation')
    if not interpolation:
        raise ValueError('Missing camera interpolation')
    clients = [records.camera(agent, frame['client_clock_domain'], interpolation[k])
               for k in ('lower_client_sample_seq', 'upper_client_sample_seq')]
    world = records.world(frame['server_clock_domain'], frame['world_capture_tick'])
    if any(c['episode_id'] != world['episode_id'] or c['server_clock_domain'] != frame['server_clock_domain'] for c in clients):
        raise ValueError('Mismatched source identities')
    client_keys = ('client_sample_seq', 'sample_seq_estimate', 'server_time_estimate_ns',
                   'action', 'agent_state', 'camera', 'subject_observation', 'scene_observation')
    world_keys = ('episode_id', 'sample_seq', 'server_monotonic_ns', 'server_clock_domain',
                  'environment', 'agents', 'vehicles', 'peds', 'objects', 'task_state',
                  'relation_edges', 'entity_lifecycle_events', 'entity_scope')
    return {'proxy_frame_index': frame['proxy_frame_index'],
            'target_time_ns': frame['server_time_estimate_ns'],
            'interpolation': interpolation,
            'client_brackets': [{k: c[k] for k in client_keys if k in c} for c in clients],
            'world_record': {k: world[k] for k in world_keys if k in world},
            'compiler_metadata': {k: frame[k] for k in ('static_candidates', 'pose_sources',
                'entity_ids', 'excluded_animals', 'scene_observation_error', 'scene_observation_entities') if k in frame}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--indices', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--follow', action='store_true')
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    connections = {}
    counts = Counter()
    done = set()
    try:
        while True:
            for path in sorted((a.root / 'training/samples').glob('*/sample.json')):
                key = path.parent.name
                if key in done:
                    continue
                destination = a.output / (key + '.json')
                if destination.exists():
                    done.add(key)
                    counts['existing'] += 1
                    continue
                sample = json.loads(path.read_bytes())
                if sample['training_status'].startswith('EXCLUDED'):
                    done.add(key)
                    counts['excluded'] += 1
                    continue
                episode = sample['episode_id']
                receipt = a.indices / (episode + '.json')
                if not receipt.exists():
                    counts['waiting_index'] += 1
                    continue
                if episode not in connections:
                    connections[episode] = Records(Path(json.loads(receipt.read_bytes())['index']))
                records = connections[episode]
                target = json.loads((path.parent / 'target_state.json').read_bytes())
                ref = json.loads((path.parent / 'reference_state.json').read_bytes())
                frames = json.loads((path.parent / 'target_frame_states.json').read_bytes())
                reference_frame = None
                with Path(ref['source_manifest']).with_name('frames.jsonl').open() as stream:
                    for line in stream:
                        frame = json.loads(line)
                        if frame['proxy_frame_index'] == ref['frame_index']:
                            reference_frame = frame
                            break
                if reference_frame is None:
                    raise ValueError('Missing historical frame: ' + key)
                payload = {'sample_id': key, 'episode_id': episode,
                           'usage': 'Audit and auxiliary supervision only; not injected into generation conditions',
                           'visibility': 'Scene/world records are scoped observations, not visibility labels',
                           'temporal_scope': 'Camera interpolation brackets; actions and discrete states are not interpolated',
                           'reference': extract(records, ref['agent_id'], reference_frame),
                           'target_samples': [extract(records, target['agent_id'], frames[i]) for i in (0,30,60,90,119)]}
                temporary = destination.with_suffix('.tmp')
                temporary.write_text(json.dumps(payload, separators=(',', ':')), encoding='utf-8')
                temporary.replace(destination)
                done.add(key)
                counts['written'] += 1
                print('ENRICHED', key, flush=True)
                if a.limit and counts['written'] >= a.limit:
                    return
            (a.output / 'STATUS.json').write_text(json.dumps({'counts':dict(counts), 'updated_at':time.time(),
                'status':'following' if a.follow else 'complete', 'training_acceptance':False}, indent=2))
            if not a.follow:
                break
            time.sleep(20)
    finally:
        for records in connections.values():
            records.close()


if __name__ == '__main__':
    main()
