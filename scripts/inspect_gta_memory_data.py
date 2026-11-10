"""Inspect a downloaded GTA segment without treating candidates as visibility truth."""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--segment', default='segment_00001')
    parser.add_argument('--agent', default='agent_000')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.episode
    segment = root / 'derived/segments' / args.agent / args.segment
    args.output.mkdir(parents=True, exist_ok=True)
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    manifest = read(segment / 'manifest.json')
    with (segment / 'frames.jsonl').open(encoding='utf-8') as f:
        frames = [json.loads(line) for line in f if line.strip()]
    valid = [f for f in frames if f.get('status') == 'estimated_alignment'
             and f.get('scene_index') and f.get('server_time_estimate_ns') is not None]
    if not valid:
        raise ValueError('No inspectable estimated-alignment frames')
    reference = valid[0]
    target = next(f for f in valid if f['proxy_frame_index'] >= reference['proxy_frame_index'] + 120)
    assert reference['server_time_estimate_ns'] < target['server_time_estimate_ns']
    assert reference['server_clock_domain'] == target['server_clock_domain']
    scene = read(segment / target['scene_index'])
    report = {
        'episode_root': str(root), 'segment_manifest': manifest,
        'rows': len(frames), 'status_counts': dict(Counter(f.get('status') for f in frames)),
        'frame_fields': sorted(set().union(*(f.keys() for f in frames))),
        'appearance': read(segment / 'appearance.json'),
        'static_candidate_count': len(scene.get('static_placement_ids', [])),
        'static_candidate_examples': scene.get('static_placement_ids', [])[:5],
        'surface_candidate_examples': scene.get('surface_components', [])[:3],
        'training_ready': False,
        'limitations': ['Candidate IDs do not prove pixel visibility or occlusion.',
                       'Reference and target state compatibility is not verified.',
                       'Estimated clock alignment needs visual validation.',
                       'Static world identity and component ID stability need verification.'],
        'parquet': {},
    }
    try:
        import pyarrow.parquet as pq
        for rel in [f'agents/{args.agent}/camera.parquet', f'agents/{args.agent}/agent_state.parquet',
                    'server/entities.parquet', 'server/world_state.parquet', 'server/events.parquet']:
            path = root / rel
            if not path.exists():
                report['parquet'][rel] = {'available': False}
                continue
            pf = pq.ParquetFile(path)
            first = next(pf.iter_batches(batch_size=1)).to_pylist()
            report['parquet'][rel] = {'rows': pf.metadata.num_rows, 'schema': str(pf.schema_arrow),
                                      'first_row': first}
        report['sample_joins'] = {}
        for role, frame in [('reference', reference), ('target', target)]:
            interpolation = frame['camera_interpolation']
            wanted = {interpolation['lower_client_sample_seq'], interpolation['upper_client_sample_seq']}
            cameras = []
            pf = pq.ParquetFile(root / f'agents/{args.agent}/camera.parquet')
            for batch in pf.iter_batches(batch_size=512):
                for row in batch.to_pylist():
                    if row['client_sample_seq'] in wanted and row['client_clock_domain'] == frame['client_clock_domain']:
                        cameras.append(json.loads(row['payload_json']))
                        wanted.remove(row['client_sample_seq'])
                if not wanted:
                    break
            world = None
            pf = pq.ParquetFile(root / 'server/world_state.parquet')
            for batch in pf.iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    if row['sample_seq'] == frame['world_capture_tick']:
                        world = json.loads(row['payload_json'])
                        break
                if world is not None:
                    break
            report['sample_joins'][role] = {'camera_samples_found': len(cameras),
                'missing_camera_sample_seqs': sorted(wanted), 'camera_samples': cameras,
                'world_tick_found': world is not None, 'world_state': world}
    except ImportError:
        report['parquet_error'] = 'pyarrow is not installed'
    cards = []
    pairs = []
    for name, row in [('reference', reference), ('target', target)]:
        images = []
        for kind, key, idxkey in [('rgb', 'rgb_video', 'rgb_decode_index'),
                                  ('whitebox', 'proxy_video', 'proxy_frame_index')]:
            path = root / manifest[key]
            cap = cv2.VideoCapture(str(path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, row[idxkey])
            ok, image = cap.read()
            cap.release()
            if not ok:
                raise ValueError(f'Cannot decode {path} frame {row[idxkey]}')
            cv2.imwrite(str(args.output / f'{name}_{kind}.jpg'), image)
            images.append(cv2.resize(image, (640, 360)))
        import numpy as np
        cards.append(np.concatenate(images, axis=1))
        pairs.append({'role': name, 'rgb_video': str(root / manifest['rgb_video']),
                      'whitebox_video': str(root / manifest['proxy_video']),
                      'frame_metadata': row})
    cv2.imwrite(str(args.output / 'reference_target_pairs.jpg'), np.concatenate(cards, axis=0))
    common = sorted(set(reference.get('entity_ids', [])) & set(target.get('entity_ids', [])))
    draft = {'status': 'CANDIDATE_ONLY_NOT_TRAINING_READY', 'pairs': pairs,
             'reference_edges': [{'entity_id': entity, 'reference_frame': reference['proxy_frame_index'],
                                  'target_frame': target['proxy_frame_index'],
                                  'reference_time_ns': reference['server_time_estimate_ns'],
                                  'used_at_time_ns': target['server_time_estimate_ns'],
                                  'visibility_verified': False, 'state_compatibility': 'unknown'}
                                 for entity in common]}
    for name, data in [('inspection.json', report), ('memory_candidate.json', draft)]:
        (args.output / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(json.dumps({'rows': len(frames), 'status_counts': report['status_counts'],
                      'common_candidate_entities': len(common), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
