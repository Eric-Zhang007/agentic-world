"""Verify an extracted delivery using only stdlib, numpy, cv2 and its dataset.py.

Usage: python verify_gta_delivery.py --dataset PATH
Metadata is checked for every sample; one sample per nonempty split is decoded
through the delivered loader by default. This is not full media revalidation.
"""
import argparse
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path, PureWindowsPath
import sys

import cv2
import numpy as np


SPLITS = ('train', 'validation', 'test')
EDGE_FIELDS = {'object_id', 'category', 'reference_id', 'reference_bbox_xyxy',
               'target_projected_bboxes', 'observed_at_ns', 'used_at_ns', 'used_until_ns'}
TIME_FIELDS = ('observed_at_ns', 'used_at_ns', 'used_until_ns')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def asset(root, relative, directory=False):
    require(isinstance(relative, str) and bool(relative), 'Invalid asset path')
    require(not Path(relative).is_absolute() and not PureWindowsPath(relative).drive,
            'Asset path must be relative')
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), 'Asset path escapes root')
    require(path.is_dir() if directory else path.is_file(), 'Missing asset: ' + relative)
    return path


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_lines(path):
    with path.open(encoding='utf-8') as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def boxes(value, count):
    array = np.asarray(value, dtype=np.float64)
    require(array.shape == (count, 4) and np.isfinite(array).all(), 'Invalid boxes')
    require((array[:, :2] >= 0).all() and (array[:, 2:] > array[:, :2]).all(), 'Invalid box bounds')


def sample_metadata(root, sample):
    require(sample.get('training_ready') is True, 'Sample not training_ready')
    require(type(sample['frame_count']) is int and sample['frame_count'] == 120, 'Expected 120 frames')
    require(type(sample['fps']) in (int, float) and math.isfinite(sample['fps']) and sample['fps'] > 0,
            'Invalid FPS')
    conditions, supervision = sample['conditions'], sample['supervision']
    for key in ('prompt', 'target_whitebox'):
        asset(root, conditions[key])
    for key in ('target_rgb', 'valid_pixels'):
        asset(root, supervision[key])
    for key in ('camera_records', 'world_records', 'review'):
        asset(root, sample[key])
    frames = read_json(asset(root, sample['frame_state_index']))
    times = [frame['server_time_estimate_ns'] for frame in frames]
    require(len(times) == 121 and all(type(t) is int for t in times), 'Invalid frame-state timeline')
    require(all(a < b for a, b in zip(times, times[1:])), 'Nonmonotonic frame times')
    for field in ('proxy_frame_index', 'rgb_decode_index'):
        indices = [frame[field] for frame in frames]
        require(all(type(i) is int and i >= 0 for i in indices), 'Invalid decoder indices')
        require(indices[0] < indices[1] and all(b == a + 1 for a, b in zip(indices[1:], indices[2:])),
                'Noncontiguous target decoder indices')
    pairs = {}
    for index, pair in enumerate(conditions['reference_pairs']):
        rid = pair.get('reference_id', f"{sample['sample_id']}:ref{index}")
        require(isinstance(rid, str) and rid and rid not in pairs, 'Duplicate/invalid reference ID')
        pairs[rid] = {kind: asset(root, pair[kind]) for kind in ('rgb', 'whitebox')}
    require(bool(pairs), 'No reference pairs')
    edges = {}
    for edge in sample['object_references']:
        require(EDGE_FIELDS <= edge.keys(), 'Incomplete object reference')
        key = (edge['object_id'], edge['reference_id'])
        require(isinstance(key[0], str) and key[0] and key not in edges, 'Duplicate/invalid object edge')
        require(edge['category'] is None or isinstance(edge['category'], str), 'Invalid object category')
        require(key[1] in pairs, 'Unknown reference ID')
        require(edge['used_by'] == sample['sample_id'], 'Wrong reference used_by')
        edge_times = [edge[k] for k in TIME_FIELDS]
        require(all(type(t) is int for t in edge_times) and edge_times == [times[0], times[1], times[-1]],
                'Object edge time mismatch')
        boxes([edge['reference_bbox_xyxy']], 1)
        boxes(edge['target_projected_bboxes'], 120)
        for kind in ('rgb', 'whitebox'):
            require(asset(root, edge['reference_' + kind]) == pairs[key[1]][kind], 'Reference asset mismatch')
        for field in ('reference_mask', 'reference_crop'):
            asset(root, edge[field])
        if edge.get('embedding'):
            asset(root, edge['embedding'])
        edges[key] = edge
    require(bool(edges), 'No object edges')
    return edges


def loaded_contract(loaded, sample):
    require(loaded['sample_id'] == sample['sample_id'], 'Loader sample ID mismatch')
    conditions = loaded['conditions']
    require(set(conditions) == {'prompt', 'reference_pairs', 'target_whitebox', 'object_references'},
            'Condition whitelist violation')
    require(isinstance(conditions['prompt'], str), 'Prompt must be text')
    rgb = loaded['supervision']['target_rgb']
    proxy = conditions['target_whitebox']
    require(isinstance(rgb, np.ndarray) and isinstance(proxy, np.ndarray), 'Loader must return video arrays')
    require(rgb.ndim == 4 and rgb.shape[0] == 120 and rgb.shape[-1] == 3 and rgb.shape == proxy.shape,
            'Loaded paired shape/count mismatch')
    valid = loaded['supervision']['valid_pixels']
    require(isinstance(valid, np.ndarray) and valid.shape == rgb.shape[1:3]
            and np.isfinite(valid).all() and (valid >= 0).all() and (valid <= 1).all() and valid.any(),
            'Invalid loaded valid_pixels')
    pairs = {}
    for pair in conditions['reference_pairs']:
        require(set(pair) == {'reference_id', 'rgb', 'whitebox'}, 'Reference-pair whitelist violation')
        rid = pair['reference_id']
        require(rid not in pairs, 'Duplicate loaded reference ID')
        for kind in ('rgb', 'whitebox'):
            require(isinstance(pair[kind], np.ndarray) and pair[kind].shape == rgb.shape[1:],
                    'Reference image must be directly consumable')
        require(not pair['rgb'][valid == 0].any(), 'HUD pixels remain in reference RGB')
        pairs[rid] = pair
    expected_pairs = {p.get('reference_id', f"{sample['sample_id']}:ref{i}")
                      for i, p in enumerate(sample['conditions']['reference_pairs'])}
    require(set(pairs) == expected_pairs, 'Loaded reference IDs differ from sample')
    expected = {(e['object_id'], e['reference_id']): e for e in sample['object_references']}
    seen = set()
    for edge in conditions['object_references']:
        require(set(edge) == EDGE_FIELDS | {'reference_mask', 'reference_crop'}, 'Object condition whitelist violation')
        key = (edge['object_id'], edge['reference_id'])
        require(key in expected and key not in seen and key[1] in pairs, 'Loaded object-reference mismatch')
        seen.add(key)
        for field in EDGE_FIELDS:
            require(edge[field] == expected[key][field], 'Loaded edge field mismatch: ' + field)
        mask = edge['reference_mask']
        require(isinstance(mask, np.ndarray) and mask.shape == valid.shape and np.isfinite(mask).all()
                and mask.any() and not mask[valid == 0].any(), 'Invalid loaded object mask')
        box = np.asarray(edge['reference_bbox_xyxy'])
        x0, y0 = np.floor(box[:2]).astype(int)
        x1, y1 = np.ceil(box[2:]).astype(int)
        require(0 <= x0 < x1 <= rgb.shape[2] and 0 <= y0 < y1 <= rgb.shape[1], 'Loaded bbox out of bounds')
        crop = edge['reference_crop']
        require(isinstance(crop, np.ndarray) and np.array_equal(crop, pairs[key[1]]['rgb'][y0:y1, x0:x1]),
                'Crop must come from HUD-cleaned reference RGB')
    require(seen == set(expected), 'Missing loaded object edges')


def verify(dataset, loads_per_split=1):
    root = Path(dataset).resolve()
    report = {'status': 'FAILED', 'dataset': str(root), 'errors': [], 'split_samples': {},
              'samples': 0, 'edges': 0, 'loaded_samples': [],
              'scope': 'All split/sample/use metadata; sampled real loader decoding, not full media validation'}
    try:
        require(loads_per_split >= 1, 'At least one loader sample per nonempty split is required')
        acceptance = read_json(asset(root, 'acceptance_report.json'))
        require(acceptance.get('status') == 'RELEASE_VALIDATED', 'Release not validated')
        samples, sample_roots, episodes, rows_by_split = {}, set(), {}, {}
        for split in SPLITS:
            rows = list(read_lines(asset(root, f'splits/{split}.jsonl')))
            rows_by_split[split] = rows
            report['split_samples'][split] = len(rows)
            for row in rows:
                sid, episode = row['sample_id'], row['episode_id']
                require(sid not in samples, 'Duplicate sample: ' + sid)
                path = asset(root, row['sample_root'], directory=True)
                require(path not in sample_roots, 'Duplicate sample directory')
                sample_roots.add(path)
                require(episodes.get(episode, split) == split, 'Episode appears in multiple splits: ' + episode)
                episodes[episode] = split
                sample = read_json(asset(path, 'sample.json'))
                require(sample['sample_id'] == sid and sample['episode_id'] == episode, 'Split/sample identity mismatch')
                edges = sample_metadata(path, sample)
                require(row['objects'] == len(edges), 'Split object count mismatch')
                samples[sid] = (split, path, sample, edges)
        report['samples'] = len(samples)
        report['edges'] = sum(len(entry[3]) for entry in samples.values())
        require(bool(samples), 'Empty delivery')
        require(acceptance['accepted_samples'] == report['samples'], 'Acceptance sample count mismatch')
        require(acceptance['object_reference_edges'] == report['edges'], 'Acceptance edge count mismatch')
        require(acceptance['target_frames'] == 120 * report['samples'], 'Acceptance frame count mismatch')
        require(set(acceptance['split_samples']) <= set(SPLITS), 'Unknown acceptance split')
        require(all(acceptance['split_samples'].get(s, 0) == report['split_samples'][s] for s in SPLITS),
                'Acceptance split counts mismatch')
        require(acceptance['episode_split_map'] == episodes, 'Acceptance episode map mismatch')
        uses_seen = set()
        for use in read_lines(asset(root, 'object_reference_uses.jsonl')):
            sid = use['used_by']
            require(sid in samples, 'Use points to unknown sample')
            split, path, sample, edges = samples[sid]
            key = (use['object_id'], use['reference_id'])
            require((sid, *key) not in uses_seen, 'Duplicate reference use')
            uses_seen.add((sid, *key))
            require(key in edges and use['split'] == split, 'Use object/split mismatch')
            require(asset(root, use['sample_root'], directory=True) == path, 'Use sample root mismatch')
            require(use['frame_state_index'] == sample['frame_state_index'], 'Use frame index mismatch')
            for field, value in edges[key].items():
                require(field in use and use[field] == value, 'Use/edge mismatch: ' + field)
        require(len(uses_seen) == report['edges'], 'Missing object reference uses')
        # The delivered loader is executable Python and must be trusted by its recipient.
        spec = importlib.util.spec_from_file_location('_gta_delivery_dataset', asset(root, 'dataset.py'))
        module = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(module)
            for split, rows in rows_by_split.items():
                loader = module.GTAObjectMemoryDataset(root, split)
                require(len(loader) == len(rows), 'Loader split length mismatch')
                for index in range(min(loads_per_split, len(rows))):
                    sid = rows[index]['sample_id']
                    loaded_contract(loader[index], samples[sid][2])
                    report['loaded_samples'].append({'split': split, 'sample_id': sid})
        report['status'] = 'PASSED'
    except Exception as error:
        report['errors'].append({'type': type(error).__name__, 'message': str(error)})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--loads-per-split', type=int, default=1,
                        help='Real loader decodes per nonempty split (default: 1, minimum: 1)')
    args = parser.parse_args()
    # Loading dataset.py should not leave bytecode files in an extracted delivery.
    sys.dont_write_bytecode = True
    report = verify(args.dataset, args.loads_per_split)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['status'] == 'PASSED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
