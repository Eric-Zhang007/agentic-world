"""Validate expanded scene contracts and exercise the real mixed data loader."""
from collections import Counter
from pathlib import Path

import cv2
import orjson

from memory.gta_mixed_dataset import MixedMemoryDataset
from memory.gta_release_media import open_video


def validate(root):
    root = Path(root)
    training = root / 'training'
    mixture = root / 'mixture'
    rows = [orjson.loads(line) for line in (training / 'samples.jsonl').read_bytes().splitlines()]
    assert rows and len({row['sample_id'] for row in rows}) == len(rows)
    episodes = {}
    for row in rows:
        folder = training / row['sample_root']
        sample = orjson.loads((folder / 'sample.json').read_bytes())
        ref = orjson.loads((folder / 'reference_state.json').read_bytes())
        target = orjson.loads((folder / 'target_state.json').read_bytes())
        states = orjson.loads((folder / 'target_frame_states.json').read_bytes())
        assert len(states) == sample['frame_count'] == 120
        assert sample['reference_time_ns'] < sample['target_time_ns']
        assert ref['episode_id'] == target['episode_id'] == row['episode_id']
        assert ref['server_clock_domain'] == target['server_clock_domain']
        assert abs((target['time_ns'] - ref['time_ns']) / 1e9 - sample['reference']['age_sec']) < .001
        assert all(min(scores) >= 26 for scores in sample['alignment_psnr'].values())
        assert row['kind'] != 'gap_unknown'
        assert sample['training_status'] in ('AUTO_REFERENCE_WITH_INHERITED_TARGET_QA', 'AUTO_REFERENCE_WITH_REVIEWED_TARGET_QA')
        assert episodes.setdefault(row['episode_id'], row['split']) == row['split']
        for key in ('reference_rgb.png', 'reference_whitebox.png', 'target_rgb.png', 'target_whitebox.png', 'valid_pixels.png'):
            image = cv2.imread(str(folder / key))
            assert image is not None and list(image.shape[1::-1]) == target['resolution'], (row['sample_id'], key)
        for key in ('target_rgb.mp4', 'target_whitebox.mp4'):
            cap = open_video(folder / key)
            try:
                assert round(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 120, (row['sample_id'], key)
                assert abs(cap.get(cv2.CAP_PROP_FPS) - sample['fps']) < .01, (row['sample_id'], key, cap.get(cv2.CAP_PROP_FPS))
            finally:
                cap.release()
    plans = []
    for path in sorted(mixture.glob('train_round_*.jsonl')):
        plan = [orjson.loads(line) for line in path.read_bytes().splitlines()]
        assert len({(r['dataset'], r['sample_id']) for r in plan}) == len(plan)
        assert all(r['split'] == 'train' for r in plan)
        scenes = [r for r in plan if r['task'] == 'scene_background']
        assert 2 * sum(r['kind'] == 'nearby_history' for r in scenes) <= len(scenes)
        plans.append({'manifest': path.name, 'count': len(plan), 'tasks': dict(Counter(r['task'] for r in plan))})
    roots = orjson.loads((mixture/'roots.example.json').read_bytes())
    exercised = []
    combined_episodes = {}
    combined_ids = set()
    target_windows = Counter()
    for split in ('train', 'validation', 'test'):
        dataset = MixedMemoryDataset(mixture / (split + '.jsonl'), roots, temporal_indices=[0, 30, 60, 90, 119])
        groups = {}
        for i, row in enumerate(dataset.rows):
            assert row['split'] == split
            assert combined_episodes.setdefault(row['episode_id'], split) == split, row['episode_id']
            identity = (row['dataset'], row['sample_id'])
            assert identity not in combined_ids, identity
            combined_ids.add(identity)
            if row['task'] == 'scene_background':
                segment, frame = row['sample_id'].rsplit('_f', 1)
                start = int(frame)
            else:
                segment, window = row['sample_id'].rsplit('_w', 1)
                start = (int(window) + 1) * 120
            target_windows[(segment, start)] += 1
            groups.setdefault((row['task'], row.get('kind'),row['dataset']), i)
        for key, i in groups.items():
            sample = dataset[i]
            assert sample['conditions']['target_whitebox'].shape == sample['supervision']['target_rgb'].shape
            assert sample['conditions']['target_whitebox'].shape[0] == 5
            assert 'target_rgb' not in sample['conditions']
            if key[0] == 'scene_background':
                assert not sample['conditions']['object_references']
            exercised.append({'split': split, 'task': key[0], 'kind': key[1], 'sample_id': sample['sample_id'],
                              'shape': list(sample['supervision']['target_rgb'].shape)})
    report = {'scene_samples_checked': len(rows), 'checks': ['unique_ids', 'historical_reference', 'episode_clock_isolation',
              '120_frame_states', 'export_alignment_scores', 'all_images_readable', 'video_frame_count_and_fps',
              'round_no_replacement', 'round_nearby_fraction', 'real_mixed_loader'],
              'combined_episode_split_isolation': True,
              'combined_records': len(combined_ids),
              'combined_unique_target_windows': len(target_windows),
              'target_windows_reused_across_tasks': sum(n > 1 for n in target_windows.values()),
              'target_window_count_basis': 'Source segment and 120-frame target start; not independent scene count',
              'loader_examples': exercised, 'rounds': plans,
              'full_video_decode': 'Performed during export; metadata rechecked here',
              'visual_qa_scope': 'See each sample; this validator does not assert exhaustive visual inspection',
              'passed': True}
    (mixture / 'verification.json').write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('D:/GTA_passage_pairs_20261020'))
    print(validate(parser.parse_args().root))
