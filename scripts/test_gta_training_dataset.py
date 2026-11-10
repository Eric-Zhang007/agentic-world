"""Small synthetic contract tests; never open or modify the released dataset."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory.gta_training_dataset import GTAObjectMemoryDataset


class TrainingDatasetContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sample_root = self.root / 'samples' / 'sample_a'
        self.sample_root.mkdir(parents=True)
        (self.root / 'splits').mkdir()
        (self.root / 'splits/train.jsonl').write_text(
            json.dumps({'sample_root': 'samples/sample_a'}) + '\n', encoding='utf-8')
        (self.sample_root / 'prompt.txt').write_text('Object A uses reference sample_a:ref0.', encoding='utf-8')
        self.rgb = np.full((2, 4, 6, 3), 70, dtype=np.uint8)
        self.whitebox = np.full_like(self.rgb, 20)
        self.valid = np.full((4, 6), 255, dtype=np.uint8)
        self.valid[0] = 0
        self.write_image('valid.png', self.valid)
        self.write_image('ref.png', np.full((4, 6, 3), (10, 30, 90), dtype=np.uint8))
        self.write_image('whitebox.png', self.whitebox[0])
        self.write_image('object_mask.png', np.full((4, 6), 255, dtype=np.uint8))
        np.save(self.sample_root / 'embedding.npy', np.ones(384, dtype=np.float32))
        self.edge = {
            'object_id': 'episode:entity:A', 'category': 'agent', 'reference_id': 'sample_a:ref0',
            'reference_bbox_xyxy': [1, 0, 5, 4],
            'target_projected_bboxes': [[1, 0, 5, 4], [0, 1, 4, 4]],
            'observed_at_ns': 100, 'used_at_ns': 200, 'used_until_ns': 300,
            'reference_rgb': 'ref.png', 'reference_whitebox': 'whitebox.png',
            'reference_mask': 'object_mask.png', 'reference_crop': 'MUST_NOT_READ.png',
            'embedding': 'embedding.npy',
            'lifecycle': {'despawn_server_monotonic_ns': 999999, 'despawn_basis': 'identity_changed'},
            'state_change_review': 'FUTURE_ONLY_SECRET', 'unrecognized_future_field': 'FUTURE_ONLY_SECRET',
        }
        self.sample = {
            'sample_id': 'sample_a', 'frame_count': 2,
            'conditions': {'prompt': 'prompt.txt', 'target_whitebox': 'target_whitebox.mp4',
                           'reference_pairs': [{'rgb': 'ref.png', 'whitebox': 'whitebox.png'}],
                           'unrecognized_future_field': 'FUTURE_ONLY_SECRET'},
            'supervision': {'target_rgb': 'target_rgb.mp4', 'valid_pixels': 'valid.png'},
            'object_references': [self.edge], 'frame_state_index': 'frames.json',
        }
        self.write_json('frames.json', [{'server_time_estimate_ns': t} for t in (100, 200, 300)])
        self.save_sample()
        self.video_mock = patch('memory.gta_training_dataset.read_video', side_effect=self.video)
        self.video_mock.start()
        self.addCleanup(self.video_mock.stop)

    def video(self, path):
        self.assertTrue(path.is_absolute())
        return self.rgb.copy() if path.name == 'target_rgb.mp4' else self.whitebox.copy()

    def write_image(self, name, image):
        self.assertTrue(cv2.imwrite(str(self.sample_root / name), image))

    def write_json(self, name, value):
        (self.sample_root / name).write_text(json.dumps(value), encoding='utf-8')

    def save_sample(self):
        self.write_json('sample.json', self.sample)

    def load(self):
        self.save_sample()
        return GTAObjectMemoryDataset(self.root)[0]

    def test_conditions_are_whitelisted_and_assets_are_consumable(self):
        result = self.load()
        conditions = result['conditions']
        self.assertEqual(set(conditions), {'prompt', 'reference_pairs', 'target_whitebox', 'object_references'})
        pair = conditions['reference_pairs'][0]
        self.assertEqual(set(pair), {'reference_id', 'rgb', 'whitebox'})
        self.assertEqual(pair['reference_id'], 'sample_a:ref0')
        edge = conditions['object_references'][0]
        self.assertEqual(set(edge), {
            'object_id', 'category', 'reference_id', 'reference_bbox_xyxy',
            'target_projected_bboxes', 'observed_at_ns', 'used_at_ns', 'used_until_ns',
            'reference_mask', 'reference_crop'})
        self.assertEqual(edge['reference_mask'].dtype, np.bool_)
        self.assertFalse(edge['reference_mask'][0].any())
        self.assertTrue(edge['reference_mask'][1:].all())
        np.testing.assert_array_equal(edge['reference_crop'], pair['rgb'][0:4, 1:5])
        self.assertFalse(edge['reference_crop'][0].any())
        np.testing.assert_array_equal(pair['rgb'][1, 1], [90, 30, 10])
        np.testing.assert_array_equal(result['supervision']['target_rgb'], self.rgb)
        aux = result['auxiliary_object_features'][0]
        self.assertEqual(aux['reference_id'], pair['reference_id'])
        self.assertEqual(aux['embedding'].shape, (384,))

        def check(value):
            if isinstance(value, dict):
                self.assertFalse({'lifecycle', 'despawn_server_monotonic_ns', 'state_change_review',
                                  'embedding', 'unrecognized_future_field'} & value.keys())
                for child in value.values(): check(child)
            elif isinstance(value, list):
                for child in value: check(child)
            elif isinstance(value, str):
                self.assertNotIn('FUTURE_ONLY_SECRET', value)
                self.assertFalse(value.endswith(('.png', '.npy', '.mp4')))
        check(conditions)

    def test_explicit_reference_id_and_optional_embedding(self):
        self.sample['conditions']['reference_pairs'][0]['reference_id'] = 'historic-camera-B'
        self.edge['reference_id'] = 'historic-camera-B'
        del self.edge['embedding']
        result = self.load()
        self.assertEqual(result['conditions']['reference_pairs'][0]['reference_id'], 'historic-camera-B')
        self.assertEqual(result['auxiliary_object_features'], [])

    def test_loading_survives_cwd_change(self):
        old = Path.cwd()
        try:
            os.chdir(self.root.parent)
            dataset = GTAObjectMemoryDataset(self.root.name)
            os.chdir(self.sample_root)
            result = dataset[0]
            self.assertIsInstance(result['conditions']['object_references'][0]['reference_mask'], np.ndarray)
        finally:
            os.chdir(old)

    def test_unknown_reference_and_mismatched_asset_rejected(self):
        for field, value in [('reference_id', 'missing'), ('reference_rgb', 'whitebox.png')]:
            with self.subTest(field=field):
                old = self.edge[field]
                self.edge[field] = value
                with self.assertRaises(ValueError): self.load()
                self.edge[field] = old

    def test_duplicate_pairs_and_object_edges_rejected(self):
        pairs = self.sample['conditions']['reference_pairs']
        pairs[0]['reference_id'] = 'sample_a:ref0'
        pairs.append(copy.deepcopy(pairs[0]))
        with self.assertRaises(ValueError): self.load()
        pairs.pop()
        self.sample['object_references'].append(copy.deepcopy(self.edge))
        with self.assertRaises(ValueError): self.load()

    def test_reference_shape_and_mask_shape_rejected(self):
        for name in ('ref.png', 'whitebox.png', 'object_mask.png', 'valid.png'):
            with self.subTest(name=name):
                path = self.sample_root / name
                original = path.read_bytes()
                self.write_image(name, np.ones((2, 2, 3), dtype=np.uint8))
                with self.assertRaises(ValueError): self.load()
                path.write_bytes(original)

    def test_hud_only_mask_rejected(self):
        mask = np.zeros((4, 6), dtype=np.uint8)
        mask[0] = 255
        self.write_image('object_mask.png', mask)
        with self.assertRaises(ValueError): self.load()

    def test_invalid_boxes_rejected(self):
        for box in ([0, 0, 7, 4], [2, 0, 1, 4], [0, 0, float('nan'), 4], [-1, 0, 4, 4]):
            with self.subTest(box=box):
                self.edge['reference_bbox_xyxy'] = box
                with self.assertRaises(ValueError): self.load()
        self.edge['reference_bbox_xyxy'] = [1, 0, 5, 4]
        self.edge['target_projected_bboxes'].pop()
        with self.assertRaises(ValueError): self.load()

    def test_invalid_edge_and_frame_times_rejected(self):
        for field, value in [('observed_at_ns', 200), ('used_until_ns', 199),
                             ('observed_at_ns', 99), ('used_at_ns', float('nan'))]:
            with self.subTest(field=field, value=value):
                old = self.edge[field]
                self.edge[field] = value
                with self.assertRaises(ValueError): self.load()
                self.edge[field] = old
        self.write_json('frames.json', [{'server_time_estimate_ns': t} for t in (100, 300, 200)])
        with self.assertRaises(ValueError): self.load()

    def test_asset_path_escape_rejected(self):
        self.edge['reference_mask'] = '../outside.png'
        with self.assertRaises(ValueError): self.load()

    def test_invalid_embedding_rejected(self):
        for data in (np.ones(10), np.full(384, np.nan)):
            np.save(self.sample_root / 'embedding.npy', data)
            with self.assertRaises(ValueError): self.load()

    def test_video_pair_mismatch_rejected(self):
        self.whitebox = self.whitebox[:1]
        with self.assertRaises(ValueError): self.load()


if __name__ == '__main__':
    unittest.main()
