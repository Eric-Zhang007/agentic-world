"""Compiler category-boundary regressions, including real CLI artifact checks."""
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compile_geometry_assets import (
    GEOMETRY_CATEGORIES, map_geometry_category, partition_geometry_masks,
)


class CategoryMappingTests(unittest.TestCase):
    def test_supported_roles_and_multiword_category(self):
        for category, role in GEOMETRY_CATEGORIES.items():
            with self.subTest(category=category):
                mapped = map_geometry_category({'category': category})
                self.assertEqual(mapped['geometry_role'], role)
                self.assertEqual(mapped['accepted'], role != 'excluded')
        self.assertEqual(map_geometry_category({'category': 'traffic light'})['geometry_role'], 'entity')

    def test_normalization_is_not_substring_matching(self):
        self.assertEqual(map_geometry_category({'category': '  SIDEWALK.  '})['canonical_category'], 'sidewalk')
        for label in ('roadside', 'wallpaper', 'building site', 'car park', 'red car', 'traffic', '', None, 12):
            with self.subTest(label=label):
                self.assertFalse(map_geometry_category({'category': label})['accepted'])

    def test_composite_labels_all_roles_and_orders(self):
        for a, b in itertools.permutations(('road', 'ground', 'sidewalk', 'person', 'tree', 'car', 'building', 'wall'), 2):
            for separator in (' ', '. ', '/', ' and '):
                with self.subTest(a=a, b=b, separator=separator):
                    mapped = map_geometry_category({'category': a + separator + b, 'confidence': 1.0})
                    self.assertEqual(mapped['geometry_role'], 'unresolved')
                    self.assertIsNone(mapped['canonical_category'])

    def test_unknowns_do_not_gain_solid_role(self):
        for label in ('unlisted object', 'river', 'water', 'sky', 'unknown'):
            self.assertFalse(map_geometry_category({'category': label})['accepted'])

    def test_partition_preserves_ids_masks_and_unmodified_input(self):
        mask = np.ones((8, 8), bool)
        raw = [{'id': 'geo_000', 'category': 'road sidewalk', 'mask': mask, 'raw_mask': mask,
                'confidence': .99, 'source_observation_id': 'sam_0'},
               {'id': 'geo_001', 'category': 'Tree.', 'mask': mask, 'confidence': .8}]
        accepted, blockers, candidates, mapping = partition_geometry_masks(raw)
        self.assertEqual([x['id'] for x in accepted], ['geo_001'])
        self.assertEqual(accepted[0]['category'], 'tree')
        self.assertEqual(raw[1]['category'], 'Tree.')
        self.assertEqual(blockers[0]['category'], 'unknown')
        self.assertIs(blockers[0]['raw_mask'], mask)
        self.assertFalse(candidates[0]['solid'])
        self.assertFalse(candidates[0]['accepted'])
        self.assertEqual(candidates[0]['source_observation_id'], 'sam_0')
        self.assertEqual(candidates[0]['raw_pixels'], 64)
        self.assertEqual(mapping[0]['raw_label'], 'road sidewalk')

    def fixture(self, root, with_ground):
        h, w = 64, 96
        yy, xx = np.indices((h, w))
        points = np.stack(((xx - w/2) / 20, np.ones((h, w)), 3 + yy / 20), axis=-1)
        image = np.full((h, w, 3), 128, np.uint8)
        Image.fromarray(image).save(root / 'input.png')
        masks = [('sidewalk' if with_ground else 'road car', yy >= 32),
                 ('road sidewalk', (xx > 30) & (xx < 60) & (yy > 36)),
                 ('tree', (xx < 20) & (yy < 24))]
        observations = []
        for index, (category, mask) in enumerate(masks):
            path = root / f'mask_{index}.png'
            Image.fromarray(mask.astype(np.uint8)*255).save(path)
            observations.append({'category': category, 'mask_path': str(path), 'confidence': .9,
                                 'observation_id': f'obs_{index}'})
        (root / 'result.json').write_text(json.dumps({'image_size': [w, h], 'observations': observations}))
        record = dict(original_wh=[w, h], resized_wh=[w, h], crop_top=0, padding_ltrb=[0, 0, 0, 0],
                      original_to_processed_pixel_centers=np.eye(3).tolist())
        np.savez(root / 'reconstruction.npz', points=points[None],
                 images=(image.astype(float)/255).transpose(2, 0, 1)[None],
                 depth_conf=np.ones((1, h, w)), extrinsic=np.eye(4)[:3][None],
                 intrinsic=np.array([[[60, 0, w/2], [0, 60, h/2], [0, 0, 1]]]),
                 metadata=json.dumps({'preprocessing': {'images': [record]}}))
        return subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/compile_geometry_assets.py'),
                               '--reconstruction', str(root / 'reconstruction.npz'),
                               '--perception', str(root / 'result.json'), '--output', str(root / 'candidate')],
                              capture_output=True, text=True, timeout=60)

    def test_cli_quarantines_unknown_without_changing_known_entity_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc = self.fixture(root, True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            out = root / 'candidate'
            scene = json.loads((out / 'scene.json').read_text())
            self.assertEqual([e['id'] for e in scene['entities']], ['geo_002'])
            self.assertEqual(scene['entities'][0]['category'], 'tree')
            candidates = json.loads((out / 'source_candidates.json').read_text())
            self.assertEqual([c['source_mask_id'] for c in candidates], ['geo_001'])
            self.assertFalse(candidates[0]['solid'])
            report = json.loads((out / 'report.json').read_text())
            self.assertEqual(report['ground_surface']['source_categories'], ['sidewalk'])
            self.assertEqual(report['category_mapping']['unresolved_observations'], 1)
            with np.load(out / 'semantic_candidate_masks.npz') as masks:
                raw = masks['geo_001__raw']
            with np.load(out / 'ground_evidence.npz') as evidence:
                self.assertFalse(np.any(raw & evidence['visible_mask']))
                self.assertFalse(np.any(raw & evidence['inferred_mask']))

    def test_cli_no_ground_fails_but_retains_semantic_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc = self.fixture(root, False)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn('segmented ground/road/sidewalk region is required', proc.stderr)
            out = root / 'candidate'
            self.assertFalse((out / 'scene.json').exists())
            candidates = json.loads((out / 'source_candidates.json').read_text())
            self.assertEqual(len(candidates), 2)
            self.assertTrue((out / 'category_mapping.json').is_file())
            self.assertTrue((out / 'semantic_candidate_masks.npz').is_file())


if __name__ == '__main__':
    unittest.main()
