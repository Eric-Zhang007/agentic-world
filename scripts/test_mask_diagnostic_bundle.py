from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from scripts.diagnose_mask_alignment import diagnose
from scripts.test_whitebox_world import fixture
from whitebox.motion import prepare_scene, validate_geometry


class MaskDiagnosticBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bundle = Path(self.temp.name)
        self.scene = prepare_scene(fixture())
        self.image = np.zeros((96, 128, 3), dtype=np.uint8)
        mask = np.zeros((96, 128), dtype=np.uint8)
        mask[20:70, 30:90] = 255
        self.assertTrue(cv2.imwrite(str(self.bundle / 'input.png'), self.image))
        self.assertTrue(cv2.imwrite(str(self.bundle / 'mask.png'), mask))
        self.perception = {'observations': [
            {'observation_id': 'mapped', 'mask_path': '/old/bundle/mask.png'},
            {'observation_id': 'unmapped', 'mask_path': 'not-loaded.png'}]}
        self.mapping = {'mapped': 'wall'}

    def test_coverage_rendering_and_no_mutation(self):
        before = deepcopy((self.scene, self.perception, self.mapping))
        report, overlay = diagnose(self.scene, self.perception, self.mapping, self.bundle)
        self.assertEqual((self.scene, self.perception, self.mapping), before)
        self.assertEqual(report['unmapped_observation_ids'], ['unmapped'])
        self.assertEqual(report['unreviewed_world_ids'], ['player', 'guard', 'patrol', 'staff'])
        self.assertEqual(report['counts'], dict(observations=2, world_entities=5,
                         mapped_observations=1, reviewed_world_entities=1,
                         unmapped_observations=1, unreviewed_world_entities=4))
        self.assertFalse(report['world_committed'])
        self.assertEqual(report['geometry_errors'], validate_geometry(self.scene))
        self.assertEqual(len(report['results']), 1)
        self.assertIn('visibility', report['results'][0])
        self.assertEqual(overlay.shape, self.image.shape)
        self.assertTrue(np.any(overlay != self.image))
        np.testing.assert_array_equal(cv2.imread(str(self.bundle / 'input.png')), self.image)

    def test_empty_mapping_retains_all_ids(self):
        report, _ = diagnose(self.scene, self.perception, {}, self.bundle)
        self.assertEqual(report['results'], [])
        self.assertEqual(report['unmapped_observation_ids'], ['mapped', 'unmapped'])
        self.assertEqual(report['unreviewed_world_ids'], [e['id'] for e in self.scene['entities']])

    def test_group_mapping_counts_members_and_checks_unknowns(self):
        extra = deepcopy(next(e for e in self.scene['entities'] if e['id']=='wall'))
        extra['id']='wall_trim'
        self.scene['entities'].append(extra)
        report,_=diagnose(self.scene,self.perception,{'mapped':['wall','wall_trim']},self.bundle)
        self.assertEqual(report['counts']['mapped_observations'],1)
        self.assertEqual(report['counts']['reviewed_world_entities'],2)
        self.assertEqual(report['results'][0]['member_ids'],['wall','wall_trim'])
        self.assertNotIn('wall_trim',report['unreviewed_world_ids'])
        for members in [[],['wall','wall'],['wall','unknown']]:
            with self.assertRaises(ValueError):
                diagnose(self.scene,self.perception,{'mapped':members},self.bundle)

    def test_unknown_ids_rejected_without_mutation(self):
        for mapping, message in [({'unknown': 'wall'}, 'Unknown mapping observation ID'),
                                 ({'mapped': 'unknown'}, 'Unknown mapping world ID')]:
            before = deepcopy((self.scene, self.perception, mapping))
            with self.assertRaisesRegex(ValueError, message):
                diagnose(self.scene, self.perception, mapping, self.bundle)
            self.assertEqual((self.scene, self.perception, mapping), before)

    def test_duplicate_ids_rejected(self):
        for source, key, message in [(self.scene, 'entities', 'Duplicate world ID'),
                                     (self.perception, 'observations', 'Duplicate observation ID')]:
            source[key].append(deepcopy(source[key][0]))
            with self.assertRaisesRegex(ValueError, message):
                diagnose(self.scene, self.perception, self.mapping, self.bundle)
            source[key].pop()


if __name__ == '__main__':
    unittest.main()
