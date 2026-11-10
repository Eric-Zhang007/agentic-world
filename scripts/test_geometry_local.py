import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from whitebox.geometry_local import assess_region


class LocalTests(unittest.TestCase):
    def setUp(self):
        self.points = np.random.default_rng(8).normal(size=(200, 3))

    def test_identity_preserved(self):
        result = assess_region(self.points, self.points+.001, threshold=.01)
        self.assertEqual(result['status'], 'CONSISTENT_KEEP_OLD')
        self.assertFalse(result['apply'])

    def test_translation_only_proposal(self):
        original = self.points.copy()
        result = assess_region(self.points, self.points+[.3, 0, 0], threshold=.01)
        self.assertEqual(result['status'], 'LOCAL_OFFSET_REVIEW')
        np.testing.assert_allclose(result['candidate_local_offset'], [-.3, 0, 0], atol=1e-12)
        self.assertFalse(result['apply'])
        np.testing.assert_array_equal(original, self.points)

    def test_shape_change_not_hidden_by_offset(self):
        result = assess_region(self.points, self.points*1.5, threshold=.01)
        self.assertEqual(result['status'], 'SHAPE_CONFLICT_KEEP_OLD')

    def test_holdout_failure(self):
        new = self.points.copy(); new[1::2] += .2
        result = assess_region(self.points, new, threshold=.01)
        self.assertEqual(result['status'], 'SHAPE_CONFLICT_KEEP_OLD')

    def test_sparse(self):
        self.assertEqual(assess_region(self.points[:3], self.points[:3], threshold=.01)['status'], 'INSUFFICIENT')


if __name__ == '__main__':
    unittest.main()
