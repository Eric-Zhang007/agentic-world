import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception.depth_order import compare_depth_order


class DepthOrderTests(unittest.TestCase):
    def setUp(self):
        self.scene = {'camera': {'position': [0, 0, 0], 'target': [0, 0, -1], 'fov': 55},
                      'entities': [{'id': 'a', 'position': [0, 0, -1]},
                                   {'id': 'b', 'position': [0, 0, -3]}]}
        self.depth = np.array([[4., 1.], [np.nan, np.inf]])
        self.regions = {'a': [0, 0, 1, 1], 'b': [1, 0, 2, 1]}

    def test_sign_and_immutable(self):
        scene, regions, depth = copy.deepcopy(self.scene), copy.deepcopy(self.regions), self.depth.copy()
        for larger, expected in [(True, False), (False, True)]:
            pair = compare_depth_order(self.depth, self.scene, self.regions,
                                       nearer_is_larger=larger)['pairs'][0]
            self.assertEqual(pair['disagreement'], expected)
            self.assertEqual(pair['world_margin'], 2)
            self.assertEqual(pair['predicted_margin'], 3 if larger else -3)
        self.assertEqual(scene, self.scene)
        self.assertEqual(regions, self.regions)
        np.testing.assert_array_equal(depth, self.depth)

    def test_unavailable(self):
        for box, status in [(None, 'missing_region'), ([0, 0, 0, 1], 'empty_region'),
                            ([0, 1, 2, 2], 'nonfinite_region'), ([-1, 0, 1, 1], 'invalid_region')]:
            result = compare_depth_order(self.depth, self.scene, {'a': box, 'b': self.regions['b']}, nearer_is_larger=True)
            self.assertEqual(result['regions']['a']['status'], status)
            self.assertIsNone(result['pairs'][0]['disagreement'])
        result = compare_depth_order(self.depth, self.scene, {'absent': [0, 0, 1, 1]}, nearer_is_larger=True)
        self.assertEqual(result['regions']['absent']['status'], 'missing_entity')

    def test_partial_nonfinite_and_tie(self):
        result = compare_depth_order(self.depth, self.scene, {'a': [0, 0, 1, 2], 'b': [0, 0, 1, 1]}, nearer_is_larger=True)
        self.assertEqual(result['regions']['a']['finite_pixels'], 1)
        self.assertEqual(result['pairs'][0]['status'], 'tie')


if __name__ == '__main__':
    unittest.main()
