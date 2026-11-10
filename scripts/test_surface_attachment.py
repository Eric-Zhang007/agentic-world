import math
import sys
import unittest
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from whitebox.surface_attachment import apply_surface_attachments


class SurfaceAttachmentTests(unittest.TestCase):
    def setUp(self):
        self.scene = {'entities': [
            {'id': 'p', 'kind': 'box', 'position': [1, 2, 3], 'size': [4, 6, 8], 'rotation_y': 0},
            {'id': 'c', 'kind': 'box', 'position': [9, 8, 7], 'size': [2, 4, 6], 'rotation_y': 0},
            {'id': 'other', 'kind': 'actor', 'pose': {'value': [1]}}]}
        self.relation = dict(child='c', parent='p', axis=0, sign=1, gap=0)

    def test_geometry_and_immutable(self):
        for yaw, childyaw, axis, sign, gap, expected in [
            (0, 0, 0, 1, 0, [4, 8, 7]),
            (math.pi/2, 0, 0, 1, 0, [9, 8, -2]),
            (0, math.pi/2, 0, 1, .5, [6.5, 8, 7]),
            (0, math.pi/4, 0, -1, 1, [-2-2*math.sqrt(2), 8, 7]),
            (0, 0, 1, -1, 2, [9, -5, 7]),
            (0, 0, 2, 1, 1, [9, 8, 11])]:
            with self.subTest(yaw=yaw, childyaw=childyaw, axis=axis):
                self.scene['entities'][0]['rotation_y'] = yaw
                self.scene['entities'][1]['rotation_y'] = childyaw
                before = deepcopy(self.scene)
                relation = dict(self.relation, axis=axis, sign=sign, gap=gap)
                result = apply_surface_attachments(self.scene, [relation])
                for actual, wanted in zip(result['entities'][1]['position'], expected):
                    self.assertAlmostEqual(actual, wanted)
                restored = deepcopy(result)
                restored['entities'][1]['position'] = before['entities'][1]['position']
                self.assertEqual(restored, before)
                result['entities'][2]['pose']['value'].append(2)
                self.assertEqual(self.scene, before)

    def test_malformed(self):
        for field, values in {'axis': [-1, 3, True, 1.0], 'sign': [0, 2, True],
                              'gap': [-1, float('inf'), float('nan'), '1', True],
                              'child': ['missing', 'p', 'other', []], 'parent': ['missing', 'other']}.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    apply_surface_attachments(self.scene, [dict(self.relation, **{field: value})])
        for relations in [[{}], [None], [self.relation, self.relation],
                          [self.relation, dict(self.relation, child='p', parent='c')],
                          [self.relation, dict(self.relation, child='p', parent='q')]]:
            scene = deepcopy(self.scene)
            scene['entities'].append(dict(scene['entities'][0], id='q'))
            with self.assertRaises(ValueError):
                apply_surface_attachments(scene, relations)

    def test_empty_copy(self):
        result = apply_surface_attachments(self.scene, [])
        self.assertEqual(result, self.scene)
        self.assertIsNot(result['entities'], self.scene['entities'])


if __name__ == '__main__':
    unittest.main()
