"""Regression coverage for box heights constrained by support planes."""
from copy import deepcopy
import unittest

import numpy as np

from scripts import test_multiview_scene_fit as base
from scripts.fit_scene_landmarks import fit_scene
from whitebox.image_alignment import landmark_points, project_points
from whitebox.world import scene_from_db


class SupportedSceneFitTests(unittest.TestCase):
    tearDown = base.MultiviewSceneFitTests.tearDown

    def setUp(self):
        base.MultiviewSceneFitTests.setUp(self)
        self.box = next(e for e in self.scene['entities'] if e['id'] == self.identity)
        self.box['rotation_y'] = .63
        self.support = self.box['position'][1] - self.box['size'][1] / 2
        self.height = self.box['size'][1] * 1.4
        self.data['support_heights'] = {self.identity: self.support}
        self.data['parameters'] = [dict(id=self.identity, field='size', axis=1,
                                       bounds=[self.height / 3, self.height * 2])]
        target = deepcopy(self.scene)
        box = next(e for e in target['entities'] if e['id'] == self.identity)
        box['size'][1] = self.height
        box['position'][1] = self.support + self.height / 2
        for view in self.data['views']:
            view.pop('heldout', None)
            rows = [dict(id=self.identity, local_fraction=[.2, y, -.2])
                    for y in (.5, -.5)]
            pixels, valid = project_points(landmark_points(target, rows),
                                          view['camera'], view['width'], view['height'])
            self.assertTrue(valid.all())
            for row, pixel in zip(rows, pixels):
                row['pixel'] = pixel.tolist()
            view['landmarks'] = rows

    def test_size_fit_keeps_support_with_inconsistent_bottom(self):
        for offset in (0, 25):
            with self.subTest(bottom_offset=offset):
                data = deepcopy(self.data)
                for view in data['views']:
                    view['landmarks'][1]['pixel'][1] += offset
                original_scene, original_data = deepcopy(self.scene), deepcopy(data)
                database_before = scene_from_db(self.db)
                fitted, report = fit_scene(self.scene, data)
                box = next(e for e in fitted['entities'] if e['id'] == self.identity)
                self.assertAlmostEqual(box['size'][1], self.height, places=5)
                self.assertAlmostEqual(box['position'][1] - box['size'][1]/2,
                                       self.support, places=10)
                for view in report['views']:
                    np.testing.assert_allclose(view['after_errors_px'], [0, offset], atol=1e-4)
                expected = deepcopy(self.scene)
                item = next(e for e in expected['entities'] if e['id'] == self.identity)
                item['size'][1] = box['size'][1]
                item['position'][1] = box['position'][1]
                self.assertEqual(fitted, expected)
                self.assertEqual(self.scene, original_scene)
                self.assertEqual(data, original_data)
                self.assertEqual(scene_from_db(self.db), database_before)
                self.assertIs(report['world_committed'], False)

    def test_direct_and_linked_vertical_translation_rejected(self):
        other = deepcopy(self.box)
        other['id'] = 'support_translation_driver'
        self.scene['entities'].append(other)
        for linked in (False, True):
            with self.subTest(linked=linked):
                data = deepcopy(self.data)
                parameter = dict(id=other['id'] if linked else self.identity,
                                 field='position', axis=1,
                                 bounds=[other['position'][1]-1, other['position'][1]+1])
                if linked:
                    parameter['linked_ids'] = [self.identity]
                data['parameters'].append(parameter)
                with self.assertRaises(ValueError):
                    fit_scene(self.scene, data)

    def test_invalid_support_ids_and_heights_rejected(self):
        nonbox = next(e['id'] for e in self.scene['entities'] if e['kind'] != 'box')
        mappings = [{'missing_support_box': 0}, {nonbox: 0}]
        mappings += [{self.identity: value} for value in
                     (float('nan'), float('inf'), float('-inf'))]
        for mapping in mappings:
            with self.subTest(mapping=mapping):
                data = deepcopy(self.data)
                data['support_heights'] = mapping
                with self.assertRaises(ValueError):
                    fit_scene(self.scene, data)

    def test_supported_attachment_child_rejected(self):
        parent = deepcopy(self.box)
        parent.update(id='support_attachment_parent', rotation_y=0)
        self.scene['entities'].append(parent)
        self.data['attachments'] = [dict(child=self.identity, parent=parent['id'],
                                         axis=2, sign=-1, gap=.01)]
        with self.assertRaises(ValueError):
            fit_scene(self.scene, self.data)

    def test_absent_and_empty_supports_preserve_unconstrained_fit(self):
        data = deepcopy(self.data)
        del data['support_heights']
        data['parameters'].append(dict(id=self.identity, field='position', axis=1,
                                      bounds=[self.box['position'][1]-self.height,
                                              self.box['position'][1]+self.height]))
        fitted, report = fit_scene(self.scene, data)
        empty, empty_report = fit_scene(self.scene, dict(data, support_heights={}))
        self.assertEqual(fitted, empty)
        np.testing.assert_allclose(report['after_errors_px'], 0, atol=1e-4)
        np.testing.assert_allclose(empty_report['after_errors_px'], report['after_errors_px'])


if __name__ == '__main__':
    unittest.main()
