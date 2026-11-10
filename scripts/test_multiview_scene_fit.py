"""Synthetic coverage for shared geometry fitted against fixed cameras."""
from copy import deepcopy
import unittest

import numpy as np

from scripts import test_scene_refinement as base
from scripts.fit_scene_landmarks import fit_scene
from whitebox.image_alignment import landmark_points, project_points


class MultiviewSceneFitTests(unittest.TestCase):
    def setUp(self):
        base.SceneRefinementTests.setUp(self)
        self.scene = deepcopy(self.before)
        item = next(e for e in self.scene['entities'] if e['kind'] == 'box')
        self.identity = item['id']
        self.target = deepcopy(self.scene)
        target = next(e for e in self.target['entities'] if e['id'] == self.identity)
        target['position'] = (np.asarray(item['position']) + [.35, .25, -.3]).tolist()
        self.target_position = target['position']
        self.data = {'parameters': [
            {'id': self.identity, 'field': 'position', 'axis': axis,
             'bounds': [value - 1, value + 1]}
            for axis, value in enumerate(item['position'])], 'views': []}
        for offset, width, height in [([7, 5, 9], 640, 480), ([-8, 4, 6], 960, 540)]:
            camera = {'position': (np.asarray(item['position']) + offset).tolist(),
                      'target': list(item['position']), 'fov': 50, 'near': .05}
            view = {'camera': camera, 'width': width, 'height': height}
            for key, fractions in [('landmarks', [[0, 0, 0]]),
                                   ('heldout', [[.3, .2, -.2], [-.2, -.3, .3]])]:
                rows = [{'id': self.identity, 'local_fraction': fraction}
                        for fraction in fractions]
                pixels, valid = project_points(landmark_points(self.target, rows),
                                               camera, width, height)
                self.assertTrue(valid.all())
                for row, pixel in zip(rows, pixels):
                    row['pixel'] = pixel.tolist()
                view[key] = rows
            self.data['views'].append(view)

    tearDown = base.SceneRefinementTests.tearDown

    def test_shared_xyz_recovery_and_per_view_reports(self):
        original_scene, original_data = deepcopy(self.scene), deepcopy(self.data)
        fitted, report = fit_scene(self.scene, self.data)
        item = next(e for e in fitted['entities'] if e['id'] == self.identity)
        np.testing.assert_allclose(item['position'], self.target_position, atol=1e-5)
        self.assertEqual(self.scene, original_scene)
        self.assertEqual(self.data, original_data)
        self.assertEqual(fitted['camera'], original_scene['camera'])
        expected = deepcopy(original_scene)
        next(e for e in expected['entities'] if e['id'] == self.identity)['position'] = item['position']
        self.assertEqual(fitted, expected)
        self.assertIs(report['world_committed'], False)
        self.assertEqual(len(report['views']), 2)
        for view, result in zip(self.data['views'], report['views']):
            for scene, key in [(original_scene, 'before_errors_px'),
                               (fitted, 'after_errors_px')]:
                pixels, valid = project_points(landmark_points(scene, view['landmarks']),
                    view['camera'], view['width'], view['height'])
                self.assertTrue(valid.all())
                errors = np.linalg.norm(pixels - [r['pixel'] for r in view['landmarks']], axis=1)
                np.testing.assert_allclose(result[key], errors, atol=1e-7)
            self.assertGreater(max(result['before_errors_px']), 1)
            self.assertLess(max(result['after_errors_px']), 1e-4)
            self.assertEqual(len(result['heldout_errors_px']), 2)
            np.testing.assert_allclose(result['heldout_errors_px'], 0, atol=1e-4)

    def test_heldout_is_optional_and_not_used_for_fitting(self):
        data = deepcopy(self.data)
        del data['views'][0]['heldout']
        for row in data['views'][1]['heldout']:
            row['pixel'][0] += 100
        fitted, report = fit_scene(self.scene, data)
        item = next(e for e in fitted['entities'] if e['id'] == self.identity)
        np.testing.assert_allclose(item['position'], self.target_position, atol=1e-5)
        np.testing.assert_allclose(report['views'][1]['heldout_errors_px'], 100, atol=1e-4)

    def test_nonfinite_pixels_are_rejected(self):
        for key in ('landmarks', 'heldout'):
            for value in (float('nan'), float('inf'), float('-inf')):
                for axis in (0, 1):
                    with self.subTest(key=key, value=value, axis=axis):
                        data = deepcopy(self.data)
                        data['views'][1][key][0]['pixel'][axis] = value
                        with self.assertRaises(ValueError):
                            fit_scene(self.scene, data)

    def test_empty_views_are_rejected(self):
        for views in ([], [dict(self.data['views'][0], landmarks=[])]):
            with self.subTest(views=views):
                data = deepcopy(self.data)
                data['views'] = views
                with self.assertRaises(ValueError):
                    fit_scene(self.scene, data)


if __name__ == '__main__':
    unittest.main()
