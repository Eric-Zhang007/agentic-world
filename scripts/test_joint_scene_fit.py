"""Synthetic joint geometry/camera fitting with an anchored first view."""
from copy import deepcopy
import unittest

import numpy as np

from scripts import test_multiview_scene_fit as base
from scripts.fit_scene_landmarks import fit_scene
from whitebox.image_alignment import landmark_points, project_points


class JointSceneFitTests(unittest.TestCase):
    def setUp(self):
        base.MultiviewSceneFitTests.setUp(self)
        self.cameras = [deepcopy(v['camera']) for v in self.data['views']]
        for key in ('position', 'target'):
            self.cameras[1][key][0] += .45
        self.data['views'][1]['camera_bounds'] = [[-1, 1]] + [[0, 0] for _ in range(5)]
        anchor = deepcopy(next(e for e in self.scene['entities'] if e['id'] == self.identity))
        anchor['id'] = 'joint_fit_fixed_anchor'
        anchor['position'][0] += 3
        self.scene['entities'].append(anchor)
        self.target['entities'].append(deepcopy(anchor))
        for view, camera in zip(self.data['views'], self.cameras):
            view['landmarks'].extend(
                {'id': anchor['id'], 'local_fraction': fraction}
                for fraction in [[-.5, -.5, -.5], [.5, -.5, .5],
                                 [-.5, .5, .5], [.5, .5, -.5]])
            for key in ('landmarks', 'heldout'):
                pixels, valid = project_points(landmark_points(self.target, view[key]),
                                               camera, view['width'], view['height'])
                self.assertTrue(valid.all())
                for row, pixel in zip(view[key], pixels):
                    row['pixel'] = pixel.tolist()

    tearDown = base.MultiviewSceneFitTests.tearDown

    def test_joint_recovery_and_reports_preserve_inputs(self):
        original_scene, original_data = deepcopy(self.scene), deepcopy(self.data)
        fitted, report = fit_scene(self.scene, self.data)
        self.assertEqual(self.scene, original_scene)
        self.assertEqual(self.data, original_data)
        item = next(e for e in fitted['entities'] if e['id'] == self.identity)
        np.testing.assert_allclose(item['position'], self.target_position, atol=1e-5)
        expected = deepcopy(self.scene)
        next(e for e in expected['entities'] if e['id'] == self.identity)['position'] = item['position']
        self.assertEqual(fitted, expected)
        self.assertEqual(len(report['view_cameras']), 2)
        self.assertEqual(len(report['views']), 2)
        self.assertFalse(report['world_committed'])
        for view, truth, camera, errors in zip(self.data['views'], self.cameras,
                                              report['view_cameras'], report['views']):
            for key in ('position', 'target'):
                np.testing.assert_allclose(camera[key], truth[key], atol=1e-5)
            self.assertEqual(camera['fov'], truth['fov'])
            for rows_key, error_key in [('landmarks', 'after_errors_px'),
                                        ('heldout', 'heldout_errors_px')]:
                pixels, valid = project_points(landmark_points(fitted, view[rows_key]),
                                               camera, view['width'], view['height'])
                self.assertTrue(valid.all())
                expected_errors = np.linalg.norm(pixels - [r['pixel'] for r in view[rows_key]], axis=1)
                np.testing.assert_allclose(errors[error_key], expected_errors, atol=1e-7)
                np.testing.assert_allclose(expected_errors, 0, atol=1e-4)

    def test_heldout_does_not_affect_joint_fit(self):
        fitted, report = fit_scene(self.scene, self.data)
        changed = deepcopy(self.data)
        del changed['views'][0]['heldout']
        for row in changed['views'][1]['heldout']:
            row['pixel'][0] += 100
        other, other_report = fit_scene(self.scene, changed)
        self.assertEqual(other, fitted)
        self.assertEqual(other_report['view_cameras'], report['view_cameras'])
        np.testing.assert_allclose(other_report['views'][1]['heldout_errors_px'], 100, atol=1e-4)

    def test_absent_bounds_matches_explicit_fixed_cameras(self):
        data = deepcopy(self.data)
        for view, camera in zip(data['views'], self.cameras):
            view['camera'] = deepcopy(camera)
            view.pop('camera_bounds', None)
        fitted, report = fit_scene(self.scene, data)
        for view in data['views']:
            view['camera_bounds'] = [[0, 0] for _ in range(6)]
        explicit, explicit_report = fit_scene(self.scene, data)
        self.assertEqual(fitted, explicit)
        self.assertEqual(report['view_cameras'], explicit_report['view_cameras'])
        np.testing.assert_allclose(report['after_errors_px'], 0, atol=1e-4)

    def test_all_free_cameras_rejected(self):
        data = deepcopy(self.data)
        data['views'][0]['camera_bounds'] = deepcopy(data['views'][1]['camera_bounds'])
        with self.assertRaises(ValueError):
            fit_scene(self.scene, data)

    def test_invalid_camera_bounds_rejected(self):
        invalid = [[], [[0, 0]] * 5, [[0, 0]] * 7,
                   [[1, -1]] + [[0, 0]] * 5,
                   [[0, 1, 2]] + [[0, 0]] * 5,
                   [[float('nan'), 1]] + [[0, 0]] * 5,
                   [[0, float('inf')]] + [[0, 0]] * 5]
        for bounds in invalid:
            with self.subTest(bounds=bounds):
                data = deepcopy(self.data)
                data['views'][1]['camera_bounds'] = bounds
                with self.assertRaises(ValueError):
                    fit_scene(self.scene, data)


if __name__ == '__main__':
    unittest.main()
