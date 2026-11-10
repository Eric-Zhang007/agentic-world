import unittest
from copy import deepcopy

import numpy as np
import json
from pathlib import Path

from whitebox.image_alignment import landmark_points
from whitebox.mask_alignment import projected_bounds
from whitebox.posture import HEIGHT_FACTORS
from whitebox.visibility import box_intersections


CASES = (
    ('standing', {'standing': 1.0}, 1.0),
    ('crouched', {'crouched': 1.0}, 0.65),
    ('blend', {'standing': 0.25, 'crouched': 0.75}, 0.25 + 0.75 * 0.65),
    ('low_guard', {'low_guard': 1.0}, 0.4),
)
CAMERA = dict(position=[0, 1, 0], target=[0, 1, -1], fov=90)


def actor(weights):
    return dict(id='actor', kind='actor', position=[0, 1, -10],
                size=[2, 4, 2], posture_weights=dict(weights))


class PostureProjectionTests(unittest.TestCase):
    def test_known_height_factors(self):
        for name, expected in [('standing', 1.0), ('crouched', 0.65), ('low_guard', 0.4)]:
            self.assertEqual(HEIGHT_FACTORS[name], expected)

    def test_landmarks_use_posture_height_and_preserve_root(self):
        for name, weights, factor in CASES:
            with self.subTest(posture=name):
                scene = {'entities': [actor(weights)]}
                rows = [dict(id='actor', local_fraction=p)
                        for p in ([0, 1, 0], [0, 0, 0], [0.5, 0.5, 0.5])]
                before = deepcopy((scene, rows))
                np.testing.assert_allclose(landmark_points(scene, rows),
                                           [[0, 1 + 4 * factor, -10], [0, 1, -10],
                                            [1, 1 + 2 * factor, -9]])
                self.assertEqual((scene, rows), before)

    def test_projected_top_shrinks_with_fixed_root_bottom(self):
        standing = projected_bounds(actor({'standing': 1}), CAMERA, 800, 600)
        for name, weights, factor in CASES:
            with self.subTest(posture=name):
                entity = actor(weights)
                camera = deepcopy(CAMERA)
                before = deepcopy((entity, camera))
                bounds = projected_bounds(entity, camera, 800, 600)
                catalog=json.loads((Path(__file__).resolve().parents[1]/'assets/gta/catalog.json').read_text())['assets']['gta_human']
                vertices=np.asarray([v for p in catalog['parts'] for v in p['vertices']])*([2,4*factor,2]/np.asarray(catalog['size']))
                uv=np.column_stack([400+300*vertices[:,0]/(10-vertices[:,2]),
                                    300-300*vertices[:,1]/(10-vertices[:,2])])
                np.testing.assert_allclose(bounds,np.r_[uv.min(0),uv.max(0)])
                if factor < 1:
                    self.assertGreater(bounds[1], standing[1])
                self.assertEqual(bounds[3], standing[3])
                self.assertEqual((entity, camera), before)

    def test_horizontal_rays_respect_top_and_root(self):
        for name, weights, factor in CASES:
            with self.subTest(posture=name):
                entity = actor(weights)
                before = deepcopy(entity)
                for height, hit in [(1.01, True), (0.99, False),
                                    (1 + 4 * factor - 0.01, True),
                                    (1 + 4 * factor + 0.01, False),
                                    (1 + 4 * 0.9, name == 'standing')]:
                    eye = np.array([0., height, 0.])
                    rays = np.array([[0., 0., -1.]])
                    distance = box_intersections(eye, rays, entity)[0]
                    self.assertAlmostEqual(distance, 8.7 if hit else np.inf)
                    np.testing.assert_array_equal(eye, [0, height, 0])
                    np.testing.assert_array_equal(rays, [[0, 0, -1]])
                self.assertEqual(entity, before)

    def test_box_ignores_posture_and_retains_center_origin(self):
        for name, weights, _ in CASES:
            with self.subTest(posture=name):
                box = actor(weights)
                box['kind'] = 'box'
                before = deepcopy(box)
                points = landmark_points({'entities': [box]},
                                         [dict(id='actor', local_fraction=[0, 0.5, 0])])
                np.testing.assert_allclose(points, [[0, 3, -10]])
                np.testing.assert_allclose(projected_bounds(box, CAMERA, 800, 600),
                                           [400 - 300 / 9, 300 - 600 / 9,
                                            400 + 300 / 9, 300 + 600 / 9])
                for height, expected in [(-0.99, 9.), (2.99, 9.), (-1.01, np.inf), (3.01, np.inf)]:
                    distance = box_intersections(np.array([0., height, 0.]),
                                                 np.array([[0., 0., -1.]]), box)[0]
                    self.assertEqual(distance, expected)
                self.assertEqual(box, before)


if __name__ == '__main__':
    unittest.main()
