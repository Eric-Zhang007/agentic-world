"""Coordinate and asset-contract tests; these do not certify reconstruction quality."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from whitebox.geometry_assets import fit_ground, rotation_between, transform_points, upright_box


class GeometryAssetsTests(unittest.TestCase):
    def test_invalid_plane_threshold(self):
        for threshold in (float('nan'), float('inf'), 0, -1):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                fit_ground(np.zeros((100, 3)), threshold)

    def test_yaw_box_recovers_rotated_surface_extent(self):
        rng = np.random.default_rng(4)
        local = rng.uniform([-3, -1, -.4], [3, 1, .4], (10000, 3))
        angle = .7
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        center = np.array([4, 2, -8])
        position, size, yaw = upright_box(local @ rotation.T+center)
        np.testing.assert_allclose(position, center, atol=.04)
        np.testing.assert_allclose(size, [5.76, 1.92, .768], atol=.07)
        self.assertLess(abs(np.sin(yaw-angle)), .03)

    def test_ground_alignment_preserves_camera_projection(self):
        rng = np.random.default_rng(10)
        normal = np.array([.1, 1., .2]); normal /= np.linalg.norm(normal)
        level = rotation_between(normal, [0, 1, 0])
        np.testing.assert_allclose(level @ normal, [0, 1, 0], atol=1e-8)
        matrix = np.eye(4); matrix[:3, :3] = 2.3*level; matrix[:3, 3] = [2, 3, 4]
        points = rng.normal(size=(50, 3)); points[:, 2] += 10
        mapped = transform_points(points, matrix)
        restored = transform_points(mapped, np.linalg.inv(matrix))
        np.testing.assert_allclose(restored[:, :2]/restored[:, 2:],
                                   points[:, :2]/points[:, 2:], atol=1e-9)

    def test_ground_robust_to_outliers(self):
        rng = np.random.default_rng(8)
        points = rng.uniform(-3, 3, (1500, 3))
        points[:1200, 1] = .2*points[:1200, 0] + .01*rng.normal(size=1200)
        normal, center, report = fit_ground(points, .035)
        expected = np.array([-.2, 1, 0]); expected /= np.linalg.norm(expected)
        self.assertGreater(normal @ expected, .999)
        self.assertGreater(report['inlier_fraction'], .75)
        self.assertLess(abs(center @ normal), .015)

    def test_degenerate_ground_rejected(self):
        with self.assertRaises(ValueError):
            fit_ground(np.zeros((100, 3)), .01)


if __name__ == '__main__':
    unittest.main()
