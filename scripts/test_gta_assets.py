"""Geometry invariants for portable source assets and renderer coordinates."""
import json
import unittest
from pathlib import Path

import numpy as np


class AssetTests(unittest.TestCase):
    def test_human_floor_and_forward(self):
        catalog = json.loads((Path(__file__).resolve().parents[1] / "assets/gta/catalog.json").read_text())
        parts = {p["name"]: np.asarray(p["vertices"]) for p in catalog["assets"]["gta_human"]["parts"]}
        self.assertEqual(len(parts), 13)
        self.assertAlmostEqual(min(p[:, 1].min() for p in parts.values()), 0)
        self.assertAlmostEqual(max(p[:, 1].max() for p in parts.values()), 1.8)
        self.assertGreater(parts["nose"][:, 2].mean(), parts["head"][:, 2].mean())

    def test_coordinate_transform_preserves_orientation(self):
        basis = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        self.assertAlmostEqual(np.linalg.det(basis), 1)
        np.testing.assert_array_equal(basis @ [0, 0, 1], [0, 1, 0])

    def test_vehicle_front_slope(self):
        catalog = json.loads((Path(__file__).resolve().parents[1] / "assets/gta/catalog.json").read_text())
        vertices = np.asarray(catalog["assets"]["gta_vehicle"]["parts"][0]["vertices"])
        self.assertAlmostEqual(vertices[6, 2], .4)
        self.assertAlmostEqual(vertices[4, 2], -1.6)


if __name__ == "__main__":
    unittest.main()
