"""Synthetic contract/regression tests for local ground completion; no GT assets."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from whitebox.geometry_ground import GroundConfig, complete_ground, ground_mesh


class GroundTests(unittest.TestCase):
    def test_duplicate_observation_ids_do_not_merge_occluders(self):
        other = np.zeros_like(self.dynamic)
        other[20:40, 46:49] = True
        self.masks[0]['mask'] &= ~other
        self.masks.append(dict(id='actor', category='car', mask=other))
        duplicate = self.run_fill()
        self.masks[-1]['id'] = 'different'
        distinct = self.run_fill()
        np.testing.assert_array_equal(duplicate.inferred_mask, distinct.inferred_mask)
        self.assertFalse(distinct.inferred_mask.any())

    def test_collapsed_mesh_cells(self):
        from whitebox.geometry_ground import GroundCompletion
        for scale in (.01, 1., 100.):
            points = np.zeros((3, 3, 3))
            points[..., 0] = np.arange(3) * scale
            result = GroundCompletion(points, np.ones((3, 3), bool), np.zeros((3, 3), bool), [])
            triangles, tags = ground_mesh(result, max_edge=1000)
            self.assertEqual(len(triangles), 0)
            self.assertEqual(len(tags), 0)

    def setUp(self):
        h, w = 60, 80
        yy, xx = np.indices((h, w))
        self.intr = np.array([[100., 0, 40], [0, 100., -30], [0, 0, 1]])
        self.ext = np.column_stack((np.eye(3), [0, 1., 0]))
        rays = np.stack(((xx-40)/100, (yy+30)/100, np.ones((h, w))), -1)
        self.points = np.array([0, -1., 0])+rays/rays[..., 1:2]
        self.dynamic = np.zeros((h, w), bool)
        self.dynamic[20:40, 32:46] = True
        self.masks = [dict(id='road', category='road', mask=~self.dynamic),
                      dict(id='actor', category='person', mask=self.dynamic)]

    def run_fill(self, **kw):
        return complete_ground(self.points, self.masks, self.ext, self.intr, np.eye(4), **kw)

    def test_local_hole_and_mesh(self):
        self.points[self.dynamic] = np.nan
        original = self.points.copy()
        result = self.run_fill()
        np.testing.assert_array_equal(result.inferred_mask, self.dynamic)
        np.testing.assert_allclose(result.points[self.dynamic, 1], 0, atol=1e-10)
        np.testing.assert_array_equal(self.points, original)
        triangles, tags = ground_mesh(result)
        self.assertTrue(tags.any())
        self.assertTrue(np.isfinite(triangles).all())
        self.assertTrue(all(e['status'] == 'INFERRED' for e in result.evidence))

    def test_car(self):
        self.masks[1]['category'] = 'car'
        self.assertEqual(self.run_fill().inferred_mask.sum(), self.dynamic.sum())

    def test_raw_mask_survives_confidence_filtering(self):
        self.masks[1]['raw_mask'] = self.dynamic.copy()
        self.masks[1]['mask'] = np.zeros_like(self.dynamic)
        result = self.run_fill(valid=~self.dynamic)
        np.testing.assert_array_equal(result.inferred_mask, self.dynamic)
        self.assertEqual(result.report()['accepted_regions'], 1)

    def test_raw_building_blocks_even_when_filtered_empty(self):
        self.masks.append(dict(category='building', mask=np.zeros_like(self.dynamic), raw_mask=self.dynamic))
        self.assertFalse(self.run_fill().inferred_mask.any())

    def test_narrow_hole_full_resolution_mesh(self):
        self.dynamic[:] = False
        self.dynamic[21:39, 33:35] = True
        self.masks[0]['mask'] = ~self.dynamic
        result = self.run_fill()
        triangles, tags = ground_mesh(result)
        self.assertTrue(tags.any())
        self.assertEqual(int(result.inferred_mask.sum()), 36)

    def test_bounded_silhouette_margin(self):
        self.masks[0]['mask'][20:40, 29:49] = False
        result = self.run_fill()
        self.assertTrue(result.inferred_mask.any())
        self.assertEqual(int(result.inferred_mask.sum()), 400)
        self.masks[0]['mask'][20:40, 10:60] = False
        self.assertFalse(self.run_fill().inferred_mask.any())

    def test_no_dynamic_and_building(self):
        self.masks[1]['category'] = 'building'
        self.assertFalse(self.run_fill().inferred_mask.any())

    def test_building_overlap_excluded(self):
        building = self.dynamic.copy()
        building[:, :39] = False
        self.masks.append(dict(category='building', mask=building))
        self.assertFalse(self.run_fill().inferred_mask.any())

    def test_unknown_corridor_not_bridged(self):
        self.masks[0]['mask'][20:40, 46:] = False
        self.assertFalse(self.run_fill().inferred_mask.any())

    def test_different_surfaces_not_bridged(self):
        right = self.masks[0]['mask'].copy()
        right[:, :46] = False
        self.masks[0]['mask'][:, 46:] = False
        self.masks.append(dict(category='sidewalk', mask=right))
        self.assertFalse(self.run_fill().inferred_mask.any())

    def test_residual_rejected(self):
        self.points[..., 1] += np.random.default_rng(3).normal(0, .3, self.dynamic.shape)
        result = self.run_fill()
        self.assertFalse(result.inferred_mask.any())
        self.assertEqual(result.evidence[0]['reason'], 'plane_quality')

    def test_steep_plane_rejected(self):
        self.points[..., 1] = self.points[..., 0]
        self.assertFalse(self.run_fill().inferred_mask.any())

    def test_size_limits(self):
        for config in (replace(GroundConfig(), max_extent=.1),
                       replace(GroundConfig(), max_area=.001),
                       replace(GroundConfig(), max_gap_px=5)):
            with self.subTest(config=config):
                self.assertFalse(self.run_fill(config=config).inferred_mask.any())

    def test_confidence_support_only(self):
        valid = ~self.dynamic
        self.assertTrue(self.run_fill(valid=valid).inferred_mask.any())
        self.assertFalse(self.run_fill(valid=np.zeros_like(valid)).inferred_mask.any())

    def test_wrong_projection_boundary_rejected(self):
        self.intr[0, 2] += 40
        result = self.run_fill()
        self.assertFalse(result.inferred_mask.any())
        self.assertEqual(result.evidence[0]['reason'], 'boundary_discontinuity')

    def test_no_ground(self):
        self.masks = self.masks[1:]
        result = self.run_fill()
        triangles, tags = ground_mesh(result)
        self.assertEqual(triangles.shape, (0, 3, 3))
        self.assertEqual(len(tags), 0)


def verify_asset_support(root, output):
    """Query unchanged actor roots and neutral rig feet, without rig floor snapping."""
    from whitebox.surfaces import SurfaceWorld
    from whitebox.motion import rotation

    rows = []
    for name in ('coarse_assets', 'structured_assets'):
        folder = root/name
        scene = json.loads((folder/'scene.json').read_text(encoding='utf-8'))
        report = json.loads((folder/'report.json').read_text(encoding='utf-8'))
        triangles = np.fromfile(folder/scene['static_geometry']['collision_path'], dtype='<f4').reshape(-1, 3, 3)
        worlds = {'all': SurfaceWorld(triangles)}
        evidence_path = folder/'ground_evidence.npz'
        if evidence_path.exists():
            with np.load(evidence_path) as evidence:
                inferred = evidence['inferred_triangles']
            if len(inferred) != len(triangles):
                raise ValueError('Triangle labels do not match collision mesh')
            worlds['visible_only'] = SurfaceWorld(triangles[~inferred])
            worlds['inferred_only'] = SurfaceWorld(triangles[inferred])
        for actor in scene['entities']:
            if actor['id'] not in ('geo_000', 'geo_001'):
                continue
            position = np.array(actor['position'], float)
            probes = {'root': position}
            for side, sign in (('left_foot', -1), ('right_foot', 1)):
                probes[side] = position + rotation(actor.get('rotation_y', 0)) @ np.array([sign*.12*actor['size'][1], 0, 0])
            contacts = {}
            for label, point in probes.items():
                floors = {key: world.floor(point[0], point[2], position[1]) for key, world in worlds.items()}
                contacts[label] = dict(query=point.tolist(), floors=floors,
                                       root_to_floor=None if floors['all'] is None else float(position[1]-floors['all']))
            rows.append(dict(asset=name, actor=actor['id'], original_position=actor['position'],
                             original_size=actor['size'], contacts=contacts,
                             both_feet_supported=all(contacts[s]['floors']['all'] is not None for s in ('left_foot', 'right_foot')),
                             reconstruction=report['reconstruction'],
                             ground_surface=report.get('ground_surface', {})))
    result = dict(method='SurfaceWorld.floor defaults: max_step=.35, max_drop=.5; original root and neutral rig foot XZ; no snapping or relocation',
                  caveat='Each scene uses its own original coordinate frame; visible_only is the same-frame completion ablation. Feet are neutral rig contacts, not detected anatomical keypoints.',
                  results=rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps([{k: v for k, v in row.items() if k != 'ground_surface'} for row in rows], indent=2))


if __name__ == '__main__':
    if '--verify-assets' in sys.argv:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--verify-assets', type=Path, required=True)
        parser.add_argument('--output', type=Path, required=True)
        args = parser.parse_args()
        verify_asset_support(args.verify_assets, args.output)
    else:
        unittest.main()
