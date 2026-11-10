"""Independent region tests and real reconstruction/SAM projection evaluation.

python -B scripts/test_geometry_regions.py
python -B scripts/test_geometry_regions.py --real --assets outputs/geometry_init/urban/coarse_assets

Real evaluation prints a JSON report without modifying source assets. The target
is the processed confidence-filtered SAM mask, NOT ground-truth building shape.
Projection scores are same-view silhouette proxies, not novel-view accuracy.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np
from PIL import Image

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from whitebox.geometry_assets import transform_points, upright_box
from whitebox.geometry_regions import deduplicate_masks, fit_facade_boxes


def facade(x0=-3., x1=3., y0=0., y1=4., z=10., nx=61, ny=41):
    x, y = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    return np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z)))


def project_boxes(boxes, matrix, extrinsic, intrinsic, shape, near=1e-5):
    """Rasterize union of convex box silhouettes with camera near clipping."""
    output = np.zeros(shape, dtype=np.uint8)
    signs = np.array(list(itertools.product((-1., 1.), repeat=3)))
    edges = [(i, j) for i in range(8) for j in range(i)
             if np.count_nonzero(signs[i] != signs[j]) == 1]
    inverse = np.linalg.inv(matrix)
    for box in boxes:
        c, s = np.cos(box['rotation_y']), np.sin(box['rotation_y'])
        rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        corners = signs * np.asarray(box['size']) / 2 @ rotation.T + box['position']
        model = transform_points(corners, inverse)
        camera = model @ extrinsic[:, :3].T + extrinsic[:, 3]
        clipped = list(camera[camera[:, 2] >= near])
        for i, j in edges:
            a, b = camera[i], camera[j]
            if (a[2] < near) != (b[2] < near):
                clipped.append(a + (b-a) * ((near-a[2]) / (b[2]-a[2])))
        if len(clipped) < 3:
            continue
        homogeneous = np.asarray(clipped) @ intrinsic.T
        xy = homogeneous[:, :2] / homogeneous[:, 2:]
        hull = cv2.convexHull(xy.astype(np.float32))
        h, w = shape
        viewport = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32)
        _, polygon = cv2.intersectConvexConvex(hull, viewport)
        if polygon is not None and len(polygon) >= 3:
            cv2.fillConvexPoly(output, np.rint(polygon).astype(np.int32), 1)
    return output.astype(bool)


def projection_metrics(predicted, target):
    intersection = int((predicted & target).sum())
    union = int((predicted | target).sum())
    fp, fn = int((predicted & ~target).sum()), int((target & ~predicted).sum())
    boundaries = []
    for mask in (predicted, target):
        eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                           borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        boundaries.append(mask & ~eroded)
    boundary_error = None
    if all(b.any() for b in boundaries):
        distances = [cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2,
                                           cv2.DIST_MASK_PRECISE) for b in boundaries]
        boundary_error = float((distances[0][boundaries[1]].mean()
                                + distances[1][boundaries[0]].mean()) / 2)
    return dict(iou=intersection / union if union else 1.,
                precision=intersection / int(predicted.sum()) if predicted.any() else None,
                recall=intersection / int(target.sum()) if target.any() else None,
                false_positive_pixels=fp, false_negative_pixels=fn,
                symmetric_difference_pixels=fp+fn,
                symmetric_boundary_mean_px=boundary_error)


class GeometryRegionsTests(unittest.TestCase):
    def test_diagonal_line_is_not_a_facade(self):
        for scale in (.1, 1., 10.):
            for angle in (0., .7, 1.5):
                t = np.linspace(-2, 2, 200)
                points = scale * np.column_stack((t*np.cos(angle), t, -t*np.sin(angle)))
                result = fit_facade_boxes(points, plane_threshold=.01*scale,
                                          cell_size=10*scale, max_patch_extent=10*scale)
                self.assertEqual(result['boxes'], [])
                self.assertEqual(len(result['unassigned_indices']), len(points))

    def test_rotated_facade_and_evidence(self):
        p = facade()
        yaw = .63
        c, s = np.cos(yaw), np.sin(yaw)
        r = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        p = p @ r.T + [7, 2, -4]
        before = p.copy()
        result = fit_facade_boxes(p)
        self.assertEqual(len(result['boxes']), 1)
        box = result['boxes'][0]
        self.assertAlmostEqual(result['coverage'], 1.)
        np.testing.assert_allclose(box['size'], [6, 4, .05], atol=1e-8)
        self.assertLess(abs(np.sin(box['rotation_y']-yaw)), 1e-8)
        self.assertIsNone(box['geometry_evidence']['hidden_thickness'])
        self.assertFalse(box['solid'])
        np.testing.assert_array_equal(p, before)

    def test_disconnected_coplanar_buildings(self):
        p = np.concatenate((facade(-9, -5), facade(5, 9)))
        result = fit_facade_boxes(p, max_patch_extent=100)
        self.assertEqual(len(result['boxes']), 2)
        self.assertAlmostEqual(result['coverage'], 1.)
        self.assertTrue(all(b['size'][0] <= 4.01 for b in result['boxes']))

    def test_two_vertical_planes_with_outliers(self):
        p = np.concatenate((facade(), facade(z=14)))
        rng = np.random.default_rng(9)
        p += rng.normal(0, .015, p.shape)
        p = np.concatenate((p, rng.uniform(-20, 20, (100, 3)), [[np.nan, 0, 0]]))
        result = fit_facade_boxes(p, plane_threshold=.08)
        self.assertGreater(result['coverage'], .95)
        ids = [i for b in result['boxes'] for i in b['point_indices']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(result['invalid_indices'], [len(p)-1])
        self.assertEqual(len(ids)+len(result['unassigned_indices'])+1, len(p))
        self.assertTrue(all(b['geometry_evidence']['residual_max'] <= .080001
                            for b in result['boxes']))

    def test_perpendicular_facades(self):
        first = facade()
        second = facade()
        second[:, [0, 2]] = second[:, [2, 0]]
        p = np.concatenate((first, second))
        result = fit_facade_boxes(p)
        self.assertEqual(len(result['boxes']), 2)
        self.assertEqual(result['coverage'], 1.)
        normals = [b['geometry_evidence']['plane_normal'] for b in result['boxes']]
        self.assertAlmostEqual(float(np.dot(*normals)), 0.)

    def test_horizontal_surface_is_not_a_facade(self):
        p = facade()
        p[:, [1, 2]] = p[:, [2, 1]]
        result = fit_facade_boxes(p)
        self.assertEqual(result['boxes'], [])
        self.assertEqual(len(result['unassigned_indices']), len(p))

    def test_hole_and_maximum_patch_extent(self):
        p = facade(-8, 8, 0, 12, nx=161, ny=121)
        p = p[~((abs(p[:, 0]) < 4) & (p[:, 1] > 3) & (p[:, 1] < 9))]
        result = fit_facade_boxes(p, max_patch_extent=5, cell_size=.3, min_fill=.8)
        self.assertGreater(result['coverage'], .9)
        for box in result['boxes']:
            self.assertLessEqual(max(box['size'][:2]), 5.000001)
            self.assertGreaterEqual(box['geometry_evidence']['grid_fill'], .8)
        k = np.array([[40., 0, 50], [0, 40, 10], [0, 0, 1]])
        e = np.column_stack((np.eye(3), np.zeros(3)))
        image = project_boxes(result['boxes'], np.eye(4), e, k, (80, 100))
        self.assertFalse(image[34, 50], 'Central occlusion must not become a giant box')

    def test_empty_degenerate_and_invalid(self):
        for p in (np.empty((0, 3)), np.zeros((100, 3)),
                  np.column_stack((np.arange(100), np.zeros((100, 2))))):
            self.assertEqual(fit_facade_boxes(p)['boxes'], [])
        for kwargs in ({'plane_threshold': np.nan}, {'min_points': 2},
                       {'max_planes': 1.5}, {'min_fill': 0}):
            with self.assertRaises(ValueError):
                fit_facade_boxes(facade(), **kwargs)

    def test_scale_equivariance_and_determinism(self):
        p = facade()
        first = fit_facade_boxes(p)
        self.assertEqual(first, fit_facade_boxes(p))
        scaled = fit_facade_boxes(p*7, plane_threshold=.15*7, cell_size=7,
                                  max_patch_extent=70, display_thickness=.35)
        np.testing.assert_allclose(np.array(scaled['boxes'][0]['position'])/7,
                                   first['boxes'][0]['position'])
        np.testing.assert_allclose(np.array(scaled['boxes'][0]['size'])/7,
                                   first['boxes'][0]['size'])

    def test_dedup_semantics_identity_and_containment(self):
        full = np.ones((20, 20), bool)
        almost = full.copy(); almost[:2] = False
        tiny = np.zeros_like(full); tiny[:5, :5] = True
        masks = [dict(id='a', category='Building.', mask=full, confidence=.9),
                 dict(id='b', category='building', mask=almost),
                 dict(id='nested', category='building', mask=tiny),
                 dict(id='wall', category='wall', mask=full),
                 dict(id='other_frame', category='building', mask=full, frame_id=1)]
        kept, report = deduplicate_masks(masks)
        self.assertEqual([m['id'] for m in kept], ['a', 'nested', 'wall', 'other_frame'])
        self.assertEqual(report['suppressed'][0]['kept_id'], 'a')
        self.assertTrue(masks[0]['mask'].all())
        identities = [dict(category='building', mask=full, entity_id=i) for i in (1, 2)]
        self.assertEqual(len(deduplicate_masks(identities)[0]), 2)
        anonymous = dict(category='building', mask=full, confidence=1.)
        self.assertEqual(len(deduplicate_masks([anonymous, *identities])[0]), 3)
        # Containment with area ratio .8 passes even though IoU is below .85.
        contained = full.copy(); contained[:4] = False
        self.assertEqual(len(deduplicate_masks([masks[0], dict(category='building',
                                                               mask=contained)])[0]), 1)

    def test_dedup_no_transitive_merging_or_unknown_semantics(self):
        masks = []
        for i in range(3):
            m = np.zeros((10, 14), bool); m[:, i:i+10] = True
            masks.append(dict(id=i, category='building', mask=m, confidence=3-i))
        kept, report = deduplicate_masks(masks, iou_threshold=.8, containment_threshold=1)
        self.assertEqual([m['id'] for m in kept], [0, 2])
        self.assertEqual(len(report['suppressed']), 1)
        for m in masks:
            m['category'] = 'unknown'
        self.assertEqual(len(deduplicate_masks(masks)[0]), 3)

    def test_mask_input_validation_and_empty_masks(self):
        empty = dict(category='building', mask=np.zeros((4, 4), bool))
        self.assertEqual(len(deduplicate_masks([empty, empty])[0]), 2)
        for mask in (np.zeros(4), np.full((4, 4), np.nan), np.full((4, 4), 2)):
            with self.assertRaises(ValueError):
                deduplicate_masks([dict(category='building', mask=mask)])
        with self.assertRaises(ValueError):
            deduplicate_masks([empty], iou_threshold=0)
        with self.assertRaises(ValueError):
            deduplicate_masks([empty, dict(category='building', mask=np.zeros((5, 5)))])

    def test_projector_clips_near_plane(self):
        e = np.column_stack((np.eye(3), np.zeros(3)))
        k = np.array([[40., 0, 50], [0, 40, 50], [0, 0, 1]])
        box = dict(position=[0, 0, -5], size=[2, 2, 2], rotation_y=0)
        self.assertFalse(project_boxes([box], np.eye(4), e, k, (100, 100)).any())
        box['position'] = [0, 0, .5]
        self.assertTrue(project_boxes([box], np.eye(4), e, k, (100, 100)).all())


def evaluate_real(assets, fit_parameters):
    assets = assets.resolve()
    report = json.loads((assets / 'report.json').read_text(encoding='utf-8'))
    scene = json.loads((assets / 'scene.json').read_text(encoding='utf-8'))
    perception_path = Path(report['perception'])
    perception = json.loads(perception_path.read_text(encoding='utf-8'))
    with np.load(report['reconstruction'], allow_pickle=False) as archive:
        f = report['frame']
        points, conf = archive['points'][f], archive['depth_conf'][f].squeeze()
        image = archive['images'][f].transpose(1, 2, 0)
        e, k = archive['extrinsic'][f], archive['intrinsic'][f]
        metadata = json.loads(str(archive['metadata'].item()))
    record = metadata['preprocessing']['images'][f]
    h, w = points.shape[:2]
    warp = np.asarray(record['original_to_processed_pixel_centers'])
    # Replay the original compiler exactly, including mask erosion/confidence.
    with Image.open(perception_path.parent / 'input.png') as source:
        source = source.convert('RGB').resize(tuple(record['resized_wh']), Image.Resampling.BICUBIC)
        top = record['crop_top']
        source = source.crop((0, top, source.width, min(source.height, top+h)))
        canvas = Image.new('RGB', (w, h), 'white')
        canvas.paste(source, tuple(record['padding_ltrb'][:2]))
    if np.max(abs(np.asarray(canvas)/255.-image)) > .01:
        raise AssertionError('SAM source and reconstruction RGB mismatch')
    finite = np.isfinite(points).all(axis=2) & np.isfinite(conf)
    finite &= conf >= np.quantile(conf[finite], report['confidence_quantile'])
    masks = []
    with np.load(assets / 'masks.npz', allow_pickle=False) as saved:
        for i, raw in enumerate(perception['observations']):
            mask_path = perception_path.parent / Path(raw['mask_path'].replace('\\', '/')).name
            with Image.open(mask_path) as source:
                if list(source.size) != record['original_wh']:
                    raise AssertionError('Mask source size mismatch')
                mask = (np.asarray(source.convert('L')) > 0).astype(np.uint8)
            mask = cv2.warpPerspective(mask, warp, (w, h), flags=cv2.INTER_NEAREST)
            mask = (cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0) & finite
            identity = f'geo_{i:03d}'
            np.testing.assert_array_equal(mask, saved[identity], err_msg=identity)
            masks.append(dict(id=identity, category=raw['category'].strip().rstrip('.'),
                              confidence=raw['confidence'], mask=mask))
    facade_masks = [m for m in masks if m['category'] in ('building', 'wall', 'building wall')]
    kept, duplicates = deduplicate_masks(facade_masks)
    matrix = np.asarray(report['model_to_scene'])
    world = transform_points(points, matrix)
    entities = {entity['id']: entity for entity in scene['entities']}
    old_all = [entities[m['id']] for m in facade_masks]
    old_kept = [entities[m['id']] for m in kept]
    rows, new_all, effective_parameters = [], [], None
    supported_pixels = np.zeros((h, w), bool)
    for item in kept:
        cloud = world[item['mask']]
        old = entities[item['id']]
        center, size, yaw = upright_box(cloud)
        np.testing.assert_allclose(center, old['position'], atol=1e-6)
        np.testing.assert_allclose(size, old['size'], atol=1e-6)
        if abs(np.sin(yaw-old['rotation_y'])) > 1e-6:
            raise AssertionError('Stored PCA yaw does not match reconstruction')
        fitted = fit_facade_boxes(cloud, **fit_parameters)
        effective_parameters = fitted['parameters']
        support_ids = []
        for part, box in enumerate(fitted['boxes'], 1):
            box.update(id=item['id'] if part == 1 else f"{item['id']}__part{part:02d}",
                       source_mask_id=item['id'], category=item['category'],
                       color=np.median(image[item['mask']][box['point_indices']], axis=0).tolist())
            support_ids.extend(box['point_indices'])
        if support_ids:
            supported_pixels.ravel()[np.flatnonzero(item['mask'])[support_ids]] = True
        new_all.extend(fitted['boxes'])
        old_projection = project_boxes([old], matrix, e, k, (h, w))
        new_projection = project_boxes(fitted['boxes'], matrix, e, k, (h, w))
        rows.append(dict(id=item['id'], category=item['category'], points=len(cloud),
                         point_range=np.ptp(cloud, axis=0).tolist(),
                         old_size=old['size'], parts=len(fitted['boxes']),
                         supported_fraction=fitted['coverage'],
                         max_part_size=np.max([b['size'] for b in fitted['boxes']], axis=0).tolist()
                         if fitted['boxes'] else None,
                         before=projection_metrics(old_projection, item['mask']),
                         after=projection_metrics(new_projection, item['mask'])))
    target = np.logical_or.reduce([m['mask'] for m in facade_masks])
    output = dict(assets=str(assets), reconstruction=report['reconstruction'],
                  perception=str(perception_path), frame=report['frame'],
                  scale=report['scale'], scale_source=report['scale_source'],
                  verification=dict(sam_rgb_matches=True, all_saved_masks_exact=True,
                                    original_pca_boxes_match=True),
                  parameters=effective_parameters, duplicates=duplicates,
                  original_masks=len(facade_masks), retained_masks=len(kept),
                  parts=len(new_all), supported_union_fraction=float(supported_pixels[target].mean()),
                  aggregate={name: projection_metrics(project_boxes(boxes, matrix, e, k, (h, w)), target)
                             for name, boxes in [('original', old_all), ('dedup_only', old_kept),
                                                 ('facade_parts', new_all)]},
                  rows=rows,
                  limitations=['Same-view SAM silhouette proxy, not true geometry or novel-view validation.',
                               'Projection unions ignore inter-object depth occlusion.',
                               'Unassigned reconstruction points are explicitly omitted, not filled.',
                               'Display thickness is not measured hidden building depth.',
                               'World scale inherits the original assumed actor-height prior.'])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--assets', type=Path, default=ROOT/'outputs/geometry_init/urban/coarse_assets')
    parser.add_argument('--plane-threshold', type=float, default=.15)
    parser.add_argument('--cell-size', type=float, default=1.)
    parser.add_argument('--max-patch-extent', type=float, default=10.)
    parser.add_argument('--min-points', type=int, default=30)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(GeometryRegionsTests)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)
    if args.real:
        result = evaluate_real(args.assets, dict(plane_threshold=args.plane_threshold,
                               cell_size=args.cell_size, max_patch_extent=args.max_patch_extent,
                               min_points=args.min_points))
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
