"""Offline synthetic geometry checks: python -B scripts/test_geometry_alignment.py."""

from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
import json
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from whitebox.geometry_alignment import AlignmentError, fit_anchor_sim3
from scripts.evaluate_geometry_alignment import (
    build_static_mask, check_anchor_rgb, main as evaluate_main, replay_preprocessing,
    residual_breakdown,
)


class GeometryAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(734)
        self.new = self.rng.normal(size=(20, 25, 3))
        a, b = .61, -.37
        rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        self.rotation = rz @ ry
        self.scale = 2.3
        self.translation = np.array([4., -3., 1.2])
        self.old = self.truth(self.new)
        self.mask = np.ones(self.new.shape[:-1], dtype=bool)
        self.conf = np.ones(self.mask.shape)

    def truth(self, points):
        return self.scale * points @ self.rotation.T + self.translation

    def fit(self, new=None, old=None, **kwargs):
        options = dict(valid_static_mask=self.mask, new_confidence=self.conf,
                       old_confidence=self.conf, inlier_threshold=.04, max_rmse=.02,
                       ransac_iterations=256)
        options.update(kwargs)
        return fit_anchor_sim3(self.new if new is None else new,
                               self.old if old is None else old, **options)

    def test_exact_direction_full_batch_and_no_mutation(self):
        before = [x.copy() for x in (self.new, self.old, self.mask, self.conf)]
        result = self.fit()
        np.testing.assert_allclose(result.transform[:3, :3], self.scale * self.rotation, atol=1e-12)
        np.testing.assert_allclose(result.transform[:3, 3], self.translation, atol=1e-12)
        batch = self.rng.normal(size=(3, 8, 9, 3))
        np.testing.assert_allclose(result.apply(batch), self.truth(batch), atol=1e-12)
        self.assertTrue(result.inliers.all())
        self.assertAlmostEqual(np.linalg.det(result.transform[:3, :3] / result.metrics['scale']), 1.)
        for actual, original in zip((self.new, self.old, self.mask, self.conf), before):
            np.testing.assert_array_equal(actual, original)

    def test_outliers_noise_mask_and_repeatability(self):
        self.old += self.rng.normal(scale=.003, size=self.old.shape)
        bad = self.rng.choice(500, 150, replace=False)
        self.old.reshape(-1, 3)[bad] = self.rng.uniform(-30, 30, (150, 3))
        self.mask[:3] = False  # HUD
        self.mask[8:12, 8:15] = False  # dynamic object
        self.old[~self.mask] = 1e6
        result = self.fit()
        self.assertFalse(result.inliers[~self.mask].any())
        self.assertFalse(result.inliers.ravel()[bad].any())
        np.testing.assert_allclose(result.transform[:3, :3], self.scale * self.rotation, atol=.001)
        np.testing.assert_allclose(result.transform[:3, 3], self.translation, atol=.001)
        self.assertLess(result.metrics['rmse'], .007)
        np.testing.assert_array_equal(result.transform, self.fit().transform)

    def test_mask_excludes_dominant_moving_geometry(self):
        self.mask[:15] = False
        self.old[~self.mask] += [20, 10, -7]
        result = self.fit()
        self.assertEqual(result.metrics['inlier_count'], 125)
        np.testing.assert_allclose(result.apply(self.new), self.truth(self.new), atol=1e-12)

    def test_confidence_and_finite_filtering(self):
        new_conf, old_conf = self.conf.copy(), self.conf.copy()
        new_conf[0] = 0
        old_conf[1] = .1
        old_conf[2] = np.nan
        self.new[3] = np.nan
        self.old[4] = np.inf
        result = self.fit(new_confidence=new_conf, old_confidence=old_conf, min_confidence=.2)
        self.assertEqual(result.metrics['eligible_count'], 375)
        self.assertFalse(result.inliers[:5].any())

    def test_confidence_weights_reduce_biased_measurements(self):
        self.old[:10] += [.12, -.06, .04]
        weighted_conf = self.conf.copy()
        weighted_conf[:10] = .001
        plain = self.fit(inlier_threshold=.5, max_rmse=.3)
        weighted = self.fit(new_confidence=weighted_conf, inlier_threshold=.5, max_rmse=.3)
        plain_error = np.linalg.norm(plain.transform[:3, 3] - self.translation)
        weighted_error = np.linalg.norm(weighted.transform[:3, 3] - self.translation)
        self.assertLess(weighted_error, plain_error * .02)

    def test_multiple_paired_anchors(self):
        new = self.new.reshape(2, 10, 25, 3)
        old = self.old.reshape(new.shape)
        result = self.fit(new, old, valid_static_mask=self.mask.reshape(2, 10, 25),
                          new_confidence=self.conf.reshape(2, 10, 25),
                          old_confidence=self.conf.reshape(2, 10, 25))
        self.assertEqual(result.inliers.shape, (2, 10, 25))
        np.testing.assert_allclose(result.apply(new), old, atol=1e-12)

    def test_planar_noncollinear_support_is_valid(self):
        self.new[..., 2] = 0
        result = self.fit(old=self.truth(self.new))
        np.testing.assert_allclose(result.transform[:3, :3], self.scale * self.rotation, atol=1e-12)

    def test_degenerate_source_and_target_rejected(self):
        line = np.zeros_like(self.new)
        line[..., 0] = np.arange(500).reshape(20, 25)
        near_line = line + self.rng.normal(scale=1e-8, size=line.shape)
        for points in (np.ones_like(line), line, near_line):
            for new, old in ((points, self.truth(points)), (self.new, points), (points, self.old)):
                with self.subTest(spread=float(points.std())):
                    with self.assertRaises(AlignmentError):
                        self.fit(new, old, ransac_iterations=32)

    def test_insufficient_and_no_consensus_rejected(self):
        for n in (0, 2, 19):
            mask = np.zeros_like(self.mask)
            mask.ravel()[:n] = True
            with self.assertRaisesRegex(AlignmentError, 'insufficient eligible') as caught:
                self.fit(valid_static_mask=mask)
            self.assertEqual(caught.exception.metrics['eligible_count'], n)
        shuffled = self.old.reshape(-1, 3)[self.rng.permutation(500)].reshape(self.old.shape)
        with self.assertRaisesRegex(AlignmentError, 'consensus'):
            self.fit(old=shuffled)

    def test_high_residual_rejected_after_consensus(self):
        noisy = self.old + self.rng.normal(scale=.05, size=self.old.shape)
        with self.assertRaisesRegex(AlignmentError, 'residual') as caught:
            self.fit(old=noisy, inlier_threshold=.5, max_rmse=.01)
        self.assertGreater(caught.exception.metrics['rmse'], .05)
        self.assertEqual(caught.exception.metrics['inlier_count'], 500)

    def test_reflection_is_not_a_similarity(self):
        reflected = self.new * [-1, 1, 1]
        with self.assertRaises(AlignmentError):
            self.fit(old=reflected, min_inlier_ratio=.95)

    def test_contract_validation(self):
        with self.assertRaises(TypeError):
            fit_anchor_sim3(self.new, self.old, new_confidence=self.conf,
                            old_confidence=self.conf, inlier_threshold=.1, max_rmse=.1)
        for options in (dict(valid_static_mask=None), dict(valid_static_mask=self.conf),
                        dict(new_confidence=np.ones((1, 1))), dict(old_confidence=-self.conf),
                        dict(inlier_threshold=0), dict(max_rmse=np.nan),
                        dict(min_inliers=2), dict(ransac_iterations=0),
                        dict(min_inlier_ratio=1.1), dict(degeneracy_ratio=0)):
            with self.subTest(options=list(options)):
                with self.assertRaises(ValueError):
                    self.fit(**options)
        with self.assertRaises(ValueError):
            self.fit(old=self.old[:10])


class EvaluationContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'result.json'
        self.record = dict(original_wh=[40, 20], resized_wh=[20, 10],
                           output_wh=[20, 10], crop_top=0, padding_ltrb=[0, 0, 0, 0],
                           original_to_processed_pixel_centers=[
                               [.5, 0, -.25], [0, .5, -.25], [0, 0, 1]])

    def observation(self, category, name, box):
        pixels = np.zeros((20, 40), dtype=np.uint8)
        x0, y0, x1, y1 = box
        pixels[y0:y1, x0:x1] = 255
        Image.fromarray(pixels).save(self.path.parent / name)
        return dict(category=category, mask_path='/remote/server/' + name)

    def test_positive_only_dynamic_priority_hud_and_basename(self):
        observations = [self.observation('building wall', 'static.png', (0, 0, 30, 20)),
                        self.observation('person', 'person.png', (10, 4, 14, 16)),
                        self.observation('car', 'car.png', (20, 4, 24, 16))]
        static, dynamic, hud, report = build_static_mask(
            self.path, dict(observations=observations), self.record)
        self.assertTrue(static[5, 8])
        self.assertFalse(static[:, 15:].any())  # Unsegmented pixels stay excluded.
        self.assertFalse(static[dynamic].any())
        self.assertFalse(static[hud].any())
        self.assertTrue(dynamic[5, 5])
        self.assertTrue(dynamic[5, 10])
        self.assertTrue(hud[0, 0])
        self.assertEqual(report['static_pixels'], int(static.sum()))

    def test_no_static_segmentation_is_rejected(self):
        observations = [self.observation('person', 'person.png', (10, 4, 14, 16)),
                        dict(category='sky', mask_path='/not/read.png')]
        with self.assertRaisesRegex(ValueError, 'No positive'):
            build_static_mask(self.path, dict(observations=observations), self.record)

    def test_crop_padding_and_pixel_center_warp(self):
        pixels = np.arange(8 * 6, dtype=np.uint8).reshape(8, 6)
        record = dict(original_wh=[6, 8], resized_wh=[6, 8], output_wh=[8, 6],
                      crop_top=2, padding_ltrb=[1, 1, 1, 1],
                      original_to_processed_pixel_centers=[[1, 0, 1], [0, 1, -1], [0, 0, 1]])
        warped = replay_preprocessing(Image.fromarray(pixels), record, Image.Resampling.NEAREST)
        np.testing.assert_array_equal(warped[1:5, 1:7], pixels[2:6])
        self.assertFalse(warped[0].any())
        record['original_to_processed_pixel_centers'][0][2] = 0
        with self.assertRaisesRegex(ValueError, 'warp disagrees'):
            replay_preprocessing(Image.fromarray(pixels), record, Image.Resampling.NEAREST)

    def test_rgb_identity_including_sam_image(self):
        pixels = np.random.default_rng(23).integers(0, 256, (20, 40, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        image.save(self.path.parent / 'target.png')
        rgb = replay_preprocessing(image, self.record, Image.Resampling.BICUBIC)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255
        single = dict(images=tensor[None], metadata=dict(preprocessing=dict(images=[self.record])))
        joint = dict(images=np.stack([tensor, tensor]),
                     metadata=dict(preprocessing=dict(images=[self.record, self.record])))
        perception = dict(image='/remote/target.png', image_size=[40, 20])
        _, evidence = check_anchor_rgb(single, joint, 1, self.path, perception)
        self.assertTrue(evidence['sam_processed_exact'])
        joint['images'][1, 0, 0, 0] += .01
        with self.assertRaisesRegex(ValueError, 'Anchor RGB mismatch'):
            check_anchor_rgb(single, joint, 1, self.path, perception)
        joint['images'][1] = tensor
        Image.new('RGB', (40, 20)).save(self.path.parent / 'target.png')
        with self.assertRaisesRegex(ValueError, 'SAM source RGB'):
            check_anchor_rgb(single, joint, 1, self.path, perception)

    def test_cli_artifacts_on_acceptance_and_rejection(self):
        root = self.path.parent
        Image.new('RGB', (40, 20), (80, 90, 100)).save(root / 'target.png')
        observation = self.observation('road', 'road.png', (0, 0, 40, 20))
        self.path.write_text(json.dumps(dict(success=True, image='/remote/target.png',
                                             image_size=[40, 20], observations=[observation])))
        rng = np.random.default_rng(91)
        source = rng.normal(size=(10, 20, 3))
        rgb = np.broadcast_to(np.array([80, 90, 100], dtype=np.float32)[:, None, None] / 255,
                              (3, 10, 20))
        for accepted, frames, anchor in ((True, 2, 1), (False, 2, 1), (True, 3, 0)):
            target = 1.7 * source + [2, -1, 3] if accepted else rng.normal(size=source.shape)
            joint_points = rng.normal(size=(frames, *source.shape))
            joint_points[anchor] = source
            for name, points in (('single', target[None]), ('joint', joint_points)):
                n = len(points)
                images = np.zeros((n, *rgb.shape), dtype=np.float32)
                images[0 if name == 'single' else anchor] = rgb
                np.savez_compressed(root / (name + '.npz'), points=points,
                                    depth=np.ones((n, 10, 20, 1)), depth_conf=np.ones((n, 10, 20)),
                                    images=images,
                                    metadata=np.asarray(json.dumps(dict(preprocessing=dict(images=[self.record] * n)))))
            output = root / f'{accepted}_{frames}_{anchor}'
            with redirect_stdout(StringIO()):
                code = evaluate_main(['--single', str(root / 'single.npz'),
                                      '--joint', str(root / 'joint.npz'),
                                      '--anchor-index', str(anchor),
                                      '--perception', str(self.path), '--output', str(output)])
            self.assertEqual(code, 0 if accepted else 2)
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual(report['status'], 'accepted' if accepted else 'rejected')
            self.assertEqual(report['thresholds']['inlier_threshold'], .05)
            self.assertEqual(report['thresholds']['max_rmse'], .03)
            self.assertEqual(report['registration_accepted'], accepted)
            self.assertFalse(report['world_merge_ready'])
            self.assertEqual(report['local_validation_status'], 'not_performed')
            diagnostics = report['residual_diagnostics']
            self.assertEqual(sum(b['eligible_count'] for b in diagnostics['depth_bins'].values()),
                             report['fit_metrics']['eligible_count'])
            self.assertEqual(diagnostics['static_instances'][0]['id'], 'road')
            self.assertEqual(diagnostics['depth_bins']['far']['eligible_count'], 0)
            if not accepted:
                self.assertIsNone(diagnostics['depth_bins']['near']['after'])
            with np.load(output / 'registration.npz', allow_pickle=False) as registration:
                self.assertEqual(bool(registration['accepted']), accepted)
                self.assertFalse(bool(registration['world_merge_ready']))
                self.assertEqual(int(registration['anchor_index']), anchor)
                self.assertEqual('transform' in registration.files, accepted)
                if accepted:
                    np.testing.assert_allclose(registration['aligned_joint_points'][anchor], target, atol=1e-12)
                    np.testing.assert_allclose(registration['aligned_joint_points'],
                                               1.7 * joint_points + [2, -1, 3], atol=1e-12)
                else:
                    self.assertTrue(np.isnan(registration['after_error']).all())
            with Image.open(output / 'before_after_error.png') as figure:
                self.assertGreater(np.asarray(figure).std(), 10)
        for bad_anchor, message in ((1, 'Anchor RGB mismatch'), (-1, 'frame range'), (3, 'frame range')):
            with self.assertRaisesRegex(ValueError, message):
                evaluate_main(['--single', str(root / 'single.npz'), '--joint', str(root / 'joint.npz'),
                               '--perception', str(self.path), '--anchor-index', str(bad_anchor),
                               '--output', str(root / 'must_not_exist')])
        self.assertFalse((root / 'must_not_exist').exists())

    def test_depth_and_instance_diagnostics_include_far_outliers(self):
        observation = self.observation('building', 'building.png', (0, 0, 40, 20))
        observation['observation_id'] = 'building_17'
        _, _, _, mask_report = build_static_mask(
            self.path, dict(observations=[observation]), self.record)
        depth = np.ones((10, 20))
        depth[:, 10:15] = 3
        depth[:, 15:] = 10
        eligible = np.ones_like(depth, dtype=bool)
        eligible[0, 19] = False
        after = np.full_like(depth, .001)
        after[:, 10:15] = .1
        after[:, 15:] = 2
        after[0, 19] = 999  # An excluded dynamic/HUD pixel must not affect diagnostics.
        inliers = eligible & (after < .05)
        report = residual_breakdown(depth, eligible, inliers, after * 2, after,
                                    2., .05, True, mask_report, self.record)
        bins = report['depth_bins']
        self.assertEqual([bins[k]['eligible_count'] for k in ('near', 'mid', 'far')], [100, 50, 49])
        self.assertEqual(bins['far']['after']['p95'], 2.)
        self.assertEqual(bins['far']['after']['rmse'], 2.)
        self.assertEqual(bins['far']['after_error_over_pixel_depth']['p95'], .2)
        self.assertEqual(bins['far']['after_p95_over_global_median_depth'], 1.)
        self.assertEqual(bins['far']['inlier_ratio'], 0.)
        self.assertEqual(bins['far']['after_fraction_above_inlier_threshold'], 1.)
        self.assertEqual(report['static_instances'][0]['id'], 'building_17')
        self.assertEqual(report['static_instances'][0]['after']['maximum'], 2.)
        self.assertEqual(report['static_categories']['building']['eligible_count'], 199)


if __name__ == '__main__':
    unittest.main(verbosity=2)
