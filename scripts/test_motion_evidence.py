import sys
from pathlib import Path
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception.motion_evidence import pair_motion, track_points


class MotionEvidenceTests(unittest.TestCase):
    def test_explicit_anchors_keep_identity_and_reject_blank(self):
        image=cv2.GaussianBlur(np.random.default_rng(4).integers(0,256,(160,240),dtype=np.uint8),(5,5),0)
        moved=cv2.warpAffine(image,np.float32([[1,0,3],[0,1,2]]),(240,160))
        points=np.array([[80,60],[120,100],[180,60]],float)
        result=track_points(image,moved,points)
        self.assertTrue(result['valid'].all())
        np.testing.assert_allclose(result['points'],points+[3,2],atol=.15)
        blank=track_points(np.zeros_like(image),np.zeros_like(image),points)
        self.assertFalse(blank['valid'].any())
        self.assertTrue(np.isnan(blank['points']).all())

    def test_empty_and_invalid_anchor_inputs(self):
        image=np.zeros((160,240),np.uint8)
        self.assertEqual(track_points(image,image,[])['points'].shape,(0,2))
        with self.assertRaises(ValueError):track_points(image,image,[[float('nan'),0]])
        with self.assertRaises(ValueError):track_points(image,image[:80],[[30,30]])

    def test_no_features_is_unknown(self):
        blank = np.zeros((160, 240), dtype=np.uint8)
        self.assertEqual(pair_motion(blank, blank)['status'], 'INSUFFICIENT_FEATURES')

    def test_translation_is_dominant_not_articulation(self):
        rng = np.random.default_rng(2)
        image = rng.integers(0, 256, (160, 240), dtype=np.uint8)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        shifted = cv2.warpAffine(image, np.float32([[1, 0, 3], [0, 1, 2]]), (240, 160))
        evidence = pair_motion(image, shifted)
        self.assertEqual(evidence['status'], 'MEASURED')
        self.assertAlmostEqual(evidence['dominant_affine'][0][2], 3, delta=.2)
        self.assertAlmostEqual(evidence['dominant_affine'][1][2], 2, delta=.2)
        self.assertLess(evidence['residual_p90_px'], .2)

    def test_background_mask_excludes_large_moving_foreground(self):
        rng = np.random.default_rng(9)
        image = cv2.GaussianBlur(rng.integers(0,256,(160,240),dtype=np.uint8),(5,5),0)
        changed = image.copy()
        changed[60:] = cv2.warpAffine(image,np.float32([[1,0,9],[0,1,0]]),(240,160))[60:]
        mask = np.zeros_like(image)
        mask[10:40,10:230] = 255
        evidence = pair_motion(image,changed,mask)
        self.assertEqual(evidence['status'],'MEASURED')
        self.assertAlmostEqual(evidence['dominant_affine'][0][2],0,delta=.2)

    def test_empty_background_mask_is_unknown(self):
        image = np.random.default_rng(3).integers(0,256,(160,240),dtype=np.uint8)
        self.assertEqual(pair_motion(image,image,np.zeros_like(image))['status'],'INSUFFICIENT_FEATURES')


if __name__ == '__main__':
    unittest.main()
