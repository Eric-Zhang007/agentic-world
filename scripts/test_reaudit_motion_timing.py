import unittest
import numpy as np
from scripts.reaudit_motion_timing import affine_fit, fits


class MotionTimingTests(unittest.TestCase):
    def test_recovers_rate_and_offset_with_heldout_motion(self):
        rng=np.random.default_rng(194)
        x=np.arange(480)
        signal=np.convolve(rng.normal(size=480),np.ones(7)/7,mode='same')
        scale,offset=1.10,-17
        proxy=np.interp((x-offset)/scale,x,signal,left=np.nan,right=np.nan)
        a=np.column_stack([signal,np.zeros(480),np.full(480,50)])
        b=np.column_stack([proxy,np.zeros(480),np.full(480,50)])
        result=affine_fit(a,b)
        self.assertAlmostEqual(result['scale'],scale,places=2)
        self.assertLessEqual(abs(result['offset']-offset),1)
        self.assertGreater(result['holdout_correlation'],.95)
        self.assertLess(result['holdout_zero_lag_correlation'],.5)

    def test_static_scene_is_not_timing_evidence(self):
        static=np.zeros((120,3))
        self.assertEqual(fits(static,static,0,120),[])
        self.assertEqual(affine_fit(static,static)['status'],'INSUFFICIENT_BACKGROUND_MOTION')


if __name__=='__main__':
    unittest.main()
