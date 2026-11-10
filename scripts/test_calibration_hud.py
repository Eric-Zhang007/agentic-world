import unittest
import numpy as np

from compiler.calibration import camera_matrix, calibrated_camera, gta_basis
from perception.hud import invalid_mask, clean_canny, clean_reference, overlap_fraction
from scripts.audit_episode_alignment import robust_clock, clock_time
from compiler.episode_alignment import bracket
from compiler.projection import project_object
from state.models import WorldObject


class CalibrationHudTests(unittest.TestCase):
    def test_camera_axes_and_stretched_recording(self):
        record={"camera_position":[0,0,0],"camera_rotation":[0,0,0],
                "camera_fov":90,"render_resolution":[1600,1000]}
        k,t=camera_matrix(record,1280,720,{})
        p=k@(t@np.array([0,5,0,1]))[:3]
        np.testing.assert_allclose(p[:2]/p[2],[640,360])
        self.assertAlmostEqual(k[0,0],400)
        self.assertAlmostEqual(k[1,1],360)
        np.testing.assert_allclose(gta_basis([12,4,50])@gta_basis([12,4,50]).T,np.eye(3),atol=1e-12)

    def test_reject_uncalibrated_and_out_of_range(self):
        with self.assertRaisesRegex(ValueError,"not passed"):
            calibrated_camera({},100,100,{"status":"rejected"},1)
        with self.assertRaisesRegex(ValueError,"outside"):
            calibrated_camera({},100,100,{"status":"validated","valid_server_time_range":[1,3]},4)

    def test_robust_clock_heldout_with_outliers(self):
        rows=[]
        for i in range(100):
            local=100000+i*2
            server=10+(local-100000)*1.00001
            rows.append({"sidecar_monotonic_ns":int(local*1e9),
                         "server_monotonic_ns":int((server+(1 if i%5==0 else 0))*1e9),
                         "round_trip_ms":1000 if i%5==0 else 2})
        clock=robust_clock(rows)
        self.assertLess(clock["holdout_abs_error_p95_ms"],.01)
        self.assertAlmostEqual(float(clock_time(clock,100010*1e9)),20.0001,places=5)

    def test_hud_canny_no_rectangle_or_overlay(self):
        image=np.zeros((100,160,3),np.uint8)
        image[70:90,5:20]=255
        image[20:60,80:100]=255
        profile={"rectangles_normalized":[[0,.65,.25,1]]}
        mask=invalid_mask(image.shape,profile)
        edge=clean_canny(image,mask)
        self.assertEqual(np.count_nonzero(edge[mask>0]),0)
        self.assertGreater(np.count_nonzero(edge[20:61,79:102]),0)
        filled=clean_reference(image,mask)
        np.testing.assert_array_equal(filled[mask==0],image[mask==0])

    def test_overlap_and_mask_clamping(self):
        mask=invalid_mask((100,160,3),{"rectangles_normalized":[[0,0,.25,.25]]},margin=3)
        self.assertEqual(overlap_fraction([0,0,10,10],mask),1)
        self.assertEqual(overlap_fraction([70,70,90,90],mask),0)
        self.assertEqual(overlap_fraction([20,20,20,20],mask),1)

    def test_interpolation_no_extrapolation_or_gaps(self):
        with self.assertRaises(ValueError):bracket(np.array([0,.1]),-.1)
        with self.assertRaises(ValueError):bracket(np.array([0,1]),.5)
        self.assertEqual(bracket(np.array([0,.1]),.05),(0,1,.5))

    def test_projection_anchor_does_not_edit_world_and_hud_excludes(self):
        obj=WorldObject("p","person",position=[0,0,2],
                        attributes={"projection_anchor_offset_xyz":[.2,0,0]})
        camera={"intrinsics":[[100,0,50],[0,100,50],[0,0,1]],
                "world_to_camera":np.eye(4).tolist(),"width":100,"height":100}
        result=project_object(obj,camera)
        np.testing.assert_allclose(result["center"],[60,50])
        self.assertEqual(obj.position,[0,0,2])
        camera["invalid_regions_normalized"]=[[.5,.4,.7,.6]]
        self.assertFalse(project_object(obj,camera)["visible"])


if __name__=="__main__":
    unittest.main()
