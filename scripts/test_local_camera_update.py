from copy import deepcopy
import unittest

from scripts.test_scene_refinement import SceneRefinementTests
from whitebox.camera_reconcile import correct_cameras, effective_fov, update_camera
from whitebox.scene_refinement import apply_proposal


class LocalCameraUpdateTests(SceneRefinementTests):
    def test_projection_and_both_update_paths(self):
        camera = self.before['camera']
        camera['intrinsics'] = dict(fx=700., fy=600., width=960, height=640, cx=480, cy=320)
        original = deepcopy(self.before)
        target_fov = effective_fov(camera) + 10
        legacy = deepcopy(camera)
        legacy['fov'] = target_fov
        self.assertEqual(effective_fov(legacy), effective_fov(camera))
        candidate = apply_proposal(self.before, dict(camera={'fov': target_fov},
                                  evidence=['fixture'], uncertainty=['fixture']))
        frames = [dict(time=1., camera=deepcopy(camera))]
        correct_cameras(self.before, frames, 1, dict(evidence=['fixture'], uncertainty=['fixture'],
            keyframes=[dict(frame_index=0, position_offset=[0,0,0], target_offset=[0,0,0], fov_offset=10)]))
        self.assertEqual(candidate['camera'], frames[0]['camera'])
        self.assertAlmostEqual(effective_fov(frames[0]['camera']), target_fov)
        k = frames[0]['camera']['intrinsics']
        self.assertAlmostEqual(k['fx']/k['fy'], 700/600)
        self.assertEqual(k['cx'], 480)
        self.assertEqual(self.before, original)

    def test_up_and_zero_offset(self):
        camera = dict(position=[0,0,0], target=[0,1,0], up=[0,0,1], fov=50)
        self.assertEqual(update_camera(camera, {}), camera)
        with self.assertRaises(ValueError):
            update_camera(camera, {'target': [0,0,1]})
        del camera['up']
        self.assertEqual(update_camera(camera, {}), camera)
        camera['intrinsics'] = dict(fx=700, fy=600, width=960, height=640)
        frames = [dict(time=1., camera=deepcopy(camera))]
        correct_cameras({'camera': camera}, frames, 1, dict(evidence=['fixture'], uncertainty=['fixture'],
            keyframes=[dict(frame_index=0, position_offset=[0,0,0], target_offset=[0,0,0])]))
        self.assertEqual(frames[0]['camera'], camera)


if __name__ == '__main__':
    unittest.main()
