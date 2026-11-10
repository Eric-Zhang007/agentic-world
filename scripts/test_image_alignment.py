from copy import deepcopy
import unittest
import numpy as np

from whitebox.image_alignment import project_points, pixels_to_plane
from scripts.fit_scene_landmarks import fit_scene


class ImageAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.camera = dict(position=[0, 0, 0], target=[0, 0, -1], fov=90)

    def test_projection_signs_and_near_plane(self):
        points = [[1, 1, -2], [0, 0, 1], [0, 0, -.01]]
        pixels, valid = project_points(points, self.camera, 800, 600)
        np.testing.assert_allclose(pixels[0], [550, 150])
        self.assertEqual(valid.tolist(), [True, False, False])
        self.assertTrue(np.isnan(pixels[1:]).all())

    def test_roll_and_anisotropic_intrinsics(self):
        self.camera.update(up=[1, 0, 0], intrinsics=dict(width=800,height=600,fx=500,fy=300,cx=400,cy=300))
        pixels, _ = project_points([[1, 1, -2]], self.camera, 800, 600)
        np.testing.assert_allclose(pixels[0], [150, 150])

    def test_overhead_default_up(self):
        self.camera['target'] = [0, -1, 0]
        pixels, _ = project_points([[1, -2, -1]], self.camera, 800, 600)
        np.testing.assert_allclose(pixels[0], [550, 150])

    def test_plane_roundtrip_with_roll_and_noninteger_viewport(self):
        camera = dict(position=[2,4,9], target=[0,0,0], up=[.2,1,.1],
                      intrinsics=dict(width=800,height=600,fx=511,fy=303,cx=400,cy=300))
        points = np.array([[0,0,0],[2,0,-3],[-4,0,2]])
        pixels, valid = project_points(points, camera, 1280,720)
        recovered, hits = pixels_to_plane(pixels,camera,1280,720)
        self.assertTrue(valid.all() and hits.all())
        np.testing.assert_allclose(recovered,points,atol=1e-12)

    def test_parallel_and_backward_ground_rays_are_not_positions(self):
        camera = dict(position=[0,2,0],target=[0,2,-1],fov=90)
        points, valid = pixels_to_plane([[400,300],[400,100],[400,500]],camera,800,600)
        self.assertEqual(valid.tolist(),[False,False,True])
        self.assertTrue(np.isnan(points[:2]).all())
        np.testing.assert_allclose(points[2],[0,0,-3])

    def test_sloped_plane_and_invalid_normal(self):
        camera = dict(position=[0,5,10],target=[0,0,0],fov=60)
        expected = [[1,2,-1]]
        pixels, _ = project_points(expected,camera,800,600)
        points, valid = pixels_to_plane(pixels,camera,800,600,normal=[0,2,2],offset=-2)
        self.assertTrue(valid[0])
        np.testing.assert_allclose(points,expected,atol=1e-12)
        with self.assertRaises(ValueError):
            pixels_to_plane(pixels,camera,800,600,normal=[0,0,0])

    def test_fit_shared_translation_preserves_offsets_and_input(self):
        from scripts.test_scene_refinement import SceneRefinementTests
        fixture = SceneRefinementTests()
        fixture.setUp()
        try:
            scene = deepcopy(fixture.before)
        finally:
            fixture.tearDown()
        static = next(e for e in scene['entities'] if e['kind']=='box')
        static.update(position=[8,3,8])
        linked = deepcopy(static)
        linked.update(id='linked', position=[8,4,8])
        scene['entities'].append(linked)
        scene['camera'] = dict(position=[0,10,20],target=[8,3,8],fov=60)
        original = deepcopy(scene)
        target, valid = project_points([[9,3,7]], scene['camera'], 800,600)
        self.assertTrue(valid[0])
        data = dict(width=800,height=600,landmarks=[dict(id=static['id'],local_fraction=[0,0,0],pixel=target[0].tolist())],
                    parameters=[dict(id=static['id'],field='position',axis=axis,bounds=[6,10],linked_ids=['linked']) for axis in (0,2)])
        result, report = fit_scene(scene, data)
        entities = {e['id']:e for e in result['entities']}
        np.testing.assert_allclose(entities[static['id']]['position'],[9,3,7],atol=1e-5)
        np.testing.assert_allclose(np.subtract(entities['linked']['position'],entities[static['id']]['position']),[0,1,0])
        self.assertEqual(scene, original)
        self.assertFalse(report['world_committed'])
        # A linked target cannot simultaneously have an independent transform.
        conflicting = deepcopy(data)
        conflicting['parameters'].append(dict(id='linked',field='position',axis=0,bounds=[6,10]))
        with self.assertRaisesRegex(ValueError,'overlap'):
            fit_scene(scene,conflicting)


if __name__ == '__main__':
    unittest.main()
