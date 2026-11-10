from copy import deepcopy
import unittest
import numpy as np
from whitebox.image_alignment import landmark_points, project_points
from whitebox.observation_alignment import align_observation


class ObservationAlignmentTests(unittest.TestCase):
    def test_camera_moves_not_static_world_and_actor_ground_updates(self):
        from scripts.test_scene_refinement import SceneRefinementTests
        fixture = SceneRefinementTests()
        fixture.setUp()
        try:
            scene = deepcopy(fixture.before)
        finally:
            fixture.tearDown()
        template = next(e for e in scene['entities'] if e['kind']=='box')
        anchors = []
        for i, pos in enumerate([[-3,0,-2],[0,0,-5],[4,0,-3],[1,2,-6]]):
            entity = deepcopy(template)
            entity.update(id='anchor_'+str(i),position=pos)
            scene['entities'].append(entity)
            anchors.append(dict(id=entity['id'],local_fraction=[0,0,0]))
        scene['camera'] = dict(position=[0,4,10],target=[0,0,-3],fov=60)
        original = deepcopy(scene)
        camera = deepcopy(scene['camera'])
        delta = [.5,.2,-.3]
        for field in ('position','target'):
            camera[field] = (np.array(camera[field])+delta).tolist()
        pixels, _ = project_points(landmark_points(scene, anchors),camera,800,600)
        for row, pixel in zip(anchors,pixels):
            row['pixel'] = pixel.tolist()
        actor = next(e for e in scene['entities'] if e['kind']=='actor')
        actor_pixel, _ = project_points([[2,0,-1]],camera,800,600)
        observation = dict(width=800,height=600,timestamp_sec=4,static_landmarks=anchors,
            camera_translation_bounds=[[-2,2]]*3,actor_roots=[dict(id=actor['id'],
            pixel=actor_pixel[0].tolist(),contact_visibility='visible',support_plane=dict(normal=[0,1,0],offset=0))])
        result, report = align_observation(scene,observation)
        np.testing.assert_allclose(report['camera_translation'],delta,atol=1e-8)
        np.testing.assert_allclose(report['actors'][0]['position'],[2,0,-1],atol=1e-8)
        self.assertEqual(scene,original)
        self.assertEqual([e for e in result['entities'] if e['kind'] not in ('actor','staff')],
                         [e for e in scene['entities'] if e['kind'] not in ('actor','staff')])
        self.assertFalse(report['world_committed'])


if __name__ == '__main__':
    unittest.main()
