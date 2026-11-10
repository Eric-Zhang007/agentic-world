from copy import deepcopy
import unittest
import numpy as np
from scripts.test_whitebox_world import fixture
from whitebox.motion import prepare_scene, point_segment_distance
from whitebox.image_alignment import landmark_points, project_points
from whitebox.observation_alignment import align_observation


class AttachmentTests(unittest.TestCase):
    def setup_scene(self, grip='both'):
        scene = fixture()
        scene['entities'][0]['grip'] = grip
        scene = prepare_scene(scene)
        scene['camera'] = dict(position=[0,4,10], target=[0,0,-3], fov=60)
        anchors = []
        for i, pos in enumerate([[-3,0,-2],[0,0,-5],[4,0,-3],[1,2,-6]]):
            item = deepcopy(scene['entities'][-1])
            item.update(id='anchor'+str(i), position=pos)
            scene['entities'].append(item)
            anchors.append(dict(id=item['id'], local_fraction=[0,0,0]))
        pixels, _ = project_points(landmark_points(scene, anchors), scene['camera'],800,600)
        for row, pixel in zip(anchors, pixels):
            row['pixel'] = pixel.tolist()
        pixels, _ = project_points([[2,0,-1]],scene['camera'],800,600)
        obs = dict(width=800,height=600,timestamp_sec=4,static_landmarks=anchors,
                   camera_translation_bounds=[[-2,2]]*3,actor_roots=[dict(
                       id='player',pixel=pixels[0].tolist(),contact_visibility='visible',
                       support_plane=dict(normal=[0,1,0],offset=0))])
        return scene, obs

    def test_held_staff_follows_and_preserves_grip(self):
        for grip in ('both','left','right'):
            with self.subTest(grip=grip):
                scene, obs = self.setup_scene(grip)
                scene['entities'][0]['pose']['head'][0] += .2
                original, original_obs = deepcopy(scene), deepcopy(obs)
                result, report = align_observation(scene,obs)
                ids = {e['id']: e for e in result['entities']}
                staff, actor = ids['staff'], ids['player']
                np.testing.assert_allclose(actor['position'],[2,0,-1],atol=1e-8)
                for key in ('a','b'):
                    np.testing.assert_allclose(np.array(staff['weapon'][key])-scene['entities'][3]['weapon'][key], [2,0,-1],atol=1e-8)
                np.testing.assert_allclose(staff['position'],(np.array(staff['weapon']['a'])+staff['weapon']['b'])/2)
                for side in ('left','right') if grip=='both' else (grip,):
                    self.assertLess(point_segment_distance(actor['pose'][side+'_hand'],staff['weapon']['a'],staff['weapon']['b']),1e-8)
                self.assertEqual(actor['grip'],grip)
                for joint,point in original['entities'][0]['pose'].items():
                    np.testing.assert_allclose(np.asarray(actor['pose'][joint])-point,[2,0,-1],atol=1e-8)
                for entity in scene['entities']:
                    if entity['id'] not in ('player','staff'):
                        self.assertEqual(ids[entity['id']],entity)
                self.assertEqual(scene,original)
                self.assertEqual(obs,original_obs)
                self.assertEqual(len(report['actors']),1)
                self.assertFalse(report['world_committed'])

    def test_unknown_and_duplicate_actor_ids(self):
        for invalid in ('unknown','duplicate'):
            scene, obs = self.setup_scene()
            original = deepcopy(scene)
            if invalid=='unknown':
                obs['actor_roots'][0]['id']='missing'
            else:
                obs['actor_roots'].append(deepcopy(obs['actor_roots'][0]))
            with self.assertRaisesRegex(ValueError,'Unknown|Duplicate'):
                align_observation(scene,obs)
            self.assertEqual(scene,original)


if __name__ == '__main__':
    unittest.main()
