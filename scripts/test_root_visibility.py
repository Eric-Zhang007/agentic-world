from copy import deepcopy
import unittest
import numpy as np

from scripts import test_observation_alignment_attachments as attachments
from whitebox.observation_alignment import align_observation


class RootVisibilityTests(unittest.TestCase):
    def test_skipped_preserves_entities_with_camera_change(self):
        for visibility in (None, 'unknown', 'occluded'):
            with self.subTest(visibility=visibility):
                scene, obs = attachments.AttachmentTests().setup_scene()
                row = obs['actor_roots'][0]
                if visibility is not None:
                    row['contact_visibility'] = visibility
                else:
                    row.pop('contact_visibility', None)
                row.pop('pixel')
                row.pop('support_plane')
                for field in ('position', 'target'):
                    scene['camera'][field][0] += .5
                original = deepcopy(scene)
                original_obs = deepcopy(obs)
                result, report = align_observation(scene, obs)
                self.assertEqual(result['entities'], original['entities'])
                self.assertNotEqual(result['camera'], original['camera'])
                self.assertEqual(report['actors'], [])
                self.assertEqual(report['skipped_actor_roots'], [dict(
                    id='player', reason='Contact visibility is not visible',
                    contact_visibility=visibility or 'unknown')])
                self.assertEqual(scene, original)
                self.assertEqual(obs, original_obs)

    def test_visible_preserves_attachment_translation(self):
        for grip in ('both', 'left', 'right'):
            scene, obs = attachments.AttachmentTests().setup_scene(grip)
            obs['actor_roots'][0]['contact_visibility'] = 'visible'
            original = deepcopy(scene)
            result, report = align_observation(scene, obs)
            before = {e['id']: e for e in scene['entities']}
            after = {e['id']: e for e in result['entities']}
            delta = np.array([2, 0, -1])
            for identity in ('player', 'staff'):
                np.testing.assert_allclose(np.array(after[identity]['position']) - before[identity]['position'], delta, atol=1e-8)
            for joint, point in before['player']['pose'].items():
                np.testing.assert_allclose(np.array(after['player']['pose'][joint]) - point, delta, atol=1e-8)
            for endpoint in ('a', 'b'):
                np.testing.assert_allclose(np.array(after['staff']['weapon'][endpoint]) - before['staff']['weapon'][endpoint], delta, atol=1e-8)
            self.assertEqual(after['player']['grip'], grip)
            self.assertEqual(report['skipped_actor_roots'], [])
            self.assertEqual(scene, original)

    def test_invalid_visibility(self):
        for value in ('Visible', '', None, 0, [], {}):
            scene, obs = attachments.AttachmentTests().setup_scene()
            obs['actor_roots'][0]['contact_visibility'] = value
            with self.assertRaisesRegex(ValueError, 'contact_visibility'):
                align_observation(scene, obs)

    def test_invalid_visible_pixels(self):
        for pixel in ([1], [[1, 2]], [1, 2, 3], [float('nan'), 1],
                      [1, float('inf')], [-1, 0], [0, -1], [800, 0], [0, 600]):
            with self.subTest(pixel=pixel):
                scene, obs = attachments.AttachmentTests().setup_scene()
                obs['actor_roots'][0].update(contact_visibility='visible', pixel=pixel)
                with self.assertRaisesRegex(ValueError, 'pixel'):
                    align_observation(scene, obs)


if __name__ == '__main__':
    unittest.main()
