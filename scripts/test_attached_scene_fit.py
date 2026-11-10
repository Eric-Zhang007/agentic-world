from copy import deepcopy
import unittest
import numpy as np

from scripts import test_multiview_scene_fit as base
from scripts.fit_scene_landmarks import fit_scene
from whitebox.surface_attachment import apply_surface_attachments
from whitebox.image_alignment import landmark_points, project_points


class AttachedSceneFitTests(unittest.TestCase):
    setUp = base.MultiviewSceneFitTests.setUp
    tearDown = base.MultiviewSceneFitTests.tearDown

    def test_fit_tangential_position_keeps_attachment(self):
        child = next(e for e in self.scene['entities'] if e['id'] == self.identity)
        parent = deepcopy(child)
        parent.update(id='support_wall', rotation_y=0,
                      position=(np.asarray(child['position'])+[0,0,1]).tolist(), size=[10,8,.2])
        self.scene['entities'].append(parent)
        relation = dict(child=child['id'], parent=parent['id'], axis=2, sign=-1, gap=.01)
        target = apply_surface_attachments(self.scene, [relation])
        target_child = next(e for e in target['entities'] if e['id'] == child['id'])
        target_child['position'][0] += .2
        data = deepcopy(self.data)
        data['attachments'] = [relation]
        data['parameters'] = [data['parameters'][0]]
        for view in data['views']:
            for group in ['landmarks', 'heldout']:
                uv, valid = project_points(landmark_points(target, view[group]),
                                           view['camera'], view['width'], view['height'])
                self.assertTrue(valid.all())
                for row, pixel in zip(view[group], uv):
                    row['pixel'] = pixel.tolist()
        original = deepcopy(self.scene)
        candidate, report = fit_scene(self.scene, data)
        result = next(e for e in candidate['entities'] if e['id'] == child['id'])
        np.testing.assert_allclose(result['position'], target_child['position'], atol=1e-6)
        self.assertEqual(self.scene, original)
        self.assertEqual(next(e for e in candidate['entities'] if e['id']==parent['id']), parent)
        self.assertLess(max(report['after_errors_px']), 1e-5)
        self.assertFalse(report['world_committed'])
        data['parameters'].append(dict(id=child['id'], field='position', axis=2,
                                       bounds=[child['position'][2]-1, child['position'][2]+1]))
        with self.assertRaisesRegex(ValueError, 'normal position'):
            fit_scene(self.scene, data)


if __name__ == '__main__':
    unittest.main()
