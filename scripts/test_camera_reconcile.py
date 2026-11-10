from copy import deepcopy
import unittest

import numpy as np

from scripts import test_video_reconcile as base
from whitebox.video_reconcile import build_correction
from whitebox.motion import simulate
from whitebox.world import scene_from_db


class CameraReconcileTests(unittest.TestCase):
    tearDown = base.VideoReconcileTests.tearDown
    commit = base.VideoReconcileTests.commit

    def setUp(self):
        base.VideoReconcileTests.setUp(self)
        self.review['camera_correction'] = {
            'evidence': ['fixture: static background shifts together'],
            'uncertainty': ['inferred coarse translation, not measured meters'],
            'keyframes': [{'frame_index': 0, 'position_offset': [0,0,0], 'target_offset': [0,0,0]},
                          {'frame_index': 71, 'position_offset': [.2,0,0], 'target_offset': [.2,0,0]}]}

    def test_camera_commit_and_next_segment_preserve_actors_and_history(self):
        ordinary = deepcopy(self.review)
        ordinary.pop('camera_correction')
        _, expected, _, _ = build_correction(self.before, self.timeline, ordinary)
        self.db.set_meta('whitebox_history', [{'prior': 'untouched'}])
        corrected, after, _ = self.commit()
        self.assertEqual(after['entities'], expected['entities'])
        np.testing.assert_allclose(np.subtract(after['camera']['position'], expected['camera']['position']), [.2,0,0])
        self.assertEqual(corrected['frames'][0]['camera'], self.timeline['frames'][0]['camera'])
        current = scene_from_db(self.db)
        self.assertEqual(current['camera'], after['camera'])
        wait = {'actor_id':'player', 'phases':[{'type':'wait','duration_sec':1,'criterion':'hold'}]}
        next_frames, _, _ = simulate(current, wait, self.db.get_meta('tick'))
        self.assertEqual(next_frames['frames'][0]['camera'], after['camera'])
        self.assertEqual(self.db.get_meta('whitebox_history')[0], {'prior':'untouched'})

    def test_invalid_camera_cannot_commit(self):
        invalid = [
            {'keyframes': []}, {'uncertainty': []},
            {'keyframes': [{'frame_index':70,'position_offset':[0,0,0],'target_offset':[0,0,0]}]},
            {'keyframes': [{'frame_index':71,'position_offset':[float('nan'),0,0],'target_offset':[0,0,0]}]},
            {'keyframes': [{'frame_index':71,'position_offset':[0,0,0],'target_offset':[0,0,0],'fov_offset':180}]},
        ]
        for patch in invalid:
            with self.subTest(patch=patch):
                review = deepcopy(self.review)
                review['camera_correction'].update(patch)
                with self.assertRaises(ValueError):
                    self.commit(review)
                self.assertEqual(scene_from_db(self.db), self.before)
                self.assertEqual(self.db.get_meta('whitebox_history'), [])


if __name__ == '__main__':
    unittest.main()
