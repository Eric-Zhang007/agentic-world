"""Offline maintenance regression using existing scene and database fixtures."""
from copy import deepcopy
import unittest

from scripts.test_scene_refinement import SceneRefinementTests
from whitebox.camera_reconcile import correct_cameras
from whitebox.scene_refinement import apply_proposal, commit_refinement
from whitebox.world import scene_from_db, validate_scene
from whitebox.motion import simulate


class LocalMaintenanceTests(SceneRefinementTests):
    def test_multistep_add_revise_roundtrip_and_simulate(self):
        before = scene_from_db(self.db)
        entity = deepcopy(next(e for e in before['entities'] if e['kind'] == 'box'))
        entity.update(id='local_new_static', position=[8, 3, 8], provenance='INFERRED')
        proposal = {'evidence':['fixture'], 'uncertainty':['estimated'],
                    'additions':[{'entity':entity, 'evidence':['fixture']}],
                    'camera':{'fov':55}}
        first = commit_refinement(self.db, before, proposal, 0, self.validation(), {})
        self.assertEqual(first['camera']['position'], before['camera']['position'])
        snapshot = deepcopy(self.db.get_meta('whitebox_representation_revisions'))
        proposal = {'evidence':['fixture'], 'uncertainty':['estimated'],
                    'edits':[{'id':entity['id'], 'position':[9, 3, 8], 'evidence':['fixture']}]}
        second = commit_refinement(self.db, first, proposal, 1, self.validation(), {})
        self.assertEqual(scene_from_db(self.db), second)
        self.assertEqual(self.db.get_object(entity['id']).position, [9, -8, 3])
        self.assertEqual(self.db.get_meta('whitebox_representation_revisions')[0], snapshot[0])
        self.assertEqual(self.db.get_meta('tick'), 0)
        timeline, _, _ = simulate(second, {'actor_id':'player', 'phases':[
            {'type':'wait', 'duration_sec':1, 'criterion':'hold'}]}, 0)
        self.assertTrue(timeline['frames'])
        self.assertIn(entity['id'], [e['id'] for e in timeline['frames'][-1]['entities']])

    def test_camera_failure_is_atomic(self):
        frames = [{'time':i+1, 'camera':deepcopy(self.before['camera'])} for i in range(2)]
        original = deepcopy(frames)
        correction = {'evidence':['fixture'], 'uncertainty':['estimate'], 'keyframes':[
            {'frame_index':1, 'position_offset':[0,0,0], 'target_offset':[0,0,0], 'fov_offset':120}]}
        with self.assertRaises(ValueError):
            correct_cameras(self.before, frames, 1, correction)
        self.assertEqual(frames, original)

    def test_invalid_identity_rejected(self):
        for identity in ('', ' ', 12, None):
            scene = deepcopy(self.before)
            scene['entities'][0]['id'] = identity
            with self.assertRaises(ValueError):
                validate_scene(scene)


if __name__ == '__main__':
    unittest.main()
