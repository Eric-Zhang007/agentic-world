"""Real SQLite regression tests; synthetic observations are not visual proof."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import test_video_reconcile as base
from state.db import WorldDB
from whitebox.scene_refinement import commit_refinement
from whitebox.video_reconcile import build_correction
from whitebox.world import scene_from_db
from harness.pi.world_tools import dispatch, save


class LocalExtensionHistoryTests(unittest.TestCase):
    def setUp(self):
        base.VideoReconcileTests.setUp(self)
        self.archive = Path(self.temp.name) / 'accepted.json'
        corrected, _, _, _ = build_correction(self.before, self.timeline, self.review)
        self.archive.write_text(json.dumps(corrected))
        self.artifacts['corrected_timeline'] = str(self.archive)
        base.VideoReconcileTests.commit(self)
        self.before = scene_from_db(self.db)
        self.history = deepcopy(self.db.get_meta('whitebox_history'))
        self.accepted_tick = self.db.get_meta('tick')
        self.archive_bytes = self.archive.read_bytes()

    tearDown = base.VideoReconcileTests.tearDown

    def proposal(self):
        entity = deepcopy(self.before['entities'][-1])
        entity.update(id='new_static', position=[8, 3, 8], size=[1, 2, 1], provenance='INFERRED')
        return {'evidence':['synthetic new observation'], 'uncertainty':['synthetic depth'],
                'additions':[{'entity':entity, 'evidence':['synthetic static structure']}]}

    def validation(self):
        return {'current':{'usable_for_camera_alignment':True},
                'joint':{'consistent':True, 'evidence':['synthetic']},
                'history':{'consistent':True, 'evidence':['synthetic']}}

    def snapshot(self):
        return '\n'.join(self.db.connection.iterdump())

    def assert_history(self):
        self.assertTrue(self.history)
        self.assertEqual(self.db.get_meta('whitebox_history'), self.history)
        self.assertEqual(self.db.get_meta('tick'), self.accepted_tick)
        self.assertEqual(self.archive.read_bytes(), self.archive_bytes)

    def test_addition_persists_without_rewriting_accepted_state(self):
        candidate = commit_refinement(self.db, self.before, self.proposal(), 1, self.validation(), {})
        self.assertEqual(candidate['entities'][:-1], self.before['entities'])
        self.db.connection.close()
        self.db = WorldDB(Path(self.temp.name) / 'world.sqlite')
        self.assertEqual(scene_from_db(self.db), candidate)
        self.assertEqual(self.db.get_meta('whitebox_revision'), 2)
        self.assertEqual(self.db.get_meta('whitebox_representation_revisions')[0]['before'], self.before)
        self.assert_history()

    def test_conflicting_additions_are_atomic(self):
        for conflict in ('duplicate', 'actor', 'audit', 'stale', 'observed'):
            with self.subTest(conflict=conflict):
                proposal = self.proposal()
                entity = proposal['additions'][0]['entity']
                validation = self.validation()
                revision = 1
                if conflict == 'duplicate': entity['id'] = self.before['entities'][-1]['id']
                if conflict == 'actor': entity['position'] = self.before['entities'][0]['position']
                if conflict == 'audit': validation = {}
                if conflict == 'stale': revision = 0
                if conflict == 'observed': entity['provenance'] = 'OBSERVED'
                before = self.snapshot()
                with self.assertRaises(ValueError):
                    commit_refinement(self.db, self.before, proposal, revision, validation, {})
                self.assertEqual(self.snapshot(), before)
                self.assert_history()

    def test_conflicting_edit_preserves_successful_addition(self):
        candidate = commit_refinement(self.db, self.before, self.proposal(), 1, self.validation(), {})
        proposal = {'evidence':['synthetic'], 'uncertainty':['depth'], 'edits':[
            {'id':'new_static', 'position':candidate['entities'][0]['position'], 'evidence':['synthetic']}]}
        before = self.snapshot()
        with self.assertRaises(ValueError):
            commit_refinement(self.db, candidate, proposal, 2, self.validation(), {})
        self.assertEqual(self.snapshot(), before)
        self.assert_history()

    def test_explicit_world_extension_preserves_history(self):
        root = Path(self.temp.name)
        entity = self.proposal()['additions'][0]['entity']
        entity.update(position=[8, 1, -10], provenance='AGENT_CREATED')
        save(root / 'edits.json', [{'op':'spawn', 'entity':entity}])
        args = {'db':str(root / 'world.sqlite'), 'revision':1,
                'edits':str(root / 'edits.json'), 'reason':'Synthetic prospective extension',
                'output':str(root / 'extension')}
        result = dispatch('extend', args)
        self.assertEqual(result['revision_after'], 2)
        self.assertEqual(scene_from_db(self.db)['entities'][:-1], self.before['entities'])
        self.assert_history()
        before = self.snapshot()
        args['revision'] = 2
        save(root / 'edits.json', [{'op':'move', 'id':'wall', 'position':[8, 1, -10]}])
        with self.assertRaisesRegex(ValueError, 'only adds'):
            dispatch('extend', args)
        self.assertEqual(self.snapshot(), before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
