import unittest
import tempfile
from pathlib import Path
from scripts.serve_controls import Session
from scripts.test_keyboard_controls import forward_scene
from harness.pi.world_tools import dispatch,save,check_plan


class BoundControlsTests(unittest.TestCase):
    def test_recording_is_bound_to_real_revision_and_preserves_world(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);scene=forward_scene();save(root/'scene.json',scene)
            db=str(root/'world.sqlite');dispatch('initialize',{'scene':str(root/'scene.json'),'db':db})
            before=dispatch('status',{'db':db})
            session=Session(None,'player',root/'recordings',db)
            for _ in range(72):session.step({'keys':['w']})
            result=session.export();check_plan(Path(result['plan']))
            self.assertEqual(before,dispatch('status',{'db':db}))
            self.assertTrue(result['world_bound'])
            session.reset();self.assertEqual(session.controller.actor['position'],scene['entities'][0]['position'])

    def test_changed_world_requires_refresh_before_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);save(root/'scene.json',forward_scene());db=str(root/'world.sqlite')
            dispatch('initialize',{'scene':str(root/'scene.json'),'db':db})
            session=Session(None,'player',root/'recordings',db)
            for _ in range(72):session.step({'keys':['w']})
            from state.db import WorldDB
            database=WorldDB(db);database.set_meta('whitebox_revision',1);database.connection.close()
            with self.assertRaisesRegex(ValueError,'World changed'):session.export()

    def test_unbound_preview_cannot_generate(self):
        with tempfile.TemporaryDirectory() as directory:
            session=Session(forward_scene(),'player',Path(directory))
            with self.assertRaisesRegex(ValueError,'--db'):session.rollout('prompt','Walk')


if __name__=='__main__':unittest.main()
