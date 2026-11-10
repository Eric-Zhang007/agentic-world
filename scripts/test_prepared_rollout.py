import json
from pathlib import Path
import tempfile
import unittest
from scripts.test_whitebox_world import fixture,action
from scripts.run_evidence_rollouts import load_prepared_plan
from whitebox.world import initialize,scene_from_db
from whitebox.motion import prepare_scene,simulate
from state.db import WorldDB


class PreparedPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.db=WorldDB(self.root/'world.sqlite');self.scene=prepare_scene(fixture())
        initialize(self.db,self.scene);self.plan=self.root/'source';self.plan.mkdir()
        self.action=action();timeline,_,_=simulate(self.scene,self.action)
        for name,data in [('scene',self.scene),('action',self.action),('timeline',timeline)]:
            (self.plan/(name+'.json')).write_text(json.dumps(data))

    def tearDown(self):
        self.db.connection.close();self.temp.cleanup()

    def test_reuse_preserves_world_and_original_plan(self):
        history=self.db.get_meta('whitebox_history')
        result=load_prepared_plan(self.db,self.plan,self.root/'copy')
        self.assertEqual(result,self.action);self.assertEqual(scene_from_db(self.db),self.scene)
        self.assertEqual(self.db.get_meta('whitebox_history'),history)
        self.assertEqual((self.plan/'timeline.json').read_bytes(),(self.root/'copy/timeline.json').read_bytes())

    def test_changed_clock_cannot_replay(self):
        self.db.set_meta('tick',1)
        with self.assertRaises(ValueError):load_prepared_plan(self.db,self.plan,self.root/'copy')
        self.assertFalse((self.root/'copy').exists())

    def test_tampered_trajectory_cannot_replay(self):
        path=self.plan/'timeline.json';data=json.loads(path.read_text())
        data['frames'][-1]['entities'][0]['position'][0]+=1
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):load_prepared_plan(self.db,self.plan,self.root/'copy')


if __name__=='__main__':unittest.main()
