import tempfile
from pathlib import Path
import unittest
import orjson
from scripts.index_gta_spool import build
from memory.gta_release import Records
from memory.gta_scene_memory import inspect_frame,index_segment
import sqlite3


class SpoolTest(unittest.TestCase):
    def test_real_pointer_reader_and_clock_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'.spool/agents/agent_000';folder.mkdir(parents=True)
            camera={'agent_id':'agent_000','episode_id':'episode','client_clock_domain':'client',
                    'client_sample_seq':1,'server_time_estimate_ns':100,'camera':{'position':[1,2,3]},
                    'scene_observation':{'entities':[{'dataset_entity_id':7}]}}
            world={'episode_id':'episode','server_clock_domain':'server','sample_seq':2,
                   'server_monotonic_ns':100,'objects':[{'dataset_entity_id':7}]}
            (folder/'client.jsonl').write_bytes(orjson.dumps(camera)+b'\n')
            (root/'.spool/world_state.jsonl').write_bytes(orjson.dumps(world)+b'\n')
            dest=root/'index.sqlite3';report=build(root,dest);records=Records(dest)
            self.assertEqual(records.camera('agent_000','client',1),{**camera,**camera['camera']})
            self.assertEqual(records.world('server',2),world)
            with self.assertRaises(ValueError):records.camera('agent_000','other',1)
            with self.assertRaises(ValueError):records.entity('7')
            records.close();self.assertEqual(report['distinct_observed_entity_ids'],1)
            self.assertEqual(build(root,dest),report)

    def test_scene_candidates_are_not_visibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'manifest.json').write_bytes(orjson.dumps({'episode_id':'e','agent_id':'a','segment':'s'}))
            (root/'frames.jsonl').write_bytes(orjson.dumps({'proxy_frame_index':0,'scene_index':'scene.json','static_candidates':1})+b'\n')
            (root/'scene.json').write_bytes(orjson.dumps({'static_placement_ids':[{'id':'7:2','kind':'object'}]}))
            result=inspect_frame(root/'manifest.json',0)
            self.assertEqual(result['placements'][0]['visibility'],'unknown')
            self.assertEqual(result['placements'][0]['memory_key'],'e:placement:7:2')
            with self.assertRaises(KeyError):inspect_frame(root/'manifest.json',4)
            index=root/'scene.sqlite3';report=index_segment(root/'manifest.json',index)
            self.assertEqual(report['unique_candidate_placements'],1)
            db=sqlite3.connect(index)
            try:
                self.assertEqual(db.execute('SELECT f.frame FROM frame_scene f JOIN scene_placement p ON f.scene=p.scene WHERE p.placement_id=?',('7:2',)).fetchall(),[(0,)])
            finally:db.close()


if __name__=='__main__':unittest.main()
