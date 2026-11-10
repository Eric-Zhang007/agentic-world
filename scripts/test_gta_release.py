import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from memory.gta_release import Records, entity_map, interpolate_object
from memory.gta_training_dataset import safe_path
from scripts.build_gta_release_index import pack
from scripts.review_gta_release import parse_json
from scripts.export_gta_release import box_overlap
import pyarrow as pa
import pyarrow.parquet as pq


class ReleaseTests(unittest.TestCase):
    def test_duplicate_keys_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'index.db'
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE camera(agent,domain,seq,data)')
                for _ in range(2):db.execute('INSERT INTO camera VALUES (?,?,?,?)',('a','c',1,pack({'ok':True})))
            db.close()
            records=Records(path)
            try:
                with self.assertRaisesRegex(ValueError,'AMBIGUOUS'):records.camera('a','c',1)
            finally:records.close()

    def test_duplicate_entity_not_silently_selected(self):
        self.assertEqual(entity_map({'scene_observation':{'entities':[{'dataset_entity_id':7},{'dataset_entity_id':7}]}}),{})

    def test_occupancy_transition(self):
        a={'model_hash':-1,'position':[0,0,0],'orientation_quaternion':[0,0,0,1],'current_vehicle_network_id':None}
        b={**a,'model_hash':4294967295,'current_vehicle_network_id':8}
        with self.assertRaisesRegex(ValueError,'OCCUPANCY'):interpolate_object(a,b,.5)
        b['current_vehicle_network_id']=None;b['position']=[2,0,0]
        np.testing.assert_allclose(interpolate_object(a,b,.5)['position'],[1,0,0])

    def test_safe_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.assertEqual(safe_path(root,'a.png'),root/'a.png')
            with self.assertRaises(ValueError):safe_path(root,'../escape.png')

    def test_json_fences(self):
        self.assertEqual(parse_json('```json\n{"ok":true}\n```'),{'ok':True})
        with self.assertRaises(json.JSONDecodeError):parse_json('Looks good')

    def test_source_location_resolves_actual_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source.parquet'
            pq.write_table(pa.table({'payload_json':['{"id":3}','{"id":7}']}),source)
            path=Path(tmp)/'index.db'
            db=sqlite3.connect(path)
            db.execute('CREATE TABLE camera(agent,domain,seq,data)')
            db.execute('INSERT INTO camera VALUES (?,?,?,?)',('a','c',2,pack({'_parquet':str(source),'group':0,'row':1})))
            db.commit();db.close()
            records=Records(path)
            try:self.assertEqual(records.camera('a','c',2),{'id':7})
            finally:records.close()

    def test_disjoint_rgb_and_world_boxes(self):
        self.assertEqual(box_overlap([0,0,10,10],[20,20,30,30]),0)
        self.assertEqual(box_overlap([0,0,10,10],[0,0,10,10]),1)


if __name__=='__main__':unittest.main()
