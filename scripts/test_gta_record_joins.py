import json
import sqlite3
import unittest

from scripts.audit_gta_record_joins import audit_frame, compact_camera, model_id


class RecordJoinsTest(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.addCleanup(self.db.close)
        self.db.execute('CREATE TABLE camera(episode,agent,domain,seq,data)')
        self.db.execute('CREATE TABLE world(episode,domain,seq,time_ns)')
        self.record = {'episode_id': 'e', 'agent_id': 'a'}
        self.frame = {'client_clock_domain': 'c', 'server_clock_domain': 's',
                      'world_capture_tick': 5, 'camera_interpolation': {
                          'lower_client_sample_seq': 1, 'upper_client_sample_seq': 2, 'alpha': .5}}
        self.camera = {'server_clock_domain': 's', 'rendered_camera_position': [0, 0, 0],
                       'entities': {'7': 12}, 'projection_rays': {'rays': [
                           {'uv': [.5, .5], 'direction': [0, 0, 1]},
                           {'uv': [.75, .5], 'direction': [.25, 0, 1]},
                           {'uv': [.5, .75], 'direction': [0, .25, 1]}]}}
        for seq in (1, 2):
            self.db.execute('INSERT INTO camera VALUES (?,?,?,?,?)', ('e', 'a', 'c', seq, json.dumps(self.camera)))
        self.db.execute('INSERT INTO world VALUES (?,?,?,?)', ('e', 's', 5, 100))

    def audit(self):
        return audit_frame(self.db, self.record, self.frame, {})

    def test_valid_join(self):
        self.assertEqual(self.audit(), (None, {'7': 12}))

    def test_missing_world_domain(self):
        self.db.execute("UPDATE world SET domain='other'")
        self.assertEqual(self.audit()[0], 'missing_or_ambiguous_world')

    def test_duplicate_camera_is_rejected(self):
        self.db.execute('INSERT INTO camera SELECT * FROM camera WHERE seq=1')
        self.assertEqual(self.audit()[0], 'missing_or_ambiguous_camera')

    def test_model_change_is_not_same_instance_evidence(self):
        self.camera['entities']['7'] = 13
        self.db.execute('UPDATE camera SET data=? WHERE seq=2', (json.dumps(self.camera),))
        self.assertEqual(self.audit(), (None, {}))

    def test_degenerate_rays(self):
        for ray in self.camera['projection_rays']['rays']:
            ray['direction'] = [0, 0, 1]
        self.db.execute('UPDATE camera SET data=?', (json.dumps(self.camera),))
        self.assertEqual(self.audit()[0], 'invalid_interpolated_camera')

    def test_duplicate_entity_and_signed_model(self):
        self.assertEqual(model_id(-1), model_id(4294967295))
        result = compact_camera({'scene_observation': {'entities': [
            {'dataset_entity_id': 7, 'model_hash': 12},
            {'dataset_entity_id': 7, 'model_hash': 12}]}})
        self.assertEqual(result['entities'], {})


if __name__ == '__main__':
    unittest.main()
