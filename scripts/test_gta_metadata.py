import unittest
from unittest.mock import Mock
from scripts.enrich_gta_training_metadata import extract


class MetadataTests(unittest.TestCase):
    def test_discrete_brackets_and_missing_fields_are_preserved(self):
        records = Mock()
        records.camera.side_effect = [dict(episode_id='e', server_clock_domain='s', action={'jump':False}),
                                      dict(episode_id='e', server_clock_domain='s', action={'jump':True})]
        records.world.return_value = dict(episode_id='e', relation_edges=[])
        frame = dict(client_clock_domain='c', server_clock_domain='s', world_capture_tick=5,
                     proxy_frame_index=10, server_time_estimate_ns=100,
                     camera_interpolation=dict(lower_client_sample_seq=1, upper_client_sample_seq=2, alpha=.5))
        result = extract(records, 'a', frame)
        self.assertEqual([c['action']['jump'] for c in result['client_brackets']], [False,True])
        self.assertNotIn('entity_lifecycle_events', result['world_record'])
        self.assertEqual(result['world_record']['relation_edges'], [])

    def test_wrong_episode_is_rejected(self):
        records = Mock()
        records.camera.return_value = dict(episode_id='wrong', server_clock_domain='s')
        records.world.return_value = dict(episode_id='e')
        with self.assertRaises(ValueError):
            extract(records, 'a', dict(client_clock_domain='c', server_clock_domain='s',
                world_capture_tick=1, camera_interpolation=dict(lower_client_sample_seq=1, upper_client_sample_seq=2)))


if __name__ == '__main__':
    unittest.main()
