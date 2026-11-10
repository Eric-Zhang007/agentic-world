import unittest
from scripts.summarize_gta_metadata import active, summarize


class SummaryTests(unittest.TestCase):
    def test_release_only_and_unknown_do_not_mean_active(self):
        self.assertFalse(active({'pressed':False,'held':False,'released':True}))
        self.assertFalse(active(None))
        self.assertTrue(active({'held':True}))

    def test_changes_missingness_and_empty_relations(self):
        def frame(state,world):
            return {'client_brackets':[{'agent_state':state,'action':{'jump':{'held':False}}}], 'world_record':world}
        result=summarize({'sample_id':'a','episode_id':'e','target_samples':[
            frame({'health':100},{'relation_edges':[]}),frame({'health':90},{})]})
        self.assertEqual(result['sampled_active_inputs'],[])
        self.assertTrue(result['sampled_state_changes']['health'])
        self.assertNotIn('alive',result['sampled_state_changes'])
        self.assertEqual(result['record_coverage']['relation_edges_recorded_samples'],1)
        self.assertEqual(result['record_coverage']['relation_edges_nonempty_samples'],0)


if __name__ == '__main__':
    unittest.main()
