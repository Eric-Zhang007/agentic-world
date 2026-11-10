import unittest
from collections import Counter
from memory.gta_mixed_dataset import balanced_round


class Sampling(unittest.TestCase):
    def rows(self):
        return [{'sample_id':f'{task}{i}','dataset':task,'task':task,'balance_group':str(i%3)}
                for task,n in [('scene',10),('person',100),('vehicle',40)] for i in range(n)]

    def test_ratios_and_no_replacement(self):
        result=balanced_round(self.rows(),{'scene':.5,'person':.25,'vehicle':.25})
        self.assertEqual(Counter(r['task'] for r in result),{'scene':10,'person':5,'vehicle':5})
        self.assertEqual(len(result),len({r['sample_id'] for r in result}))

    def test_repeatability_and_seed(self):
        weights={'scene':.5,'person':.25,'vehicle':.25}
        self.assertEqual(balanced_round(self.rows(),weights,3),balanced_round(self.rows(),weights,3))
        self.assertNotEqual(balanced_round(self.rows(),weights,3),balanced_round(self.rows(),weights,4))

    def test_missing_pool_and_invalid_weights(self):
        for weights in ({'missing':1},{'scene':.4},{'scene':1.1,'person':-.1}):
            with self.assertRaises(ValueError):balanced_round(self.rows(),weights)

    def test_group_balance(self):
        result=balanced_round(self.rows(),{'scene':.5,'person':.25,'vehicle':.25})
        counts=Counter(r['balance_group'] for r in result if r['task']=='person')
        self.assertLessEqual(max(counts.values())-min(counts.values()),1)

    def test_background_site_cap(self):
        rows=[{'sample_id':str(i),'dataset':'scenes','task':'scene_background','balance_group':str(i%3)} for i in range(30)]
        result=balanced_round(rows,{'scene_background':1},max_scene_per_site=2)
        self.assertEqual(len(result),6)
        self.assertEqual(max(Counter(r['balance_group'] for r in result).values()),2)


if __name__=='__main__':unittest.main()
