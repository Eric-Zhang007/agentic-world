import unittest
from scripts.survey_gta_passage_pairs import candidates


def point(i,x,agent='a',episode='e',clock='c'):
    return {'point_id':f'{agent}{i}','time_ns':i*5_000_000_000,'episode_id':episode,
            'server_clock_domain':clock,'client_clock_domain':agent,'agent_id':agent,
            'camera_position':[x,0,1],'forward':[1,0,0]}


class PassageTests(unittest.TestCase):
    def test_excluded_targets_still_supply_history(self):
        result=candidates([point(0,0),point(1,0),point(2,0)],excluded_targets={'a0','a1'})
        self.assertEqual([r['target_point'] for r in result],['a2'])
        self.assertTrue(all(o['reference_point'] in {'a0','a1'} for o in result[0]['options']))

    def test_recent_reference_precedes_target(self):
        points=[point(i,x) for i,x in enumerate([0,9,16,9,0])]
        result=candidates(points,20,5,90)
        item=next(r for r in result if r['target_point']=='a4')
        self.assertTrue(any(o['kind']=='same_camera_return' for o in item['options']))
        self.assertTrue(all(o['age_sec']>=5 for r in result for o in r['options']))

    def test_stationary_is_context_not_return(self):
        result=candidates([point(i,0) for i in range(6)])
        self.assertTrue(result)
        self.assertEqual({o['kind'] for r in result for o in r['options']},{'nearby_history'})

    def test_world_isolation(self):
        for p in (point(6,0,'b','other'),point(6,0,'b',clock='other')):
            self.assertEqual(candidates([point(0,0),p]),[])

    def test_multiple_references_and_nearest_time(self):
        result=candidates([point(0,0),point(2,0),point(4,0),point(6,0,'b')])
        options=next(r['options'] for r in result if r['target_point']=='b6')
        self.assertEqual(options[0]['reference_point'],'a4')
        self.assertEqual(len(options),3)


if __name__=='__main__':unittest.main()
