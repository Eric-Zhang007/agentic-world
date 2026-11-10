import unittest
import numpy as np
from memory.gta_scene_pairs import mine_pairs,background_evidence


def point(i,x,agent='a',direction=(1,0,0),episode='e',clock='c'):
    return {'point_id':str(i)+agent,'episode_id':episode,'server_clock_domain':clock,
            'client_clock_domain':agent,'agent_id':agent,'time_ns':i*5_000_000_000,
            'camera_position':[x,0,2],'forward':direction}


class ScenePairs(unittest.TestCase):
    def test_stationary_is_not_revisit(self):
        self.assertEqual(mine_pairs([point(i,0) for i in range(20)]),[])

    def test_leave_return_is_required(self):
        pts=[point(i,x) for i,x in enumerate([0,5,10,15,20,15,10,5,0])]
        result=mine_pairs(pts)
        self.assertTrue(result)
        self.assertEqual(result[0]['kind'],'same_camera_leave_and_return')
        self.assertGreaterEqual(result[0]['max_excursion'],15)

    def test_missing_track_cannot_prove_excursion(self):
        self.assertEqual(mine_pairs([point(0,0),point(4,20),point(8,0)]),[])

    def test_cross_camera_and_chronology(self):
        result=mine_pairs([point(0,0),point(8,1,'b')])
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['reference_point'],'0a')

    def test_different_world_or_opposite_view_rejected(self):
        for end in [point(8,0,'b',episode='other'),point(8,0,'b',clock='other'),point(8,0,'b',direction=(-1,0,0))]:
            self.assertEqual(mine_pairs([point(0,0),end]),[])

    def test_background_excludes_masked_foreground(self):
        rng=np.random.default_rng(1)
        image=rng.integers(0,256,(360,640,3),dtype=np.uint8)
        full=np.full((360,640),255,np.uint8)
        result=background_evidence(image,image,full,full)
        self.assertGreater(result['inliers'],20)
        self.assertEqual(background_evidence(image,image,full*0,full*0)['inliers'],0)


if __name__=='__main__':unittest.main()
