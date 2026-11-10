import unittest
from scripts.index_gta_reference_candidates import candidates


def rows():
    return [{'proxy_frame_index':i,'rgb_decode_index':i,'status':'estimated_alignment',
             'camera_interpolation':{'lower_client_sample_seq':i,'upper_client_sample_seq':i+1,'alpha':.5},
             'server_time_estimate_ns':i*41666667,'server_clock_domain':'s','client_clock_domain':'c',
             'appearance_id':'appearance_a','entity_ids':['1','2']} for i in range(360)]


class CandidateTests(unittest.TestCase):
    def test_historical_only(self):
        result=candidates(rows())
        self.assertEqual(len(result),2)
        self.assertTrue(all(r['observed_at_ns']<r['used_at_ns'] for r in result))

    def test_missing_camera_rejects_whole_target(self):
        data=rows();data[150]['status']='missing_camera'
        self.assertEqual([r['target_start_frame'] for r in candidates(data)],[240])

    def test_appearance_boundary_not_crossed(self):
        data=rows()
        for r in data[180:]:r['appearance_id']='appearance_b'
        self.assertEqual(candidates(data),[])

    def test_common_entity_required_at_every_target_frame(self):
        data=rows();data[150]['entity_ids']=['2']
        self.assertEqual(candidates(data)[0]['candidate_entity_ids'],['2'])

    def test_clock_and_decode_discontinuity(self):
        for key,value in [('server_clock_domain','new'),('rgb_decode_index',999)]:
            data=rows();data[150][key]=value
            self.assertEqual([r['target_start_frame'] for r in candidates(data)],[240])


if __name__=='__main__':unittest.main()
