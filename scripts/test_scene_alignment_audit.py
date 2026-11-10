from copy import deepcopy
import unittest

from scripts.audit_scene_alignment import validate_audit


class AlignmentAuditTests(unittest.TestCase):
    def setUp(self):
        self.scene={'entities':[{'id':'arch_a','kind':'box'},{'id':'steps_b','kind':'stairs'},
                                {'id':'actor_c','kind':'actor'}]}
        self.review={'usable_for_camera_alignment':True,'anchors':[
            {'world_id':'arch_a','status':'SUPPORTED','evidence':['Visible coarse silhouette']},
            {'world_id':'steps_b','status':'UNOBSERVED','evidence':['Outside current view']} ]}

    def test_unseen_geometry_can_remain_unobserved(self):
        self.assertIs(validate_audit(self.review,self.scene),self.review)

    def test_omission_duplicate_or_invented_id_cannot_pass(self):
        for mode in ('omit','duplicate','invent'):
            review=deepcopy(self.review)
            if mode=='omit':review['anchors'].pop()
            else:review['anchors'][1]['world_id']='arch_a' if mode=='duplicate' else 'imagined'
            with self.assertRaises(ValueError):validate_audit(review,self.scene)

    def test_empty_or_nontext_evidence_cannot_pass(self):
        for evidence in ([],[''],[False],'looks fine'):
            review=deepcopy(self.review);review['anchors'][0]['evidence']=evidence
            with self.assertRaises(ValueError):validate_audit(review,self.scene)


if __name__=='__main__':unittest.main()
