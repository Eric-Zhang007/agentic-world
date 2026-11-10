import unittest
from scripts.evaluate_camera_revisit import validated_review,validate_camera_excursion


class CameraRevisitTests(unittest.TestCase):
    def test_missing_or_tiny_planned_orbit_cannot_satisfy_requested_test(self):
        scene={'camera':{'position':[0,1,3],'target':[0,1,0]}}
        for position in ([0,1,3],[.01,1,3]):
            with self.assertRaises(ValueError):validate_camera_excursion(scene,{'frames':[
                {'camera':{'position':position,'target':[0,1,0]}}]},90)
        result=validate_camera_excursion(scene,{'frames':[
            {'camera':{'position':[3,1,0],'target':[0,1,0]}},{'camera':scene['camera']}]},90)
        self.assertAlmostEqual(result['planned_excursion_degrees'],90)

    def review(self):
        return dict(overall='PASS',**{key:{'status':'SUPPORTED','evidence':['fixture']}
            for key in ['camera_execution','return_to_start','identity_consistency',
                        'static_world_consistency','occlusion_and_reappearance']})

    def test_unchanged_world_does_not_pass_unexecuted_camera(self):
        review=self.review();review['camera_execution']['status']='CONTRADICTED'
        with self.assertRaises(ValueError):validated_review(review,True)

    def test_return_requires_supported_revisit(self):
        review=self.review();review['return_to_start']['status']='UNCERTAIN'
        with self.assertRaises(ValueError):validated_review(review,True)
        self.assertEqual(validated_review(review,False),review)

    def test_explicit_failure_and_uncertainty_remain_diagnostics(self):
        review=self.review();review['overall']='INCONCLUSIVE';review['camera_execution']['status']='UNCERTAIN'
        self.assertEqual(validated_review(review,True),review)

    def test_missing_evidence_cannot_pass(self):
        review=self.review();review['identity_consistency']['evidence']=[]
        with self.assertRaises(ValueError):validated_review(review,True)


if __name__=='__main__':unittest.main()
