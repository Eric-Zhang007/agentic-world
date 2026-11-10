from copy import deepcopy
import unittest
from scripts.propose_image_primitive import propose
from whitebox.image_alignment import project_points


class ImagePrimitiveTests(unittest.TestCase):
    def test_addition_preserves_input_and_existing_entities(self):
        from scripts.test_scene_refinement import SceneRefinementTests
        fixture=SceneRefinementTests()
        fixture.setUp()
        try:scene=deepcopy(fixture.before)
        finally:fixture.tearDown()
        scene['camera']=dict(position=[0,5,15],target=[5,1,5],fov=60)
        original=deepcopy(scene)
        pixels,valid=project_points([[8,0,8],[8,2,8]],scene['camera'],800,600)
        self.assertTrue(valid.all())
        annotation=dict(id='new_visible_box',category='prop',width=800,height=600,
            base_pixel=pixels[0].tolist(),support_height=0,size_prior=[.2,1,.2],
            evidence=['synthetic image'],uncertainty=['known test depth prior'],
            landmarks=[dict(local_fraction=[0,-.5,0],pixel=pixels[0].tolist()),
                       dict(local_fraction=[0,.5,0],pixel=pixels[1].tolist())],
            parameters=[dict(field='size',axis=1,bounds=[.1,3])])
        candidate,proposal,report=propose(scene,annotation)
        self.assertEqual(scene,original)
        self.assertEqual(candidate['entities'][:-1],scene['entities'])
        self.assertEqual(candidate['camera'],scene['camera'])
        self.assertAlmostEqual(candidate['entities'][-1]['size'][1],2,places=6)
        self.assertFalse(report['world_committed'])
        self.assertEqual(proposal['additions'][0]['entity']['provenance'],'INFERRED')
        with self.assertRaises(ValueError):propose(candidate,annotation)


if __name__=='__main__':unittest.main()
