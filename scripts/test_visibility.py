import unittest
import numpy as np
from whitebox.visibility import box_intersections,inspect_visibility


class VisibilityTests(unittest.TestCase):
    def test_parallel_inside_outside_and_rotated(self):
        box=dict(id='box',kind='box',position=[0,0,-5],size=[2,2,2])
        rays=np.array([[0,0,-1],[1,0,0]],float)
        np.testing.assert_allclose(box_intersections(np.zeros(3),rays,box),[4,np.inf])
        self.assertTrue(np.isinf(box_intersections(np.array([3,0,0]),rays[:1],box)[0]))
        box['rotation_y']=np.pi/4
        self.assertAlmostEqual(box_intersections(np.zeros(3),rays[:1],box)[0],5-np.sqrt(2))

    def test_occluder_identity_and_no_overlap(self):
        scene=dict(camera=dict(position=[0,1,0],target=[0,1,-1],fov=90),entities=[
            dict(id='actor',kind='actor',position=[0,0,-5],size=[1,2,1]),
            dict(id='wall',kind='box',position=[0,1,-3],size=[3,3,.2])])
        mask=np.zeros((100,100),bool);mask[45:55,45:55]=True
        result=inspect_visibility(scene,'actor',mask,1)
        self.assertEqual(result['blocked_fraction'],1)
        self.assertEqual(result['blockers'],{'wall':100})
        scene['entities'][1]['position'][2]=-8
        self.assertEqual(inspect_visibility(scene,'actor',mask,1)['blocked_fraction'],0)
        mask[:]=False;mask[:3,:3]=True
        self.assertEqual(inspect_visibility(scene,'actor',mask,1)['status'],'NO_TARGET_OVERLAP')


if __name__=='__main__':unittest.main()
