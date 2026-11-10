import unittest
import numpy as np
from whitebox.visibility import inspect_group_visibility, inspect_visibility


class GroupVisibilityTests(unittest.TestCase):
    def test_members_are_not_external_blockers(self):
        scene=dict(camera=dict(position=[0,1,0],target=[0,1,-1],fov=90),entities=[
            dict(id='body',kind='box',position=[0,1,-5],size=[2,2,1]),
            dict(id='trim',kind='box',position=[0,1,-3],size=[2,2,.2]),
            dict(id='obstacle',kind='box',position=[0,1,-8],size=[3,3,.2])])
        mask=np.zeros((100,100),bool);mask[45:55,45:55]=True
        self.assertEqual(inspect_visibility(scene,'body',mask,1)['blockers'],{'trim':100})
        result=inspect_group_visibility(scene,['body','trim'],mask,1)
        self.assertEqual(result['blocked_fraction'],0)
        self.assertEqual(result['comparable_pixels'],100)
        scene['entities'][2]['position'][2]=-2
        result=inspect_group_visibility(scene,['body','trim'],mask,1)
        self.assertEqual(result['blockers'],{'obstacle':100})
        self.assertEqual(result['blocked_fraction'],1)
        for members in [[],['body','body']]:
            with self.assertRaises(ValueError):inspect_group_visibility(scene,members,mask)


if __name__=='__main__':unittest.main()
