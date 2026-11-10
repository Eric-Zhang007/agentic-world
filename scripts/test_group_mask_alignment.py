import copy
import unittest
import numpy as np
from whitebox.mask_alignment import compare_group_mask, compare_mask, projected_bounds


class GroupMaskAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.camera=dict(position=[0,0,0],target=[0,0,-1],fov=90)
        self.entities=[dict(id=str(i),kind='box',size=[2,2,2],position=[x,0,-5])
                       for i,x in enumerate((-2,2))]
        self.mask=np.zeros((100,100),bool)
        self.mask[37:63,12:88]=True

    def test_union_correspondence(self):
        report=compare_group_mask(self.entities,self.camera,self.mask)
        boxes=np.array([projected_bounds(e,self.camera,100,100) for e in self.entities])
        np.testing.assert_allclose(report['projected_bbox'],
                                   np.r_[boxes[:,:2].min(0),boxes[:,2:].max(0)])
        self.assertEqual(report['member_ids'],['0','1'])
        for entity in self.entities:
            self.assertGreater(report['bbox_iou'],compare_mask(entity,self.camera,self.mask)['bbox_iou'])

    def test_single_member_equivalence_and_clipping(self):
        for kind in ('box','actor','stairs'):
            entity=copy.deepcopy(self.entities[0]);entity['kind']=kind;entity['position'][0]=-4
            single=compare_mask(entity,self.camera,self.mask)
            group=compare_group_mask([entity],self.camera,self.mask)
            for key in single.keys()-{'world_id'}:
                self.assertEqual(single[key],group[key])

    def test_invalid_projection_is_not_dropped(self):
        self.entities[1]['position'][2]=-.5
        report=compare_group_mask(self.entities,self.camera,self.mask)
        self.assertEqual(report['status'],'UNAVAILABLE_PROJECTION')
        self.assertNotIn('bbox_iou',report)
        self.assertNotIn('projected_bbox',report)
        self.assertEqual(report['member_ids'],['0','1'])

    def test_invalid_groups(self):
        for entities in ([],[self.entities[0]]*2,[dict(self.entities[0],id='')],
                         [dict(self.entities[0],id=None)]):
            with self.assertRaises(ValueError):
                compare_group_mask(entities,self.camera,self.mask)

    def test_empty_mask_and_preservation(self):
        entities=copy.deepcopy(self.entities);camera=copy.deepcopy(self.camera);mask=self.mask.copy()
        compare_group_mask(self.entities,self.camera,self.mask)
        self.assertEqual(entities,self.entities);self.assertEqual(camera,self.camera)
        np.testing.assert_array_equal(mask,self.mask)
        report=compare_group_mask(self.entities,self.camera,np.zeros_like(self.mask))
        self.assertEqual(report,dict(status='EMPTY_MASK',member_ids=['0','1']))


if __name__=='__main__':unittest.main()
