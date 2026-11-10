import unittest
import numpy as np
from whitebox.mask_alignment import projected_bounds,compare_mask


class MaskAlignmentTests(unittest.TestCase):
    def test_actor_root_and_box_center_differ(self):
        camera=dict(position=[0,0,0],target=[0,0,-1],fov=90)
        entity=dict(id='a',kind='actor',size=[2,2,2],position=[0,0,-5])
        a=projected_bounds(entity,camera,800,600)
        entity['kind']='box'
        b=projected_bounds(entity,camera,800,600)
        self.assertEqual(a[3],300)
        self.assertGreater(b[3],300)

    def test_cropping_is_explicit_and_empty_unmeasured(self):
        camera=dict(position=[0,0,0],target=[0,0,-1],fov=90)
        entity=dict(id='a',kind='box',size=[2,2,2],position=[0,0,-5])
        mask=np.zeros((600,800),bool);mask[450:,350:450]=True
        report=compare_mask(entity,camera,mask)
        self.assertTrue(report['cropped_edges']['bottom'])
        self.assertEqual(report['bbox_iou'],0)
        self.assertEqual(compare_mask(entity,camera,np.zeros_like(mask))['status'],'EMPTY_MASK')
        entity['position']=[0,0,1]
        self.assertEqual(compare_mask(entity,camera,mask)['status'],'UNAVAILABLE_PROJECTION')


if __name__=='__main__':unittest.main()
