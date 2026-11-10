from copy import deepcopy
import unittest
import numpy as np
from whitebox.camera_fit import fit_camera
from whitebox.image_alignment import landmark_points,project_points


class CameraFitTests(unittest.TestCase):
    def test_fixed_geometry_camera_recovery(self):
        scene = dict(camera=dict(position=[0,3,9],target=[0,1,-3],fov=55),entities=[])
        rows=[]
        for i,p in enumerate([[-2,0,-1],[3,0,-4],[1,2,-3],[-1,3,1],[2,1,0],[0,4,-6]]):
            scene['entities'].append(dict(id=str(i),position=p,size=[1,1,1]))
            rows.append(dict(id=str(i),local_fraction=[0,0,0]))
        original=deepcopy(scene)
        target=dict(position=[.2,3.1,9.3],target=[.1,1.1,-3],fov=58)
        pixels,_=project_points(landmark_points(scene,rows),target,1280,720)
        for row,pixel in zip(rows,pixels):row['pixel']=pixel.tolist()
        result,report=fit_camera(scene,dict(width=1280,height=720,landmarks=rows,
            bounds=[[-1,1]]*3+[[-10,10]]*3,heldout=[rows[-1]]))
        self.assertEqual(scene,original)
        self.assertEqual(result['entities'],scene['entities'])
        np.testing.assert_allclose(result['camera']['position'],target['position'],atol=1e-5)
        self.assertLess(max(report['after_errors_px']),1e-5)
        self.assertFalse(report['world_commit'])
        locked=dict(width=1280,height=720,landmarks=rows,
                    bounds=[[-1,1]]*3+[[-10,10]]*2+[[3,3]])
        result,report=fit_camera(scene,locked)
        self.assertAlmostEqual(result['camera']['fov'],58)
        self.assertEqual(report['free_parameter_indices'],[0,1,2,3,4])
        self.assertLess(max(report['after_errors_px']),1e-5)


if __name__=='__main__':unittest.main()
