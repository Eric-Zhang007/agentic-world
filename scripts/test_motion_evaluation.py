import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_motion_ablation import project, assign


class MotionEvaluationTests(unittest.TestCase):
    def test_godot_forward_and_right(self):
        camera={"position":[0,0,5],"target":[0,0,0],"fov":50}
        pixels=project([[0,0,0],[1,0,0],[0,1,0]],camera,1024,768)
        np.testing.assert_allclose(pixels[0],[512,384])
        self.assertGreater(pixels[1,0],512)
        self.assertLess(pixels[2,1],384)
        self.assertIsNone(project([[0,0,6]],camera,1024,768))

    def test_duplicate_detection_cannot_fill_two_world_roles(self):
        expected=[{"world_id":"player","bbox":[10,10,50,90]},
                  {"world_id":"tiger_guard","bbox":[60,10,100,90]}]
        def observation(i):
            return SimpleNamespace(category="warrior",confidence=.9,bbox_xyxy=[10,10,50,90],
                                   obs_id=str(i),track_id=str(i))
        pairs,missing=assign(expected,[observation(0),observation(1)],200,100)
        self.assertEqual(len(pairs),1)
        self.assertEqual(len(missing),1)

    def test_missing_is_not_zero_error(self):
        pairs,missing=assign([{"world_id":"player","bbox":[10,10,50,90]}],[],200,100)
        self.assertEqual(pairs,[])
        self.assertEqual(missing,["player"])


if __name__=="__main__":unittest.main()
