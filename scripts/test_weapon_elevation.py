import unittest
import numpy as np
from scripts.test_whitebox_world import fixture
from whitebox.motion import rig, point_segment_distance
from whitebox.world import validate_scene


class ElevationTests(unittest.TestCase):
    def test_tilt_preserves_length_and_grip(self):
        scene=fixture()
        actor=next(e for e in scene['entities'] if e['kind']=='actor')
        staff=next(e for e in scene['entities'] if e['kind']=='staff')
        actor['weapon_elevation_deg']=-30
        pose,weapon,errors=rig(actor,0,0,held=staff,terrain=scene['terrain'])
        delta=np.subtract(weapon['b'],weapon['a'])
        self.assertAlmostEqual(np.linalg.norm(delta),staff.get('length',staff['size'][1]))
        self.assertAlmostEqual(delta[1]/np.linalg.norm(delta),-.5)
        for side in ['left','right']:
            self.assertLess(point_segment_distance(pose[side+'_hand'],weapon['a'],weapon['b']),1e-8)

    def test_default_is_unchanged_and_invalid_rejected(self):
        scene=fixture();actor=next(e for e in scene['entities'] if e['kind']=='actor')
        staff=next(e for e in scene['entities'] if e['kind']=='staff')
        before=rig(actor,0,0,held=staff)
        actor['weapon_elevation_deg']=0
        self.assertEqual(before,rig(actor,0,0,held=staff))
        for invalid in [float('nan'),91,True,'30']:
            actor['weapon_elevation_deg']=invalid
            with self.assertRaises(ValueError):validate_scene(scene)


if __name__=='__main__':unittest.main()
