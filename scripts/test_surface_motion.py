import tempfile
import unittest
from pathlib import Path
import numpy as np
from whitebox.surfaces import SurfaceWorld
from whitebox.motion import simulate,prepare_scene
from scripts.test_whitebox_world import fixture


def floor(y=0,x0=-10,x1=10,z0=-10,z1=10):
    return [[[x0,y,z0],[x1,y,z0],[x1,y,z1]],[[x0,y,z0],[x1,y,z1],[x0,y,z1]]]


class SurfaceTests(unittest.TestCase):
    def test_bridge_keeps_current_layer(self):
        w=SurfaceWorld(floor()+floor(4))
        self.assertEqual(w.floor(0,0,0),0)
        self.assertEqual(w.floor(0,0,4),4)
        self.assertIsNone(w.floor(12,0,0))

    def test_step_and_cliff(self):
        w=SurfaceWorld(floor(.2))
        self.assertAlmostEqual(w.floor(0,0,0),.2)
        self.assertIsNone(SurfaceWorld(floor(1)).floor(0,0,0))
        self.assertIsNone(SurfaceWorld(floor(-2)).floor(0,0,0))

    def test_triangle_not_its_bounding_box(self):
        w=SurfaceWorld([[[0,0,0],[0,3,0],[0,0,3]]])
        self.assertTrue(w.body_blocked([0,0,.5],[.4,1.8,.4]))
        self.assertFalse(w.body_blocked([0,2,2.5],[.4,.5,.4]))

    def test_motion_stops_at_thin_wall(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'collision.bin'
            triangles=floor()+[[[.6,0,-4],[.6,3,-4],[.6,3,4]],[[.6,0,-4],[.6,3,4],[.6,0,4]]]
            np.asarray(triangles,dtype='<f4').tofile(path)
            scene=fixture();scene['entities']=[e for e in scene['entities'] if e['id']=='player']
            scene['entities'][0]['position']=[0,0,0]
            scene['static_geometry']={'collision_path':str(path)}
            scene=prepare_scene(scene)
            action={'actor_id':'player','phases':[{'type':'move','delta':[2,0,0],
                'duration_sec':3,'end_velocity':[.5,0,0],'criterion':'Walk forward'}]}
            timeline,end,report=simulate(scene,action)
            self.assertLess(end['entities'][0]['position'][0],.6)
            self.assertTrue(any(v['type']=='blocked_player' for v in report['violations']))
            self.assertEqual(end['entities'][0]['motion']['velocity'],[0,0,0])

    def test_motion_climbs_step_without_teleporting_layers(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'step.bin'
            triangles=floor(x1=.8)+floor(.2,x0=.8)+floor(4)
            np.asarray(triangles,dtype='<f4').tofile(path)
            scene=fixture();actor=next(e for e in scene['entities'] if e['id']=='player')
            actor['asset_id']='gta_human';scene['entities']=[actor]
            scene['static_geometry']={'collision_path':str(path)}
            action={'actor_id':'player','phases':[{'type':'move','delta':[2,0,0],
                'duration_sec':3,'end_velocity':[.5,0,0],'criterion':'Cross a low step'}]}
            _,end,report=simulate(prepare_scene(scene),action)
            self.assertAlmostEqual(end['entities'][0]['position'][1],.2,places=6)
            self.assertAlmostEqual(end['entities'][0]['position'][0],2)


if __name__=='__main__':unittest.main()
