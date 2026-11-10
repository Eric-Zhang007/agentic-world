import unittest
import numpy as np
from whitebox.surfaces import SurfaceWorld
from scripts.test_whitebox_world import fixture
from whitebox.motion import prepare_scene,simulate
from whitebox.posture import body_size


def scene():
    s=fixture();s['entities']=[e for e in s['entities'] if e['id']=='player']
    s['entities'][0]['asset_id']='gta_human'
    s['camera']={'position':[4,2.5,5],'target':[1,.8,0],'fov':50}
    return s


def obstacle(identity,pos,size):
    return {'id':identity,'kind':'box','position':pos,'size':size,'color':[.55,.55,.55],
        'provenance':'AGENT_CREATED','reason':'Geometric action acceptance fixture'}


def crouch_action():
    return {'actor_id':'player','goal':'Duck under beam and stand on the other side', 'phases':[
        {'type':'posture','target_posture':'crouched','duration_sec':.5,'criterion':'Lower body below beam'},
        {'type':'move','gait':'crouch','delta':[1.8,0,0],'duration_sec':2.,'end_velocity':[0,0,0],'criterion':'Clear beam'},
        {'type':'posture','target_posture':'standing','duration_sec':.5,'criterion':'Stand beyond beam'}]}


def jump_action():
    return {'actor_id':'player','goal':'Jump over low barrier and land', 'phases':[
        {'type':'jump','delta':[2,0,0],'duration_sec':1.,'end_velocity':[0,0,0],'criterion':'Land beyond barrier'}]}


class SemanticTests(unittest.TestCase):
    def test_airborne_rounded_foot_clears_sloping_takeoff_ground(self):
        floor=[[[-2,-.1,-2],[2,.1,-2],[2,.1,2]],[[-2,-.1,-2],[2,.1,2],[-2,-.1,2]]]
        w=SurfaceWorld(floor)
        self.assertFalse(w.body_blocked([0,.01,0],[.65,1.8,.5],foot_clearance=.005))
        self.assertTrue(w.body_blocked([0,-.1,0],[.65,1.8,.5],foot_clearance=.005))

    def test_crouch_can_pass_where_standing_cannot(self):
        s=scene();s['entities'].append(obstacle('beam',[.9,1.55,0],[.6,.5,2]))
        timeline,end,report=simulate(prepare_scene(s),crouch_action())
        self.assertTrue(report['valid'],report)
        self.assertAlmostEqual(end['entities'][0]['position'][0],1.8)
        self.assertAlmostEqual(body_size(timeline['frames'][12]['entities'][0])[1],1.17)
        straight={'actor_id':'player','phases':[{'type':'move','gait':'walk','delta':[1.8,0,0],
            'duration_sec':2.,'end_velocity':[0,0,0],'criterion':'Walk under beam'}]}
        _,_,blocked=simulate(prepare_scene(s),straight)
        self.assertFalse(blocked['valid'])

    def test_cannot_stand_inside_beam(self):
        s=scene();s['entities'][0]['posture_weights']={'crouched':1.}
        s['entities'][0]['position']=[.9,0,0]
        s['entities'].append(obstacle('beam',[.9,1.55,0],[.6,.5,2]))
        _,_,report=simulate(prepare_scene(s),{'actor_id':'player','phases':[crouch_action()['phases'][-1]]})
        self.assertFalse(report['valid'])

    def test_jump_has_airborne_interval_and_real_landing(self):
        s=scene();s['entities'].append(obstacle('barrier',[1,.2,0],[.3,.4,2]))
        timeline,end,report=simulate(prepare_scene(s),jump_action())
        self.assertTrue(report['valid'],report)
        a=timeline['frames'][11]['entities'][0]
        self.assertAlmostEqual(a['position'][1],9.81/8)
        self.assertEqual(a['support_contacts'],[])
        self.assertEqual(end['entities'][0]['support_state'],'grounded')
        self.assertAlmostEqual(end['entities'][0]['position'][1],0.)
        self.assertEqual([e['type'] for e in report['events']],['planned_takeoff','planned_landing'])

    def test_jump_rejects_ceiling_and_high_wall(self):
        for obj in (obstacle('ceiling',[1,2.1,0],[4,.2,3]),obstacle('wall',[1,1.5,0],[.3,3,2])):
            s=scene();s['entities'].append(obj)
            with self.assertRaisesRegex(ValueError,'Jump trajectory intersects'):
                simulate(prepare_scene(s),jump_action())

    def test_crouch_requires_slow_motion(self):
        s=scene();s['entities'][0]['posture_weights']={'crouched':1.}
        action={'actor_id':'player','phases':[dict(crouch_action()['phases'][1],delta=[3,0,0],duration_sec=.5)]}
        with self.assertRaisesRegex(ValueError,'gait speed'):simulate(prepare_scene(s),action)

    def test_jump_can_continue_into_running_without_stopping(self):
        s=scene();s['entities'][0]['motion']={'velocity':[2,0,0],'phase':'locomotion'}
        action=jump_action();action['phases'][0]['end_velocity']=[2,0,0]
        first,end,report=simulate(prepare_scene(s),action)
        next_action={'actor_id':'player','phases':[{'type':'move','gait':'run','delta':[2,0,0],
            'duration_sec':1.,'end_velocity':[2,0,0],'criterion':'Continue running'}]}
        second,_,report2=simulate(end,next_action,1.)
        points=np.array([s['entities'][0]['position']]+[f['entities'][0]['position'] for f in first['frames']+second['frames']])
        np.testing.assert_allclose(np.diff(points[:,0]),2/24,atol=1e-10)
        self.assertTrue(report['valid'] and report2['valid'])


if __name__=='__main__':unittest.main()
