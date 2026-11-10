import unittest
from copy import deepcopy
import numpy as np
import tempfile
from pathlib import Path
from scripts.test_action_semantics import scene, obstacle
from scripts.test_whitebox_world import fixture
from whitebox.controls import Controller
from whitebox.motion import simulate
from scripts.reconcile_whitebox_video import align_action


def trace(keys, ticks=72, mouse=None):
    return {'type':'controls','actor_id':'player','goal':'Recorded input',
            'inputs':[{'keys':keys,'mouse':mouse or [0,0]} for _ in range(ticks)]}


def forward_scene():
    s=scene();s['camera']={'position':[0,2,-5],'target':[0,1,0],'fov':50}
    return s


class ControlsTests(unittest.TestCase):
    def test_diagonal_has_no_speed_bonus_and_replays(self):
        s=forward_scene();snapshot=deepcopy(s)
        straight,end,_=simulate(s,trace(['w']))
        diagonal,other,_=simulate(s,trace(['w','d']))
        self.assertEqual(s,snapshot)
        self.assertEqual(simulate(s,trace(['w']))[0],straight)
        self.assertAlmostEqual(np.linalg.norm(end['entities'][0]['position']),np.linalg.norm(other['entities'][0]['position']))
        self.assertGreater(end['entities'][0]['position'][2],5)
        np.testing.assert_allclose(end['entities'][0]['motion']['velocity'],[0.,0.,2.2],atol=1e-12)

    def test_split_segment_preserves_velocity_and_camera(self):
        s=forward_scene();a=trace(['w'],72,[2,1])
        _,middle,_=simulate(s,a)
        second,end,_=simulate(middle,a,start_time=3.)
        whole,full,_=simulate(s,trace(['w'],144,[2,1]))
        np.testing.assert_allclose(end['entities'][0]['position'],full['entities'][0]['position'],atol=1e-10)
        np.testing.assert_allclose(end['camera']['position'],full['camera']['position'],atol=1e-10)
        self.assertGreater(np.linalg.norm(second['frames'][0]['entities'][0]['motion']['velocity']),2)

    def test_wall_stops_input_without_rejecting_trace(self):
        s=forward_scene();s['entities'].append(obstacle('wall',[0,1,2],[5,2,.3]))
        _,end,report=simulate(s,trace(['w','shift']))
        self.assertTrue(report['valid'])
        self.assertLess(end['entities'][0]['position'][2],1.53)
        self.assertTrue(any(e['type']=='movement_blocked' for e in report['events']))

    def test_crouch_toggle_and_stand_clearance(self):
        c=Controller(forward_scene(),'player')
        for i in range(12):c.step({'keys':['c']})
        self.assertAlmostEqual(c.actor['posture_weights']['crouched'],1.)
        c.scene['entities'].append(obstacle('beam',[0,1.55,0],[2,.5,2]))
        for i in range(12):c.step({'keys':['stand']})
        self.assertLess(c.actor['posture_weights']['standing'],1.)
        self.assertTrue(any(e['type']=='blocked_posture' for e in c.events))

    def test_jump_lands_and_airborne_export_fails(self):
        a=trace([],72);a['inputs'][0]['keys']=['space']
        tl,end,report=simulate(forward_scene(),a)
        self.assertGreater(max(f['entities'][0]['position'][1] for f in tl['frames']),.8)
        self.assertEqual(end['entities'][0]['support_state'],'grounded')
        self.assertEqual([e['type'] for e in report['events']],['planned_takeoff','planned_landing'])
        a['inputs']=a['inputs'][:8]
        with self.assertRaisesRegex(ValueError,'airborne'):simulate(forward_scene(),a)

    def test_mouse_pitch_yaw_preserves_intrinsics(self):
        s=forward_scene();c=Controller(s,'player');before=deepcopy(c.scene['camera'])
        c.step({'keys':[],'mouse':[20,10]})
        self.assertNotEqual(before['position'],c.scene['camera']['position'])
        self.assertEqual(before['fov'],c.scene['camera']['fov'])

    def test_jumping_into_wall_does_not_suspend_gravity(self):
        s=forward_scene();s['entities'].append(obstacle('wall',[0,2,1],[5,4,.2]))
        a=trace(['w'],72);a['inputs'][0]['keys']=['w','space']
        _,end,report=simulate(s,a)
        self.assertEqual(end['entities'][0]['support_state'],'grounded')
        self.assertAlmostEqual(end['entities'][0]['position'][1],0.)
        self.assertTrue(any(e['type']=='movement_blocked' for e in report['events']))

    def test_attack_contact_is_not_movement_failure(self):
        s=fixture();s['entities']=[e for e in s['entities'] if e['id'] in ('player','staff','guard')]
        for e in s['entities']:
            if e['kind']=='actor':e['asset_id']='gta_human'
            if e['id']=='guard':e['position']=[.7,0,1.];e['behavior']={'type':'guard','target':'player','speed':0,'strafe':False}
        a=trace([],72);a['inputs'][0]['keys']=['attack']
        _,end,report=simulate(s,a)
        self.assertTrue(report['valid'])
        self.assertTrue(any(e['type']=='proxy_contact' for e in report['events']),report)

    def test_no_implicit_stop_or_padding(self):
        self.assertEqual(align_action(trace(['w']))[1]['extra_settle_ticks'],0)
        with self.assertRaisesRegex(ValueError,'aligned window'):align_action(trace(['w'],71))

    def test_unsupported_key_is_explicit(self):
        with self.assertRaisesRegex(ValueError,'Unsupported'):simulate(forward_scene(),trace(['prone']))

    def test_npc_policy_runs_offscreen(self):
        s=fixture()
        _,end,_=simulate(s,trace([],72))
        self.assertNotEqual(end['entities'][2]['position'],s['entities'][2]['position'])

    def test_world_tool_records_without_commit_and_rejects_tampering(self):
        from harness.pi.world_tools import dispatch,save,check_plan
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);save(root/'scene.json',forward_scene());save(root/'action.json',trace(['w']))
            db=str(root/'world.sqlite')
            dispatch('initialize',{'scene':str(root/'scene.json'),'db':db})
            before=dispatch('status',{'db':db})
            dispatch('plan',{'db':db,'action':str(root/'action.json'),'output':str(root/'plan')})
            check_plan(root/'plan')
            self.assertEqual(before,dispatch('status',{'db':db}))
            save(root/'plan/action.json',trace(['s']))
            with self.assertRaisesRegex(ValueError,'PLAN_CHANGED'):check_plan(root/'plan')


if __name__=='__main__':unittest.main()
