import unittest
import numpy as np
from scripts.test_whitebox_world import fixture
from whitebox.motion import simulate,motion_curve,prepare_scene,validate_action
from whitebox.video_reconcile import build_correction,video_alignment
from scripts.reconcile_whitebox_video import align_action
from renderer.continuation import window
import importlib.util
import tempfile
from pathlib import Path


def moving(duration=5.125):
    return {'actor_id':'player','phases':[{'type':'move','delta':[.5,0,0],
        'end_velocity':[.1,0,0],'duration_sec':duration,'criterion':'Keep walking through boundary'}]}


class ContinuousMotionTests(unittest.TestCase):
    def test_camera_translation_keeps_speed_through_two_segments(self):
        scene=prepare_scene(fixture())
        rates={'translation_velocity':[.5,0,0],'orbit_velocity_deg':0.,'distance_velocity':0.}
        scene['camera']['motion']=rates
        action={'actor_id':'player','phases':[{'type':'wait','duration_sec':3,
            'criterion':'Camera keeps moving','camera_delta':[1.5,0,0],'camera_end_motion':rates}]}
        first,end,_=simulate(scene,action)
        second,_,_=simulate(end,action,3.)
        points=[scene['camera']['position']]+[f['camera']['position'] for f in first['frames']+second['frames']]
        np.testing.assert_allclose(np.diff(points,axis=0),np.tile([.5/24,0,0],(144,1)),atol=1e-10)
        self.assertEqual(end['camera']['motion'],rates)
        action['phases'][0].pop('camera_end_motion')
        with self.assertRaisesRegex(ValueError,'Moving camera'):simulate(end,action,3.)

    def test_camera_orbit_keeps_angular_speed_at_boundary(self):
        scene=prepare_scene(fixture())
        rates={'translation_velocity':[0,0,0],'orbit_velocity_deg':5.,'distance_velocity':0.}
        scene['camera']['motion']=rates
        action={'actor_id':'player','phases':[{'type':'wait','duration_sec':3,
            'criterion':'Continue orbit','camera_orbit_yaw_deg':15.,'camera_end_motion':rates}]}
        first,end,_=simulate(scene,action)
        second,_,_=simulate(end,action,3.)
        cameras=[scene['camera']]+[f['camera'] for f in first['frames']+second['frames']]
        offsets=np.array([np.subtract(c['position'],c['target']) for c in cameras])
        angles=np.unwrap(np.arctan2(offsets[:,0],offsets[:,2]))
        np.testing.assert_allclose(np.diff(angles),np.deg2rad(5)/24,atol=1e-10)

    def test_camera_motion_cannot_acquire_padding_stop(self):
        action={'actor_id':'player','phases':[{'type':'wait','duration_sec':5,
            'criterion':'Move camera','camera_delta':[1,0,0],
            'camera_end_motion':{'translation_velocity':[.2,0,0],'orbit_velocity_deg':0.,'distance_velocity':0.}}]}
        with self.assertRaisesRegex(ValueError,'exact generation window'):align_action(action)

    def test_camera_orbit_and_return_preserve_world_and_radius(self):
        scene=prepare_scene(fixture())
        action={'actor_id':'player','phases':[{'type':'wait','duration_sec':2,
            'criterion':'Orbit camera; keep actor planted','camera_orbit_yaw_deg':90}]}
        timeline,end,report=simulate(scene,action)
        self.assertTrue(report['valid'])
        radius=np.linalg.norm(np.subtract(scene['camera']['position'],scene['camera']['target']))
        for frame in timeline['frames']:
            np.testing.assert_allclose(frame['camera']['target'],scene['camera']['target'])
            self.assertAlmostEqual(np.linalg.norm(np.subtract(frame['camera']['position'],frame['camera']['target'])),radius)
            np.testing.assert_allclose(frame['entities'][0]['position'],scene['entities'][0]['position'])
        action['phases'][0]['camera_orbit_yaw_deg']=-90
        _,returned,_=simulate(end,action,start_time=2)
        np.testing.assert_allclose(returned['camera']['position'],scene['camera']['position'],atol=1e-12)
        np.testing.assert_allclose(returned['camera']['target'],scene['camera']['target'],atol=1e-12)

    def test_invalid_camera_orbit_rejected(self):
        scene=prepare_scene(fixture())
        for angle in (float('nan'),181,True):
            with self.assertRaises(ValueError):validate_action(scene,{'actor_id':'player','phases':[
                {'type':'wait','duration_sec':1,'criterion':'orbit','camera_orbit_yaw_deg':angle}]})
        with self.assertRaises(ValueError):validate_action(scene,{'actor_id':'player','phases':[
            {'type':'wait','duration_sec':1,'criterion':'orbit','camera_orbit_yaw_deg':90,'camera_delta':[0,0,0]}]})

    def test_camera_follows_actual_actor_path_with_constant_relative_framing(self):
        scene=prepare_scene(fixture());action=moving()
        action['phases'][0]['camera_follow_actor']='player'
        timeline,_,_=simulate(scene,action)
        for frame in timeline['frames']:
            delta=np.subtract(frame['entities'][0]['position'],scene['entities'][0]['position'])
            for key in ('position','target'):
                np.testing.assert_allclose(np.subtract(frame['camera'][key],scene['camera'][key]),delta,atol=1e-12)
        action['phases'][0]['camera_delta']=[0,0,0]
        with self.assertRaises(ValueError):validate_action(scene,action)

    def test_outward_dolly_is_continuous_and_preserves_target_follow(self):
        scene=prepare_scene(fixture());action=moving()
        action['phases'][0].update(camera_follow_actor='player',camera_distance_delta=.6)
        timeline,end,_=simulate(scene,action)
        distance=lambda camera:np.linalg.norm(np.subtract(camera['position'],camera['target']))
        self.assertAlmostEqual(distance(end['camera'])-distance(scene['camera']),.6)
        distances=[distance(scene['camera'])]+[distance(f['camera']) for f in timeline['frames']]
        self.assertTrue(np.all(np.diff(distances)>=0));self.assertLess(distances[1]-distances[0],.001)
        action['phases'][0]['camera_distance_delta']=float('nan')
        with self.assertRaises(ValueError):validate_action(scene,action)

    def test_prompt_boundary_keeps_ownership_without_procedural_joint_constraints(self):
        from agent.render_contract import boundary_actor
        scene=prepare_scene(fixture());actor=scene['entities'][0]
        actor['weapon_angle']=55;actor['grip']='both'
        boundary=boundary_actor(actor,scene['entities'])
        self.assertNotIn('pose',boundary);self.assertNotIn('grip',boundary)
        self.assertNotIn('weapon_angle',boundary)
        self.assertEqual(boundary['held_objects'][0]['holder'],'player')

    def test_curve_boundary_and_derivative(self):
        for u,p,v in [(0,[1,0,2],[.2,0,0]),(1,[2,0,2],[.4,0,0])]:
            actual,velocity=motion_curve([1,0,2],[1,0,0],[.2,0,0],[.4,0,0],3,u)
            np.testing.assert_allclose(actual,p);np.testing.assert_allclose(velocity,v)

    def test_moving_tail_and_next_incoming_velocity(self):
        timeline,end,report=simulate(prepare_scene(fixture()),moving())
        self.assertTrue(report['valid'],report)
        self.assertGreater(end['entities'][0]['position'][0]-timeline['frames'][-2]['entities'][0]['position'][0],0)
        second= moving();second['phases'][0]['delta']=[.5125,0,0]
        next_tl,_,report=simulate(end,second,timeline['frames'][-1]['time'])
        self.assertTrue(report['valid'],report)
        np.testing.assert_allclose(next_tl['frames'][0]['entities'][0]['motion']['velocity'],[.1,0,0])

    def test_cannot_implicitly_stop(self):
        _,scene,_=simulate(prepare_scene(fixture()),moving())
        with self.assertRaises(ValueError):validate_action(scene,{'actor_id':'player','phases':[
            {'type':'wait','duration_sec':1,'criterion':'stop'}]})

    def test_no_alignment_wait_for_moving_boundary(self):
        with self.assertRaises(ValueError):align_action(moving(5))
        self.assertEqual(align_action(moving())[0],moving())
        self.assertEqual(align_action(moving(4.25),prefix_frames=22)[0],moving(4.25))

    def test_overlap_does_not_advance_clock(self):
        schedule=window();self.assertEqual(schedule['new_ticks'],102)
        timeline,_,_=simulate(prepare_scene(fixture()),moving(4.25),start_time=20)
        alignment=video_alignment(timeline,(np.arange(schedule['continuation_frames'])/24).tolist())
        self.assertAlmostEqual(alignment['world_end'],24.25)
        self.assertEqual(schedule['trim_start'],21)

    @unittest.skipUnless(importlib.util.find_spec('av'),'PyAV available in server environment')
    def test_encoded_overlap_trim_has_exact_frame_count_and_rate(self):
        from renderer.continuation import save_video,trim_generated,read_video
        with tempfile.TemporaryDirectory() as folder:
            raw=Path(folder)/'raw.mp4';out=Path(folder)/'continuation.mp4'
            pixels=np.full((124,32,32,3),96,dtype=np.uint8)
            save_video(pixels,raw)
            trim_generated(raw,out,window())
            self.assertEqual(read_video(out).shape,(103,32,32,3))

    def test_endpoint_velocity_needs_video_evidence(self):
        scene=prepare_scene(fixture());timeline,_,_=simulate(scene,moving())
        review={'decision':'ACCEPT','history_consistent':True,'action_complete':True,'evidence':['fixture'],
            'actor_assessments':[{'id':e['id'],'status':'CONFIRMED_PLAN','evidence':['fixture']}
                                  for e in scene['entities'] if e['kind']=='actor']}
        with self.assertRaises(ValueError):build_correction(scene,timeline,review)
        review['motion_assessments']=[{'id':'player','velocity':[0,0,0],'phase':'stationary',
                                      'evidence':['stopped in final interval'],'uncertainty':['coarse estimate']}]
        _,after,_,_=build_correction(scene,timeline,review)
        self.assertEqual(after['entities'][0]['motion']['velocity'],[0,0,0])
        self.assertEqual(after['entities'][0]['motion']['provenance'],'AGENT_ACCEPTED_VIDEO_ESTIMATE')

    def test_optional_npc_endpoint_motion_is_preserved(self):
        scene=prepare_scene(fixture());timeline,_,_=simulate(scene,moving())
        assessment=lambda identity:dict(id=identity,velocity=[0,0,0],phase='stationary',
            evidence=['visible final interval'],uncertainty='coarse video estimate')
        review={'decision':'ACCEPT','history_consistent':True,'action_complete':True,'evidence':['fixture'],
            'actor_assessments':[{'id':e['id'],'status':'CONFIRMED_PLAN','evidence':['fixture']}
                                 for e in scene['entities'] if e['kind']=='actor'],
            'motion_assessments':[assessment('player'),assessment('guard')]}
        _,after,_,_=build_correction(scene,timeline,review)
        guard=next(e for e in after['entities'] if e['id']=='guard')
        self.assertEqual(guard['motion']['provenance'],'AGENT_ACCEPTED_VIDEO_ESTIMATE')
        self.assertEqual(guard['motion']['velocity'],[0,0,0])
        for invalid in [[assessment('guard')], [assessment('player'),assessment('wall')],
                        [assessment('player'),assessment('player')]]:
            with self.subTest(ids=[a['id'] for a in invalid]),self.assertRaises(ValueError):
                build_correction(scene,timeline,dict(review,motion_assessments=invalid))


if __name__=='__main__':unittest.main()
