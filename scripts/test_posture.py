import unittest
from copy import deepcopy
import tempfile
from pathlib import Path
import numpy as np
from scripts.test_whitebox_world import fixture
from whitebox.motion import prepare_scene,simulate,rig,point_segment_distance
from whitebox.posture import contacts
from whitebox.world import initialize,scene_from_db,validate_scene
from whitebox.video_reconcile import build_correction,commit_video
from state.db import WorldDB


def pose_action(target='low_guard'):
    return {'actor_id':'guard','goal':'Lower into a supported guard without attacking',
            'phases':[{'type':'posture','duration_sec':3.,'target_posture':target,'criterion':'Supported posture'}]}


class PostureTests(unittest.TestCase):
    def test_pose_and_support_survive_wait(self):
        scene=prepare_scene(fixture());timeline,end,report=simulate(scene,pose_action())
        self.assertTrue(report['valid'],report)
        actor=end['entities'][1]
        self.assertEqual(actor['posture_weights']['low_guard'],1)
        self.assertEqual(contacts(actor),['left_foot','right_foot','left_hand','right_hand'])
        self.assertLess(actor['pose']['head'][1],actor['size'][1]*.4)
        for side in ('left','right'):
            self.assertAlmostEqual(actor['pose'][side+'_hand'][1],actor['size'][1]*.044)
        for key,point in actor['pose'].items():
            self.assertGreaterEqual(point[1]-actor['size'][1]*(.095 if key=='head' else .044),-1e-8)
        wait={'actor_id':'player','phases':[{'type':'wait','duration_sec':1,'criterion':'Watch'}]}
        later,after,report=simulate(end,wait,3)
        self.assertTrue(report['valid'],report)
        self.assertEqual(actor['pose'],later['frames'][0]['entities'][1]['pose'])
        self.assertEqual(actor['position'],after['entities'][1]['position'])

    def test_crouch_and_rise(self):
        _,end,report=simulate(prepare_scene(fixture()),pose_action('crouched'))
        self.assertTrue(report['valid'],report)
        _,standing,report=simulate(end,pose_action('standing'),3)
        self.assertTrue(report['valid'],report)
        self.assertAlmostEqual(standing['entities'][1]['pose']['head'][1],1.8*.93)

    def test_no_low_guard_with_held_staff(self):
        action=pose_action();action['actor_id']='player'
        with self.assertRaises(ValueError):simulate(prepare_scene(fixture()),action)

    def test_low_postures_do_not_silently_slide(self):
        for target in ('crouched','low_guard'):
            _,end,_=simulate(prepare_scene(fixture()),pose_action(target))
            move={'actor_id':'guard','phases':[{'type':'move','duration_sec':1.,'delta':[.2,0,0],'criterion':'Move'}]}
            with self.assertRaises(ValueError):simulate(end,move)

    def test_contact_spheres_do_not_penetrate_during_transition(self):
        timeline,_,_=simulate(prepare_scene(fixture()),pose_action())
        for frame in timeline['frames']:
            actor=frame['entities'][1]
            for key,point in actor['pose'].items():
                self.assertGreaterEqual(point[1]-actor['size'][1]*(.095 if key=='head' else .044),-1e-8)

    def test_single_hand_still_holds_same_staff(self):
        s=prepare_scene(fixture());actor=s['entities'][0];staff=s['entities'][3]
        actor['grip']='right'
        pose,weapon,errors=rig(actor,0,0,held=staff,terrain=s['terrain'])
        self.assertFalse(errors)
        self.assertLess(point_segment_distance(pose['right_hand'],weapon['a'],weapon['b']),1e-8)
        self.assertGreater(point_segment_distance(pose['left_hand'],weapon['a'],weapon['b']),.05)
        self.assertEqual(staff['holder'],'player')
        _,_,report=simulate(s,{'actor_id':'player','phases':[{'type':'wait','duration_sec':1,'criterion':'Hold'}]})
        self.assertTrue(report['valid'],report)

    def test_invalid_weights_and_support(self):
        for change in ({'posture_weights':{'standing':2}}, {'posture_weights':{'flying':1}},
                       {'support_contacts':['head']},{'grip':'none'}):
            s=fixture();s['entities'][1].update(change)
            with self.assertRaises(ValueError):validate_scene(s)

    def test_observed_posture_commits_and_continues_without_rewriting_history(self):
        s=prepare_scene(fixture());action={'actor_id':'player','phases':[{'type':'wait','duration_sec':3,'criterion':'Observe'}]}
        timeline,_,_=simulate(s,action)
        review={'decision':'ACCEPT','history_consistent':True,'action_complete':True,'evidence':['fixture'],
                'actor_assessments':[{'id':e['id'],'status':'CORRECTED' if e['id']=='guard' else 'CONFIRMED_PLAN',
                                     'evidence':['fixture']} for e in s['entities'] if e['kind']=='actor'],
                'pose_corrections':[{'id':'guard','evidence':['fixture lowers after one second'],
                    'keyframes':[{'frame_index':23,'posture':'standing'}, {'frame_index':71,'posture':'low_guard'}]}],
                'events':[]}
        corrected,end,_,_=build_correction(s,timeline,review)
        self.assertAlmostEqual(corrected['frames'][23]['entities'][1]['pose']['head'][1],1.8*.93)
        self.assertLess(end['entities'][1]['pose']['head'][1],1.8*.4)
        with tempfile.TemporaryDirectory() as d:
            db=WorldDB(Path(d)/'world.sqlite');initialize(db,s)
            prior=[{'action':{},'events':['existing history']}];db.set_meta('whitebox_history',prior)
            commit_video(db,s,timeline,action,review,0,{'video':'fixture.mp4','last_frame':'fixture.png'})
            self.assertEqual(db.get_meta('whitebox_history')[0],prior[0])
            restored=scene_from_db(db)
            self.assertEqual(restored,end)
            later,_,_=simulate(restored,action,db.get_meta('tick'))
            self.assertEqual(later['frames'][0]['entities'][1]['posture_weights'],end['entities'][1]['posture_weights'])
            db.connection.close()
        bad=deepcopy(review);bad['pose_corrections'][0]['keyframes'].pop()
        with self.assertRaises(ValueError):build_correction(s,timeline,bad)


if __name__=='__main__':unittest.main()
