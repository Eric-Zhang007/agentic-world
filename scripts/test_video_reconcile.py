from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.test_whitebox_world import fixture, action
from state.db import WorldDB
from whitebox.world import initialize, scene_from_db
from whitebox.motion import prepare_scene, simulate
from whitebox.video_reconcile import build_correction, commit_video,video_alignment


class VideoReconcileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = WorldDB(Path(self.temp.name)/'world.sqlite')
        self.before = prepare_scene(fixture())
        initialize(self.db, self.before)
        self.action = action()
        self.timeline, self.intended_end, _ = simulate(self.before, self.action)
        self.review = {'decision': 'ACCEPT', 'history_consistent': True, 'action_complete': False,
                       'actor_assessments':[{'id':e['id'],'status':'CORRECTED' if e['id']=='player' else 'UNOBSERVED_SIMULATED',
                                             'evidence':['fixture observation status']} for e in self.before['entities'] if e['kind']=='actor'],
                       'evidence': ['fixture: useful partial motion, shorter than planned'],
                       'continuation_goal': 'Continue the remaining sidestep',
                       'corrections': [{'id': 'player', 'evidence': ['fixture endpoint estimate'],
                                        'keyframes': [{'frame_index': 23, 'position_offset': [-.3,0,0]},
                                                      {'frame_index': 71, 'position_offset': [-.3,0,0]}]}],
                       'events': [{'time_sec': 1., 'entity_ids': ['player'],
                                   'description': 'Partial step', 'evidence': ['fixture']}]}
        self.artifacts = {'video': 'fixture.mp4', 'last_frame': 'fixture.png',
                          'corrected_timeline': 'corrected.json'}

    def tearDown(self):
        self.db.connection.close()
        self.temp.cleanup()

    def commit(self, review=None):
        return commit_video(self.db, self.before, self.timeline, self.action,
                            review or self.review, 0, self.artifacts)

    def test_partial_accept_commits_actual_endpoint_and_advances_time(self):
        self.commit()
        current = scene_from_db(self.db)
        player = next(e for e in current['entities'] if e['id']=='player')
        self.assertAlmostEqual(player['position'][0], .3)
        self.assertAlmostEqual(self.db.get_meta('tick'), 3.)
        self.assertFalse(self.db.get_meta('whitebox_video_action')['action_complete'])
        history=self.db.get_meta('whitebox_history')
        self.assertFalse(history[-1]['action_complete'])
        self.assertIn('planned intention only',history[-1]['action_semantics'])
        wait = {'actor_id': 'player', 'phases': [{'type': 'wait','duration_sec': 1,'criterion': 'hold'}]}
        next_timeline, _, _ = simulate(current, wait, self.db.get_meta('tick'))
        self.assertAlmostEqual(next_timeline['frames'][0]['entities'][0]['position'][0], .3)

    def test_whole_path_and_attached_weapon_move_together(self):
        corrected, _, _, _ = build_correction(self.before,self.timeline,self.review)
        frame, planned = corrected['frames'][23], self.timeline['frames'][23]
        for key in ('head','left_hand','right_hand'):
            np.testing.assert_allclose(np.subtract(frame['entities'][0]['pose'][key],planned['entities'][0]['pose'][key]),[-.3,0,0])
        np.testing.assert_allclose(np.subtract(frame['entities'][3]['weapon']['a'],planned['entities'][3]['weapon']['a']),[-.3,0,0])
        self.assertEqual(frame['entities'][2], planned['entities'][2])
        self.assertEqual(frame['entities'][-1], planned['entities'][-1])

    def test_rejection_does_not_change_history_or_world(self):
        with self.assertRaises(ValueError):
            self.commit({**self.review,'decision':'REGENERATE'})
        self.assertEqual(scene_from_db(self.db),self.before)
        self.assertEqual(self.db.get_meta('whitebox_history'),[])

    def test_cannot_replay_or_overwrite_accepted_segment(self):
        self.commit()
        history=self.db.get_meta('whitebox_history')
        with self.assertRaises(ValueError): self.commit()
        self.assertEqual(self.db.get_meta('whitebox_history'),history)

    def test_previous_history_preserved_and_estimates_labeled(self):
        prior=[{'action': {'goal':'already established'},'events': ['prior event']}]
        self.db.set_meta('whitebox_history',prior)
        self.commit()
        history=self.db.get_meta('whitebox_history')
        self.assertEqual(history[0],prior[0])
        self.assertEqual(history[1]['events'][0]['provenance'],'AGENT_ACCEPTED_VIDEO_INTERPRETATION')

    def test_no_static_rewrite_or_implicit_spawn(self):
        for identity in ('wall','new_actor'):
            review=deepcopy(self.review);review['corrections'][0]['id']=identity
            with self.assertRaises(ValueError):self.commit(review)
        self.assertEqual(self.db.get_meta('whitebox_revision'),0)

    def test_bad_event_rolls_back(self):
        review=deepcopy(self.review);review['events'][0]['time_sec']=10
        with self.assertRaises(ValueError):self.commit(review)
        self.assertEqual(scene_from_db(self.db),self.before)

    def test_colliding_corrected_endpoint_does_not_commit(self):
        review = deepcopy(self.review)
        review['corrections'][0]['keyframes'][-1]['position_offset'] = [0, 0, 7]
        with self.assertRaisesRegex(ValueError, 'not executable'):
            self.commit(review)
        self.assertEqual(scene_from_db(self.db), self.before)
        self.assertEqual(self.db.get_meta('whitebox_history'), [])
        self.assertEqual(self.db.get_meta('whitebox_revision'), 0)

    def test_endpoint_event_accepts_roundoff_but_not_later_time(self):
        review=deepcopy(self.review);review['events'][0]['time_sec']=3.
        _,_,events,_=build_correction(self.before,self.timeline,review)
        self.assertLessEqual(events[0]['world_time'],self.timeline['frames'][-1]['time'])
        review['events'][0]['time_sec']=3.001
        with self.assertRaises(ValueError):build_correction(self.before,self.timeline,review)

    def test_endpoint_required_and_finite(self):
        review=deepcopy(self.review);review['corrections'][0]['keyframes'].pop()
        with self.assertRaises(ValueError):self.commit(review)
        review=deepcopy(self.review);review['corrections'][0]['keyframes'][-1]['position_offset'][0]=float('nan')
        with self.assertRaises(ValueError):self.commit(review)

    def test_plan_tampering_rejected(self):
        self.timeline['frames'][-1]['entities'][0]['position'][0]=9
        with self.assertRaises(ValueError):self.commit()
        self.assertEqual(scene_from_db(self.db),self.before)


    def test_no_silent_assumption_of_planned_endpoint(self):
        review=deepcopy(self.review);review.pop('actor_assessments')
        with self.assertRaises(ValueError):self.commit(review)

    def test_unsupported_edit_is_not_ignored(self):
        review=deepcopy(self.review);review['corrections'][0]['keyframes'][-1]['holder']='tiger'
        with self.assertRaises(ValueError):self.commit(review)

    def test_unrepresentable_accepted_event_does_not_commit(self):
        review=deepcopy(self.review)
        review['unsupported_state_changes']=['The staff was dropped and is no longer held.']
        with self.assertRaises(ValueError):self.commit(review)
        self.assertEqual(scene_from_db(self.db),self.before)
        self.assertEqual(self.db.get_meta('whitebox_history'),[])

    def test_original_plan_phase_is_not_an_observed_event(self):
        original=deepcopy(self.timeline)
        corrected,_,events,_=build_correction(self.before,self.timeline,self.review)
        self.assertEqual(self.timeline,original)
        for planned,actual in zip(original['frames'],corrected['frames']):
            self.assertNotIn('phase',actual)
            if 'phase' in planned:self.assertEqual(actual['planned_phase'],planned['phase'])
        self.assertEqual(corrected['accepted_events'],events)
        self.assertIn('not recovered',corrected['pose_scope'])

    def test_alignment_uses_actual_pts(self):
        result=video_alignment(self.timeline,[i/24 for i in range(73)])
        self.assertEqual(result['video_frame_to_plan_index'][0],-1)
        self.assertAlmostEqual(result['world_end'],3.)
        with self.assertRaises(ValueError):video_alignment(self.timeline,[i/25 for i in range(73)])
        with self.assertRaises(ValueError):video_alignment(self.timeline,[i/24 for i in range(72)]+[71/24])

    def test_second_commit_appends_and_starts_from_corrected_state(self):
        self.commit();history=deepcopy(self.db.get_meta('whitebox_history'))
        before=scene_from_db(self.db)
        action={'actor_id':'player','phases':[{'type':'wait','duration_sec':1.,'criterion':'hold'}]}
        timeline,_,_=simulate(before,action,self.db.get_meta('tick'))
        review={**deepcopy(self.review),'corrections':[],'action_complete':True,'events':[],
                'actor_assessments':[{'id':e['id'],'status':'CONFIRMED_PLAN','evidence':['fixture hold']} for e in before['entities'] if e['kind']=='actor']}
        commit_video(self.db,before,timeline,action,review,1,self.artifacts)
        self.assertEqual(self.db.get_meta('whitebox_history')[:-1],history)
        self.assertAlmostEqual(self.db.get_meta('tick'),4.)

    def test_yaw_correction_uses_short_arc(self):
        review=deepcopy(self.review)
        review['corrections'][0]['keyframes']=[
            {'frame_index':23,'position_offset':[0,0,0],'yaw_offset':3.1},
            {'frame_index':71,'position_offset':[0,0,0],'yaw_offset':-3.1}]
        corrected,_,_,_=build_correction(self.before,self.timeline,review)
        actual=corrected['frames'][47]['entities'][0]['rotation_y']
        planned=self.timeline['frames'][47]['entities'][0]['rotation_y']
        self.assertGreater(abs(actual-planned),3.)

    def test_prepare_frame_alignment_only_extends_settling(self):
        from scripts.reconcile_whitebox_video import align_action
        before={'actor_id':'player','phases':[{'type':'move','duration_sec':1.,'delta':[.3,0,0],'criterion':'move'}]}
        after,record=align_action(before)
        self.assertEqual(after['phases'][0],before['phases'][0])
        self.assertEqual(after['phases'][-1]['type'],'wait')
        self.assertEqual((sum(round(p['duration_sec']*24) for p in after['phases'])+1)%17,5)


if __name__=='__main__':unittest.main()
