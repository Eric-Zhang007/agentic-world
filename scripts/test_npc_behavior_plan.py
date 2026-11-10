"""Prospective NPC policies must not rewrite their starting state or history."""
import sys
from pathlib import Path
from copy import deepcopy
import unittest
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from whitebox.motion import simulate,validate_action
from whitebox.world import initialize,scene_from_db
from whitebox.video_reconcile import commit_video
from state.db import WorldDB


def scene():
    def actor(name,x,behavior):
        return dict(id=name,kind='actor',category='person',position=[x,0,0],rotation_y=0,
                    size=[.5,1.7,.4],color=[.5,.5,.5],provenance='INFERRED',reason='test',behavior=behavior)
    return dict(schema_version=1,seed=1,lore='test',terrain=dict(size=[30,30],seed=1,
                resolution=16,height_amplitude=0,flatten_radius=8),
                camera=dict(position=[0,3,7],target=[0,1,0]),entities=[
                    actor('player',0,dict(type='player')),
                    actor('npc',4,dict(type='patrol',speed=0,waypoints=[[4,0,0],[4,0,4]]))])


class PolicyTests(unittest.TestCase):
    def test_prospective_concurrent_movement(self):
        before=scene();saved=deepcopy(before)
        action=dict(goal='walk together',actor_id='player',phases=[dict(type='move',delta=[0,0,1],duration_sec=3,criterion='advance')],
                    npc_behaviors=[dict(id='npc',reason='continue walking',behavior=dict(type='patrol',speed=.5,waypoints=[[4,0,4],[4,0,0]]))])
        timeline,after,report=simulate(before,action)
        self.assertTrue(report['valid'])
        self.assertEqual(before,saved)
        self.assertGreater(after['entities'][1]['position'][2],1.)
        self.assertEqual(after['entities'][0]['position'][2],1.)
        self.assertEqual(after['entities'][1]['behavior']['speed'],.5)
        action['npc_behaviors'][0]['id']='player'
        with self.assertRaises(ValueError):validate_action(before,action)

    def test_invalid_behavior(self):
        action=dict(goal='test',actor_id='player',phases=[dict(type='wait',duration_sec=3,criterion='wait')],
            npc_behaviors=[dict(id='npc',reason='walk',behavior=dict(type='patrol',speed=-1,waypoints=[]))])
        with self.assertRaises(ValueError):validate_action(scene(),action)

    def test_policy_persists_only_with_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            db=WorldDB(Path(directory)/'world.sqlite')
            try:
                initialize(db,scene());before=scene_from_db(db)
                action=dict(goal='NPC walks',actor_id='player',phases=[dict(type='wait',duration_sec=3,criterion='watch')],
                    npc_behaviors=[dict(id='npc',reason='NPC continues its own walk',behavior=dict(type='patrol',speed=.5,waypoints=[[4,0,4],[4,0,0]]))])
                timeline,_,_=simulate(before,action)
                self.assertEqual(scene_from_db(db),before)
                review=dict(decision='ACCEPT',history_consistent=True,action_complete=True,evidence=['test fixture'],
                    actor_assessments=[dict(id=x,status='CONFIRMED_PLAN',evidence=['test fixture']) for x in ['player','npc']],
                    unsupported_state_changes=[],corrections=[],pose_corrections=[],events=[])
                commit_video(db,before,timeline,action,review,0,dict(video='fixture',last_frame='fixture'))
                after=scene_from_db(db)
                self.assertEqual(after['entities'][1]['behavior']['speed'],.5)
                self.assertEqual(db.get_meta('whitebox_history')[0]['action']['npc_behaviors'],action['npc_behaviors'])
                self.assertEqual(db.get_meta('whitebox_revision'),1)
            finally:db.connection.close()


if __name__=='__main__':unittest.main()
