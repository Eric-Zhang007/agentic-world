import copy
import tempfile
import unittest
from pathlib import Path
from state.db import WorldDB
from whitebox.world import initialize,scene_from_db,reconcile,terrain_height,validate_scene
from whitebox.motion import simulate,commit,validate_geometry,validate_action,prepare_scene


def fixture():
    def actor(identity,pos,behavior):
        return dict(id=identity,kind='actor',category='person',position=pos,rotation_y=0.,size=[.65,1.8,.5],
                    color=[.3,.6,.4],provenance='AGENT_CREATED',reason='Test scene',behavior=behavior)
    return dict(schema_version=1,seed=3,terrain=dict(size=[24,24],resolution=32,seed=3,height_amplitude=.0,flatten_radius=8),
        camera=dict(position=[6,4,-8],target=[0,1,1]),entities=[
        actor('player',[0,0,0],{'type':'player'}),
        actor('guard',[0,0,3],{'type':'guard','target':'player','distance':2.4,'speed':.5}),
        actor('patrol',[-7,0,-8],{'type':'patrol','waypoints':[[-5,0,-8],[-7,0,-8]],'speed':.6}),
        dict(id='staff',kind='staff',category='staff',position=[0,0,0],rotation_y=0.,size=[.08,2.4,.08],length=2.4,
             color=[.8,.6,.2],holder='player',provenance='AGENT_CREATED',reason='Held staff'),
        dict(id='wall',kind='box',category='wall',position=[0,1,7],rotation_y=0.,size=[8,2,.5],
             color=[.5,.5,.5],provenance='AGENT_CREATED',reason='Static wall')])


def action():
    return {'goal':'side step, sweep, recover','actor_id':'player','phases':[
        {'type':'move','delta':[.6,0,0],'duration_sec':1,'criterion':'moved'},
        {'type':'sweep','start_angle_deg':55,'end_angle_deg':-55,'duration_sec':1,'criterion':'sweep'},
        {'type':'recover','duration_sec':1,'criterion':'ready'}]}


class WhiteboxTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.db=WorldDB(Path(self.tmp.name)/'world.sqlite')
    def tearDown(self):self.db.connection.close();self.tmp.cleanup()
    def test_roundtrip_coordinate_contract(self):
        s=fixture();initialize(self.db,s);restored=scene_from_db(self.db)
        self.assertEqual(s,restored)
    def test_seed_repeatability_and_flat_center(self):
        t=fixture()['terrain'];t['height_amplitude']=1
        self.assertEqual(terrain_height(t,1,2),0)
        self.assertEqual(terrain_height(t,10,10),terrain_height(t,10,10))
        other={**t,'seed':4};self.assertNotEqual(terrain_height(t,10,10),terrain_height(other,10,10))
    def test_motion_has_articulation_weapon_and_npcs(self):
        timeline,end,report=simulate(fixture(),action())
        self.assertTrue(report['valid'],report);self.assertEqual(len(timeline['frames']),72)
        poses=timeline['frames'][30]['entities'][0]['pose'];self.assertIn('left_elbow',poses)
        a=timeline['frames'][24]['entities'][3]['weapon'];b=timeline['frames'][40]['entities'][3]['weapon']
        self.assertNotEqual(a,b);self.assertGreater(report['npc_displacements']['patrol'],.5)
        self.assertEqual(end['entities'][-1],fixture()['entities'][-1])
    def test_no_commit_on_visual_rejection(self):
        initialize(self.db,fixture());before=scene_from_db(self.db);tl,end,report=simulate(before,action())
        with self.assertRaises(ValueError):commit(self.db,end,tl,report,{'decision':'REGENERATE','evidence':['bad']},action(),0)
        self.assertEqual(scene_from_db(self.db),before);self.assertEqual(self.db.get_meta('tick'),0)
    def test_two_actions_continue_npc_time(self):
        initialize(self.db,fixture());tl,end,r=simulate(scene_from_db(self.db),action())
        commit(self.db,end,tl,r,{'decision':'ACCEPT','evidence':['unit fixture']},action(),0)
        first=scene_from_db(self.db);p=next(e['position'] for e in first['entities'] if e['id']=='patrol')
        wait={'actor_id':'player','phases':[{'type':'wait','duration_sec':1,'criterion':'wait'}]}
        tl2,end2,r2=simulate(first,wait,self.db.get_meta('tick'))
        self.assertAlmostEqual(tl2['frames'][0]['time'],3+1/24)
        commit(self.db,end2,tl2,r2,{'decision':'ACCEPT','evidence':['unit fixture']},wait,1)
        self.assertNotEqual(p,next(e['position'] for e in end2['entities'] if e['id']=='patrol'))
        self.assertEqual(len(self.db.get_meta('whitebox_history')),2)
    def test_collision_does_not_teleport(self):
        s=fixture();s['entities'][-1]['position']=[1.5,1,0];s['entities'][-1]['size']=[.5,2,2]
        a={'actor_id':'player','phases':[{'type':'move','delta':[3,0,0],'duration_sec':1,'criterion':'cross'}]}
        tl,end,r=simulate(s,a);self.assertFalse(r['valid']);self.assertLess(end['entities'][0]['position'][0],1)
    def test_initial_overlap_detected(self):
        s=fixture();s['entities'][1]['position']=[0,0,0];self.assertTrue(validate_geometry(s))
    def test_no_unsupported_action(self):
        a=action();a['phases'][0]['type']='teleport'
        with self.assertRaises(ValueError):validate_action(fixture(),a)
    def test_history_rewrite_blocked(self):
        initialize(self.db,fixture())
        with self.assertRaises(ValueError):reconcile(self.db,[{'op':'move','id':'wall','position':[0,0,0]}],'renderer drift')
    def test_creative_spawn_allowed_before_history(self):
        initialize(self.db,fixture());e=copy.deepcopy(fixture()['entities'][-1]);e.update(id='new_pillar',position=[5,1,5],size=[1,2,1])
        reconcile(self.db,[{'op':'spawn','entity':e}],'Extend imagined temple')
        self.assertIsNotNone(self.db.get_object('new_pillar'))
    def test_stale_candidate_fails(self):
        initialize(self.db,fixture());tl,end,r=simulate(fixture(),action())
        with self.assertRaises(ValueError):commit(self.db,end,tl,r,{'decision':'ACCEPT','evidence':['ok']},action(),1)
    def test_invalid_behavior(self):
        s=fixture();s['entities'][1]['behavior']={'type':'none'}
        with self.assertRaises(ValueError):validate_scene(s)
    def test_null_behavior_and_bad_camera_delta_rejected(self):
        s=fixture();s['entities'][1]['behavior']=None
        with self.assertRaises(ValueError):validate_scene(s)
        a=action();a['phases'][0]['camera_delta']=2
        with self.assertRaises(ValueError):validate_action(fixture(),a)
    def test_overhead_lintel_not_ground_obstacle(self):
        s=fixture();s['entities'][-1]['position']=[0,4,0]
        self.assertEqual(validate_geometry(s),[])
    def test_candidate_tampering_rejected(self):
        initialize(self.db,fixture());tl,end,r=simulate(fixture(),action())
        end['entities'][0]['position']=[100,0,100]
        with self.assertRaises(ValueError):commit(self.db,end,tl,r,{'decision':'ACCEPT','evidence':['ok']},action(),0)
        self.assertEqual(self.db.get_meta('tick'),0)
    def test_weapon_start_angle_is_continuous(self):
        a=action();a['phases'][1]['start_angle_deg']=-80
        with self.assertRaises(ValueError):validate_action(fixture(),a)
    def test_feet_grounded_on_seeded_terrain(self):
        s=fixture();s['terrain']['height_amplitude']=1.;s['terrain']['flatten_radius']=0
        for e in s['entities']:
            if e['kind']=='actor':e['position'][1]=terrain_height(s['terrain'],e['position'][0],e['position'][2])
        wait={'actor_id':'player','phases':[{'type':'wait','duration_sec':1,'criterion':'idle'}]}
        tl,_,_=simulate(s,wait)
        for f in tl['frames']:
            for e in f['entities']:
                if e['kind']!='actor':continue
                distances=[]
                for key in ('left_foot','right_foot'):
                    x,y,z=e['pose'][key];distances.append(y-e['size'][1]*.044-terrain_height(s['terrain'],x,z))
                self.assertGreaterEqual(min(distances),-1e-8)
                self.assertAlmostEqual(min(distances),0,places=6)
    def test_initial_staff_and_hands_are_resolved(self):
        s=prepare_scene(fixture());self.assertIn('weapon',s['entities'][3])
        self.assertIn('right_hand',s['entities'][0]['pose'])
    def test_contact_is_proxy_event_not_damage(self):
        s=fixture();s['entities'][1]['position']=[0,0,1.4]
        s['entities'][1]['behavior'].update(distance=1.4,speed=0)
        s['entities'][1]['health']=100
        a={'actor_id':'player','phases':[{'type':'sweep','start_angle_deg':55,'end_angle_deg':-55,'duration_sec':1,'criterion':'probe'}]}
        _,end,r=simulate(s,a)
        self.assertTrue(r['events']);self.assertEqual(end['entities'][1]['health'],100)
        self.assertTrue(all(e['type']=='proxy_contact' for e in r['events']))
    def test_creation_and_commit_can_rollback_together(self):
        with self.assertRaises(ValueError):
            with self.db.transaction():
                initialize(self.db,fixture());tl,end,r=simulate(fixture(),action())
                commit(self.db,end,tl,r,{'decision':'ACCEPT'},action(),0)
        self.assertEqual(self.db.query_world(),[])
    def test_reconcile_cannot_spawn_in_accepted_view(self):
        initialize(self.db,fixture());tl,end,r=simulate(fixture(),action())
        commit(self.db,end,tl,r,{'decision':'ACCEPT','evidence':['unit']},action(),0)
        e=copy.deepcopy(fixture()['entities'][-1]);e.update(id='new_wall',position=[0,1,1],size=[1,2,1])
        with self.assertRaises(ValueError):reconcile(self.db,[{'op':'spawn','entity':e}],'Video added a wall')


if __name__=='__main__':unittest.main()
