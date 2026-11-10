from copy import deepcopy
import json
from pathlib import Path
import unittest

from scripts import test_video_reconcile as base
from whitebox.scene_refinement import apply_proposal,commit_refinement
from whitebox.world import scene_from_db,put_entity


class SceneRefinementTests(unittest.TestCase):
    def setUp(self):
        base.VideoReconcileTests.setUp(self)
        item = next(e for e in self.before['entities'] if e['kind']=='box')
        item['provenance']='INFERRED'
        put_entity(self.db,item,'Test inferred static primitive')
    tearDown = base.VideoReconcileTests.tearDown

    def proposal(self):
        item = next(e for e in self.before['entities'] if e['kind']=='box' and e['provenance']=='INFERRED')
        return {'evidence':['fixture visible static anchor'],'uncertainty':['depth not measured'],
                'edits':[{'id':item['id'],'position':[8,3,8],'evidence':['fixture revised placement']}]}

    def validation(self):
        return {'current':{'usable_for_camera_alignment':True},
                'joint':{'consistent':True,'evidence':['fixture dynamic alignment']},
                'history':{'consistent':True,'evidence':['fixture historical alignment']}}

    def test_revision_preserves_dynamic_state_and_time(self):
        before = deepcopy(self.before)
        candidate = commit_refinement(self.db,before,self.proposal(),0,self.validation(), {})
        self.assertEqual([e for e in candidate['entities'] if e['kind'] in {'actor','staff'}],
                         [e for e in before['entities'] if e['kind'] in {'actor','staff'}])
        self.assertEqual(self.db.get_meta('tick'),0)
        self.assertEqual(self.db.get_meta('whitebox_history'),[])
        self.assertEqual(self.db.get_meta('whitebox_representation_revisions')[0]['before'],before)

    def test_no_implicit_actor_edit_or_spawn(self):
        for identity in ['player','new_object']:
            proposal=self.proposal();proposal['edits'][0]['id']=identity
            with self.assertRaises(ValueError):apply_proposal(self.before,proposal)

    def test_failed_visual_audit_and_stale_revision_do_not_commit(self):
        for revision,validation in [(0,{}),(2,{'current':{'usable_for_camera_alignment':True}})]:
            with self.assertRaises(ValueError):commit_refinement(self.db,self.before,self.proposal(),revision,validation,{})
            self.assertEqual(scene_from_db(self.db),self.before)
            self.assertEqual(self.db.get_meta('whitebox_revision'),0)

    def test_new_wall_cannot_intersect_prior_actor_path(self):
        old=deepcopy(self.before)
        old['entities'][0]['position']=[4,0,4]
        timeline=Path(self.temp.name)/'history.json'
        timeline.write_text(json.dumps({'frames':[{'time':1.,'entities':old['entities']}]}))
        prior=[{'artifacts':{'corrected_timeline':str(timeline)},'end':1.}]
        self.db.set_meta('whitebox_history',prior)
        proposal=self.proposal();proposal['edits'][0].update(position=[4,1,4],size=[1,2,1])
        with self.assertRaisesRegex(ValueError,'archived actor paths'):
            commit_refinement(self.db,self.before,proposal,0,self.validation(),{})
        self.assertEqual(scene_from_db(self.db),self.before)
        self.assertEqual(self.db.get_meta('whitebox_history'),prior)

    def test_arbitrary_ids_and_observed_geometry_are_not_special_cased(self):
        scene=deepcopy(self.before)
        item=next(e for e in scene['entities'] if e['kind']=='box')
        item['id']='unrelated_structure_92'
        proposal=self.proposal();proposal['edits'][0]['id']=item['id']
        candidate=apply_proposal(scene,proposal)
        self.assertEqual(next(e for e in candidate['entities'] if e['id']==item['id'])['position'],[8,3,8])
        item['provenance']='OBSERVED'
        with self.assertRaises(ValueError):apply_proposal(scene,proposal)

    def test_archived_states_remain_unchanged_after_successful_revision(self):
        timeline=Path(self.temp.name)/'history.json'
        archived={'frames':[{'time':1.,'entities':self.before['entities']}]}
        timeline.write_text(json.dumps(archived))
        prior=[{'artifacts':{'corrected_timeline':str(timeline)},'end':1.}]
        self.db.set_meta('whitebox_history',prior);self.db.set_meta('tick',1.)
        commit_refinement(self.db,self.before,self.proposal(),0,self.validation(),{})
        self.assertEqual(self.db.get_meta('whitebox_history'),prior)
        self.assertEqual(json.loads(timeline.read_text()),archived)
        self.assertEqual(self.db.get_meta('tick'),1.)

    def test_missing_actor_alignment_cannot_commit(self):
        validation=self.validation();validation.pop('joint')
        with self.assertRaises(ValueError):commit_refinement(self.db,self.before,self.proposal(),0,validation,{})
        self.assertEqual(self.db.get_meta('whitebox_revision'),0)

    def test_explicit_missing_primitive_needs_validation_and_preserves_ids(self):
        entity=deepcopy(next(e for e in self.before['entities'] if e['kind']=='box'))
        entity.update(id='previously_unmodeled',position=[8,3,8],provenance='INFERRED')
        proposal={'edits':[],'evidence':['prior accepted views'], 'uncertainty':['estimated depth'],
                  'additions':[{'entity':entity,'evidence':['visible architectural mass']} ]}
        with self.assertRaises(ValueError):commit_refinement(self.db,self.before,proposal,0,{}, {})
        self.assertIsNone(self.db.get_object(entity['id']))
        commit_refinement(self.db,self.before,proposal,0,self.validation(),{})
        current=scene_from_db(self.db)
        self.assertEqual([e['id'] for e in current['entities']],
                         [e['id'] for e in self.before['entities']]+[entity['id']])
        self.assertEqual(self.db.get_meta('whitebox_history'),[])


if __name__=='__main__':unittest.main()
