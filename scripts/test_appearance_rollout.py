"""Memory admission and actual consumer wiring, without external models."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from PIL import Image
from state.db import WorldDB
from whitebox.appearance import AppearanceMemory
from whitebox.appearance_rollout import store_proposal,retrieve,capture_accepted,retrieve_for_task
from agent.render_contract import compile_prompt,renderer_references,identity_text


class MemoryRolloutTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.db=WorldDB(self.root/'world.sqlite')
        self.addCleanup(self.db.connection.close)
        self.scene={'entities':[{'id':'hero','kind':'actor','category':'person','description':'red coat'}],
                    'camera':{}}
        self.record={'provenance':'AGENT_ACCEPTED_VIDEO_INTERPRETATION','revision_after':2,
                     'start':5.,'end':6.,'artifacts':{'video':'accepted.mp4'}}
        self.db.set_meta('whitebox_history',[self.record]);self.db.set_meta('whitebox_revision',2)
        self.path=self.root/'frame.png';Image.new('RGB',(40,20),'red').save(self.path)
        self.frames=[{'path':str(self.path),'timestamp':1.,'frame_id':24}]
        self.row={'object_id':'hero','sample_index':0,'view':'front','prompt':'red coat',
                  'bbox_normalized':[0,0,.5,1],'quality':.9,'evidence':'visible red coat'}
        self.mock=patch('whitebox.appearance_rollout.scene_from_db',return_value=self.scene)
        self.mock.start();self.addCleanup(self.mock.stop)

    def store(self,rows=None):
        return store_proposal(self.db,self.record,'accepted:2',self.frames,
                              {'references':rows if rows is not None else [self.row]})

    def test_accepted_memory_retrieves_real_crops_without_world_edits(self):
        snapshot=self.db.snapshot();history=self.db.get_meta('whitebox_history')
        result=self.store();bundle=retrieve(self.db,self.root/'bundle')
        self.assertEqual(len(result['references']),1)
        with Image.open(bundle['images'][0]) as image:self.assertEqual(image.size,(20,20))
        self.assertIn('<Picture 2>',bundle['prompt'])
        self.assertEqual(bundle['refs'][0]['role'],'identity_only')
        self.assertEqual(self.db.snapshot(),snapshot)
        self.assertEqual(self.db.get_meta('whitebox_history'),history)
        self.assertEqual(self.store(),result)
        self.assertEqual(len(self.db.get_meta(AppearanceMemory.META_KEY)),1)

    def test_invalid_identity_batch_does_not_partially_persist(self):
        with self.assertRaises(ValueError):self.store([self.row,{**self.row,'object_id':'hallucinated_dog'}])
        self.assertEqual(self.db.get_meta(AppearanceMemory.META_KEY,[]),[])

    def test_rejected_or_old_history_cannot_enter_memory(self):
        self.db.set_meta('whitebox_history',[])
        with self.assertRaises(ValueError):self.store()
        self.assertEqual(capture_accepted(self.db,{},self.root/'unused')['status'],'NO_ACCEPTED_VIDEO')

    def test_bad_box_and_duplicate_view_rejected(self):
        for rows in ([{**self.row,'bbox_normalized':[0,0,2,1]}],[self.row,self.row]):
            with self.assertRaises(ValueError):self.store(rows)

    def test_ticks_only_timestamp_uses_accepted_endpoint(self):
        self.record['end']=6.+1/24
        self.db.set_meta('whitebox_history',[self.record])
        self.frames=[dict(self.frames[0],timestamp=0),dict(self.frames[0],timestamp=1)]
        self.store()
        self.assertAlmostEqual(self.db.get_meta(AppearanceMemory.META_KEY)[0]['timestamp'],5.+1/24)

    def test_failed_second_image_rolls_back_metadata_and_owned_assets(self):
        self.frames.append(dict(self.frames[0],path=str(self.root/'missing.png')))
        with self.assertRaises(ValueError):
            self.store([self.row,{**self.row,'view':'back','sample_index':1}])
        self.assertEqual(self.db.get_meta(AppearanceMemory.META_KEY,[]),[])
        self.assertEqual(list((self.root/'reference_memory').glob('*.png')),[])

    def test_network_failure_can_resume_same_bound_capture_directory(self):
        self.db.set_meta('whitebox_last_video','accepted.mp4')
        sampler=SimpleNamespace(sample_frames=lambda *args,**kwargs:self.frames)
        with patch.dict('sys.modules',{'perception.frame_sampler':sampler}),patch(
                'whitebox.appearance_rollout.json_call',side_effect=[RuntimeError('network'),{'references':[self.row]}]) as call:
            with self.assertRaises(RuntimeError):capture_accepted(self.db,{},self.root/'capture')
            result=capture_accepted(self.db,{},self.root/'capture')
            self.assertEqual(result['status'],'CAPTURED')
            self.assertEqual(call.call_count,2)
            self.assertEqual(capture_accepted(self.db,{},self.root/'capture'),result)
            self.assertEqual(call.call_count,2)

    def test_invalid_proposal_is_archived_and_retry_calls_agent_again(self):
        self.db.set_meta('whitebox_last_video','accepted.mp4')
        sampler=SimpleNamespace(sample_frames=lambda *args,**kwargs:self.frames)
        bad={'references':[{**self.row,'object_id':'unknown'}]}
        with patch.dict('sys.modules',{'perception.frame_sampler':sampler}),patch(
                'whitebox.appearance_rollout.json_call',side_effect=[bad,{'references':[self.row]}]) as call:
            with self.assertRaises(ValueError):capture_accepted(self.db,{},self.root/'capture')
            self.assertTrue((self.root/'capture/proposal_rejected_000.json').is_file())
            self.assertEqual(capture_accepted(self.db,{},self.root/'capture')['status'],'CAPTURED')
            self.assertEqual(call.call_count,2)

    def test_old_identity_anchor_survives_recent_frames(self):
        memory=AppearanceMemory(self.db)
        first=memory.add('hero',self.path,'red',state='identity',accepted=True,timestamp=1)
        memory.add('hero',self.path,'red',state='identity',accepted=True,timestamp=2)
        latest=memory.add('hero',self.path,'red',state='identity',accepted=True,timestamp=3)
        self.assertEqual({r['id'] for r in memory.select('hero',state='identity')},{first['id'],latest['id']})

    def test_scenery_cannot_displace_primary_identity_anchor(self):
        memory=AppearanceMemory(self.db)
        first=memory.add('hero',self.path,'red',state='identity',accepted=True,timestamp=1)
        latest=memory.add('hero',self.path,'red',state='identity',accepted=True,timestamp=3)
        self.scene['entities'].append({'id':'wall','kind':'box'})
        memory.add('wall',self.path,'stone',state='identity',accepted=True,timestamp=3)
        bundle=retrieve(self.db,self.root/'bundle',limit=2)
        self.assertEqual({r['id'] for r in bundle['refs']},{first['id'],latest['id']})

    def test_prompt_consumer_receives_memory_images_and_ids(self):
        self.store();bundle=retrieve(self.db,self.root/'bundle')
        timeline={'frames':[{'entities':self.scene['entities'],'camera':{}}],'fps':24}
        with patch('agent.render_contract.json_call',return_value={'prompt':'Continue <Picture 1>.','appearance_object_ids':['hero']}) as call:
            prompt=compile_prompt({},self.scene,{'actor_id':'hero'},timeline,self.path,self.root,memory=bundle)
        self.assertEqual(call.call_args.args[3],[self.path]+bundle['images'])
        self.assertNotIn('<Picture 2>',prompt)
        selected=json.loads((self.root/'renderer_references.json').read_text())
        self.assertEqual(selected['images'],[])
        self.assertIn('red coat',prompt)
        contract=json.loads((self.root/'render_contract.json').read_text())
        self.assertEqual(contract['appearance_reference_ids'],[bundle['refs'][0]['id']])

    def test_explicit_renderer_selection_renumbers_without_mutating_memory(self):
        self.store();bundle=retrieve(self.db,self.root/'bundle')
        identity=bundle['refs'][0]['id']
        before=json.dumps(bundle,sort_keys=True)
        selected=renderer_references(bundle,[identity])
        self.assertEqual(selected['images'],bundle['images'])
        self.assertIn('<Picture 2>',selected['prompt'])
        self.assertEqual(json.dumps(bundle,sort_keys=True),before)
        for ids in [['absent'],[identity,identity]]:
            with self.assertRaises(ValueError):renderer_references(bundle,ids)
        bundle['refs'][0]['accepted']=False
        with self.assertRaises(ValueError):renderer_references(bundle,[identity])

    def test_compiler_rejects_unselected_picture_tag(self):
        self.store();bundle=retrieve(self.db,self.root/'bundle')
        timeline={'frames':[{'entities':self.scene['entities'],'camera':{}}],'fps':24}
        with patch('agent.render_contract.json_call',return_value={'prompt':'Copy <Picture 2>.','appearance_object_ids':['hero']}):
            with self.assertRaises(ValueError):
                compile_prompt({},self.scene,{'actor_id':'hero'},timeline,self.path,self.root,memory=bundle)

    def test_renderer_subset_has_its_own_picture_order(self):
        self.store();bundle=retrieve(self.db,self.root/'bundle')
        second={**bundle['refs'][0],'id':'second_reference','picture':7}
        bundle['refs'].append(second)
        bundle['images'].append(second['crop_path'])
        timeline={'frames':[{'entities':self.scene['entities'],'camera':{}}],'fps':24}
        with patch('agent.render_contract.json_call',return_value={'prompt':'Identity from <Picture 2>.','appearance_object_ids':['hero']}):
            prompt=compile_prompt({},self.scene,{'actor_id':'hero'},timeline,self.path,self.root,
                                  memory=bundle,renderer_reference_ids=['second_reference'])
        selected=json.loads((self.root/'renderer_references.json').read_text())
        self.assertEqual(selected['reference_ids'],['second_reference'])
        self.assertEqual(len(selected['images']),1)
        self.assertIn('<Picture 2>',prompt)
        self.assertNotIn('<Picture 7>',prompt)

    def test_selected_description_cannot_add_unbound_picture_tag(self):
        self.store();bundle=retrieve(self.db,self.root/'bundle')
        bundle['refs'][0]['prompt']='Identity shown in <Picture 9>.'
        timeline={'frames':[{'entities':self.scene['entities'],'camera':{}}],'fps':24}
        with patch('agent.render_contract.json_call',return_value={'prompt':'Continue <Picture 1>.','appearance_object_ids':['hero']}):
            with self.assertRaises(ValueError):
                compile_prompt({},self.scene,{'actor_id':'hero'},timeline,self.path,self.root,
                               memory=bundle,renderer_reference_ids=[bundle['refs'][0]['id']])

    def test_identity_text_preserves_selected_details_without_adding_hidden_objects(self):
        self.store();bundle=retrieve(self.db,self.root/'bundle')
        bundle['refs'].append({**bundle['refs'][0],'object_id':'hidden','prompt':'blue cloak'})
        text=identity_text(bundle,['hero'])
        self.assertIn('red coat',text)
        self.assertNotIn('blue cloak',text)
        self.assertEqual(identity_text(bundle,[]),'')
        for invalid in [None,['absent'],['hero','hero']]:
            with self.assertRaises(ValueError):identity_text(bundle,invalid)

    def test_omitted_shot_identity_selection_cannot_default_to_all_memory(self):
        self.store();bundle=retrieve(self.db,self.root/'bundle')
        timeline={'frames':[{'entities':self.scene['entities'],'camera':{}}],'fps':24}
        with patch('agent.render_contract.json_call',return_value={'prompt':'Continue <Picture 1>.'}):
            with self.assertRaisesRegex(ValueError,'appearance_object_ids'):
                compile_prompt({},self.scene,{'actor_id':'hero'},timeline,self.path,self.root,memory=bundle)
        self.assertFalse((self.root/'render_prompt.txt').exists())


    def test_requested_landmark_survives_unrelated_offscreen_actors(self):
        memory=AppearanceMemory(self.db)
        for identity,kind in [('hero','actor'),('offscreen','actor'),('gate','box')]:
            if identity!='hero':self.scene['entities'].append({'id':identity,'kind':kind})
            for tick in [1,2]:
                memory.add(identity,self.path,identity,state='identity',accepted=True,timestamp=tick)
        bundle=retrieve(self.db,self.root/'query',limit=2,
                        requests=[{'object_id':'hero'},{'object_id':'gate'}])
        self.assertEqual([r['object_id'] for r in bundle['refs']],['hero','gate'])

    def test_task_view_selects_existing_back_view_and_keeps_old_anchor(self):
        memory=AppearanceMemory(self.db)
        anchor=memory.add('hero',self.path,'front',view='front',state='identity',accepted=True,timestamp=1)
        memory.add('hero',self.path,'front',view='front',state='identity',accepted=True,timestamp=3)
        back=memory.add('hero',self.path,'back',view='back',state='identity',accepted=True,quality=.7,timestamp=2)
        bundle=retrieve(self.db,self.root/'query',limit=2,requests=[{'object_id':'hero','view':'back'}])
        self.assertEqual([r['id'] for r in bundle['refs']],[back['id'],anchor['id']])

    def test_explicit_memory_query_rejects_unknown_duplicate_and_excess_ids(self):
        for requests in [[{'object_id':'missing'}],[{'object_id':'hero'}]*2,
                         [{'object_id':'hero','view':'imagined'}]]:
            with self.assertRaises(ValueError):retrieve(self.db,self.root/'query',requests=requests)
        with self.assertRaises(ValueError):retrieve(self.db,self.root/'query',limit=0,requests=[{'object_id':'hero'}])
        bundle=retrieve(self.db,self.root/'empty',requests=[{'object_id':'hero'}])
        self.assertEqual(bundle['missing_requested_objects'],['hero'])
        self.assertEqual(bundle['images'],[])

    def test_agent_query_is_saved_and_does_not_edit_world_or_renderer(self):
        self.store();before=self.db.snapshot();history=self.db.get_meta('whitebox_history')
        answer={'requests':[{'object_id':'hero','view':'front'}],'reason':'return','missing_evidence':[]}
        with patch('whitebox.appearance_rollout.json_call',return_value=answer):
            bundle=retrieve_for_task(self.db,{},self.root/'task',{'goal':'return'},self.path)
        self.assertEqual(bundle['retrieval_requests'],answer['requests'])
        self.assertEqual(json.loads((self.root/'task/selection.json').read_text()),answer)
        self.assertEqual(renderer_references(bundle)['images'],[])
        self.assertEqual(self.db.snapshot(),before)
        self.assertEqual(self.db.get_meta('whitebox_history'),history)


    def test_view_fallback_is_reported_not_claimed_as_back_evidence(self):
        self.store()
        result=retrieve(self.db,self.root/'back',requests=[{'object_id':'hero','view':'back'}])
        self.assertTrue(result['refs'])
        self.assertEqual(result['missing_requested_views'],[{'object_id':'hero','view':'back'}])
        self.assertIn('identity fallback',result['prompt'])

    def test_view_removed_by_image_budget_is_reported_missing(self):
        self.store()
        AppearanceMemory(self.db).add('hero',self.path,'back',view='back',original=True,state='default')
        result=retrieve(self.db,self.root/'limited',limit=1,requests=[{'object_id':'hero','view':'back'}])
        self.assertEqual(len(result['refs']),1)
        self.assertEqual(result['refs'][0]['view'],'front')
        self.assertEqual(result['missing_requested_views'],[{'object_id':'hero','view':'back'}])

    def test_catalog_and_downstream_keep_absent_landmark_evidence_explicit(self):
        self.store()
        self.scene['entities'].append({'id':'gate','kind':'box'})
        AppearanceMemory(self.db).add('gate',self.path,'gate',accepted=True,state='default',view='back')
        answer={'requests':[],'reason':'No gate evidence','missing_evidence':['Gate back view is unavailable.']}
        with patch('whitebox.appearance_rollout.json_call',return_value=answer) as call:
            result=retrieve_for_task(self.db,{},self.root/'task',{'goal':'return'},self.path)
        gate=next(row for row in call.call_args.args[2]['catalog'] if row['entity']['id']=='gate')
        self.assertEqual(gate['reference_count'],0)
        self.assertEqual(gate['views'],[])
        self.assertIn('Gate back view is unavailable.',result['prompt'])
        self.assertEqual(result['selection_missing_evidence'],answer['missing_evidence'])


    def test_anchor_text_survives_landmark_image_budget_and_newer_conflicting_crop(self):
        memory=AppearanceMemory(self.db)
        anchor=memory.add('hero',self.path,'layered gold armor',state='identity',accepted=True,timestamp=1)
        memory.add('hero',self.path,'plain brown shirt',state='identity',accepted=True,timestamp=2)
        self.scene['entities'].append({'id':'gate','kind':'box'})
        memory.add('gate',self.path,'carved gate',state='identity',accepted=True,timestamp=1)
        bundle=retrieve(self.db,self.root/'query',limit=2,
                        requests=[{'object_id':'hero'},{'object_id':'gate'}])
        self.assertEqual(len(bundle['images']),2)
        self.assertEqual(bundle['refs'][0]['prompt'],'plain brown shirt')
        self.assertEqual(bundle['identity_anchors'][0]['id'],anchor['id'])
        rendered=identity_text(bundle,['hero','gate'])
        self.assertIn('Object hero: layered gold armor',rendered)
        self.assertNotIn('plain brown shirt',rendered)
        self.assertIn('Object gate: carved gate',rendered)

    def test_text_anchor_can_be_selected_without_an_image_slot(self):
        memory=AppearanceMemory(self.db)
        memory.add('hero',self.path,'stable gold armor',state='identity',accepted=True,timestamp=1)
        self.scene['entities'].append({'id':'gate','kind':'box'})
        memory.add('gate',self.path,'carved gate',state='identity',accepted=True,timestamp=1)
        bundle=retrieve(self.db,self.root/'text_only',limit=0,
                        requests=[],text_object_ids=['hero','gate'])
        self.assertEqual(bundle['images'],[])
        self.assertEqual({r['object_id'] for r in bundle['identity_anchors']},{'hero','gate'})
        self.assertIn('Object hero; anchor',bundle['prompt'])
        self.assertIn('Object gate; anchor',bundle['prompt'])
        self.assertIn('Object hero: stable gold armor',identity_text(bundle,['hero']))

    def test_invalid_text_anchor_ids_are_rejected(self):
        for identities in ['hero', [['hero']], ['missing'], ['hero', 'hero']]:
            with self.assertRaises(ValueError):
                retrieve(self.db, self.root/'invalid_text', requests=[], text_object_ids=identities)

    def test_same_text_does_not_merge_distinct_objects(self):
        self.store()
        self.scene['entities'].append({'id':'other','kind':'actor'})
        AppearanceMemory(self.db).add('other',self.path,'red coat',accepted=True,state='identity')
        bundle=retrieve(self.db,self.root/'query')
        rendered=identity_text(bundle,['hero','other'])
        self.assertIn('Object hero: red coat',rendered)
        self.assertIn('Object other: red coat',rendered)
        self.assertNotIn('Object other:',identity_text(bundle,['hero']))


if __name__=='__main__':unittest.main()

