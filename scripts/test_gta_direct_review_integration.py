import json
from pathlib import Path
import tempfile
import unittest
from collections import Counter

from memory.gta_mixed_dataset import balanced_round
from scripts.assemble_gta_audited_mixture import resolve_review


class ReviewIntegrationTests(unittest.TestCase):
    def test_direct_rejection_overrides_old_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';direct=root/'direct';old.mkdir();direct.mkdir()
            sample=root/'sample.json'
            review=dict(sample_id='x',sample_path=str(sample),alignment_ok=True,
                background_memory_useful=True,no_major_occlusion_conflict=True,
                no_ui_contamination=True,uncertain=False,reason='Visible road',
                background_evidence='Shared curve',model='prior_model')
            (old/'x.json').write_text(json.dumps(review))
            changed={**review,'no_ui_contamination':False,'method':'Direct Codex visual inspection'}
            (direct/'x.json').write_text(json.dumps(changed))
            path,result=resolve_review('x',sample,old,direct)
            self.assertEqual(path,direct/'x.json')
            self.assertFalse(result['accepted'])
            self.assertEqual(result['review_origin'],'direct_local_visual')
            _,prior=resolve_review('x',sample,old)
            self.assertTrue(prior['accepted'])
            self.assertEqual(prior['review_origin'],'prior_api_visual')
            self.assertEqual(resolve_review('missing',sample,old,direct),(None,None))

    def test_scene_subtypes_share_site_cap(self):
        rows=[]
        for task in ('scene_passage','scene_nearby','object_person'):
            for site in range(4):
                for i in range(10):
                    rows.append(dict(task=task,dataset='test',sample_id=f'{task}_{site}_{i}',balance_group=str(site)))
        for seed in range(10):
            result=balanced_round(rows,{'scene_passage':.25,'scene_nearby':.25,'object_person':.5},seed)
            sites=Counter(r['balance_group'] for r in result if r['task'].startswith('scene_'))
            self.assertLessEqual(max(sites.values()),2)
            self.assertEqual(Counter(r['task'] for r in result),{'scene_passage':4,'scene_nearby':4,'object_person':8})
            self.assertEqual(len(result),len({r['sample_id'] for r in result}))


if __name__=='__main__':unittest.main()
