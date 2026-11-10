import unittest
from scripts.audit_gta_scene_memory import validate_results


class AuditTests(unittest.TestCase):
    def row(self):
        return dict(sample_id='x',alignment_ok=True,background_memory_useful=True,
                    no_major_occlusion_conflict=True,no_ui_contamination=True,uncertain=False,
                    background_evidence='Shared facade',reason='Facade and street geometry agree')

    def test_uncertainty_cannot_pass(self):
        row=self.row();row['uncertain']=True
        self.assertFalse(validate_results({'reviews':[row]},['x'])[0]['accepted'])

    def test_overlap_required(self):
        row=self.row();row['background_memory_useful']=False
        self.assertFalse(validate_results({'reviews':[row]},['x'])[0]['accepted'])

    def test_wrong_ids_rejected(self):
        with self.assertRaises(ValueError):validate_results({'reviews':[self.row()]},['y'])

    def test_string_boolean_rejected(self):
        row=self.row();row['alignment_ok']='true'
        with self.assertRaises(ValueError):validate_results({'reviews':[row]},['x'])


if __name__=='__main__':unittest.main()
