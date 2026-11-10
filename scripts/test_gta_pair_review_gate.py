import unittest
from scripts.finalize_gta_passage_training import require_pair_review, merge_unique_rows


class PairReviewTest(unittest.TestCase):
    def test_current_scene_replaces_duplicate(self):
        old = dict(dataset='previous_scenes', sample_id='x', episode_id='e', split='train', task='scene_background')
        new = {**old, 'dataset': 'scenes'}
        rows, count = merge_unique_rows([old, new])
        self.assertEqual(rows, [new])
        self.assertEqual(count, 1)

    def test_episode_split_conflict(self):
        old = dict(dataset='objects', sample_id='x', episode_id='e', split='train', task='object_person')
        new = {**old, 'sample_id': 'y', 'split': 'test'}
        with self.assertRaises(ValueError):
            merge_unique_rows([old, new])

    def test_old_target_review_is_not_new_reference_review(self):
        with self.assertRaises(ValueError):
            require_pair_review({'sample_id': 'x', 'training_status': 'AUTO_REFERENCE_WITH_INHERITED_TARGET_QA'})

    def test_missing_reference_review_is_rejected(self):
        with self.assertRaises(ValueError):
            require_pair_review({'sample_id': 'x', 'training_status': 'AUTO_REFERENCE_WITH_REVIEWED_TARGET_QA',
                                 'supplemental_target_review': {'accepted': True}})

    def test_explicit_pair_review(self):
        require_pair_review({'sample_id': 'x', 'training_status': 'AUTO_REFERENCE_WITH_REVIEWED_TARGET_QA',
                            'supplemental_target_review': {'accepted': True, 'historical_reference_pair_reviewed': True}})


if __name__ == '__main__':
    unittest.main()
