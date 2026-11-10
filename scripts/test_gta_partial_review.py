import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.review_gta_passage_targets import main


class PartialReviewTests(unittest.TestCase):
    def test_partial_reviews_preserve_unseen_samples_and_previous_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = root / 'audit'
            batch.mkdir()
            (batch / 'index.json').write_text(json.dumps(['a', 'b', 'c']))
            (batch / 'layout.json').write_text('{"historical_reference_pair":true}')
            for key in ('a', 'b', 'c'):
                folder = root / 'training/samples' / key
                folder.mkdir(parents=True)
                (folder / 'sample.json').write_text(json.dumps({'sample_id': key, 'training_status': 'VIDEO_QA_PENDING'}))
            decisions = batch / 'review.json'
            def apply(indices, rejected):
                decisions.write_text(json.dumps({'reviewed_count': len(indices), 'reviewed_indices': indices,
                                                 'all_sheets_inspected': True, 'rejected': rejected}))
                with patch('sys.argv', ['review', '--root', str(root), '--batch', 'audit', '--decisions', str(decisions)]):
                    main()
            apply([0], {})
            apply([1], {'1': 'Invalid geometry'})
            self.assertEqual(json.loads((root / 'training/samples/c/sample.json').read_text())['training_status'], 'VIDEO_QA_PENDING')
            self.assertEqual(len((batch / 'decisions.jsonl').read_text().splitlines()), 2)
            with self.assertRaises(ValueError):
                apply([2, 2], {})
            with self.assertRaises(ValueError):
                apply([2], {'0': 'Not inspected in this submission'})


if __name__ == '__main__':
    unittest.main()
