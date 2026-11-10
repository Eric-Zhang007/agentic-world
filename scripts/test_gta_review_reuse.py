import json
from pathlib import Path
import tempfile
import unittest

from scripts.reuse_gta_passage_reviews import MEDIA, reuse


class ReviewReuseTests(unittest.TestCase):
    def test_exact_media_only_and_preserve_other_decisions(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp) / 'source', Path(temp) / 'target'
            source.mkdir()
            target.mkdir()
            sample = {'training_status': 'VIDEO_QA_PENDING', 'reference': {
                'reference_point': 'ref', 'target_point': 'target'}, 'reference_time_ns': 1,
                'target_time_ns': 2, 'frame_count': 120, 'fps': 24}
            def reset():
                (target / 'sample.json').write_text(json.dumps(sample))
            (source / 'sample.json').write_text(json.dumps({**sample, 'supplemental_target_review': {
                'accepted': False, 'historical_reference_pair_reviewed': True, 'reason': 'Occlusion'}}))
            for name in MEDIA:
                (source / name).write_bytes(b'original')
                (target / name).write_bytes(b'original')
            reset()
            self.assertEqual(reuse(source, target), 'excluded')
            self.assertEqual(reuse(source, target), 'preserved_existing_status')
            reset()
            (target / MEDIA[-1]).write_bytes(b'different')
            self.assertEqual(reuse(source, target), 'different_media')
            self.assertEqual(json.loads((target / 'sample.json').read_text())['training_status'], 'VIDEO_QA_PENDING')
            (target / MEDIA[-1]).write_bytes(b'original')
            sample['reference']['reference_point'] = 'new-ref'
            reset()
            self.assertEqual(reuse(source, target), 'different_pair')


if __name__ == '__main__':
    unittest.main()
