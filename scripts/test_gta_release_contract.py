"""Self-contained metadata mutation tests for the release publication gate.

Only tiny temporary assets are written. Media shape/ffprobe and decoder-cache
reads are mocked; identity, timeline, review and prompt validation run unchanged.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory.gta_conditioning import reference_prompt
from scripts import finalize_gta_release as finalizer


class ReleaseContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='gta_release_contract_')
        self.addCleanup(self.temp.cleanup)
        self.release = Path(self.temp.name)
        self.root = self.release / 'dataset/samples/sample_a'
        (self.root / 'references').mkdir(parents=True)
        self.review_root = self.release / 'visual_review/sample_a'
        self.review_root.mkdir(parents=True)
        self.image = np.full((8, 12, 3), 70, dtype=np.uint8)
        for name in ('rgb.png', 'whitebox.png', 'crop.png'):
            self.assertTrue(cv2.imwrite(str(self.root / 'references' / name), self.image))
        for path in (self.root / 'valid.png', self.root / 'references/mask.png'):
            self.assertTrue(cv2.imwrite(str(path), np.full((8, 12), 255, dtype=np.uint8)))
        np.save(self.root / 'references/embedding.npy', np.ones(384, dtype=np.float32))
        self.frames = []
        self.cameras = {}
        self.worlds = {}
        for index in range(121):
            seq = index + 100
            tick = index + 200
            self.frames.append({
                'proxy_frame_index': index, 'server_time_estimate_ns': 1000 + index * 10,
                'server_clock_domain': 'server_clock_a', 'client_clock_domain': 'client_clock_a',
                'camera_interpolation': {'lower_client_sample_seq': seq, 'upper_client_sample_seq': seq},
                'world_capture_tick': tick,
            })
            self.cameras[str(seq)] = {
                'episode_id': 'episode_a', 'agent_id': 'agent_000',
                'server_clock_domain': 'server_clock_a', 'client_clock_domain': 'client_clock_a',
                'client_sample_seq': seq,
            }
            self.worlds[str(tick)] = {
                'episode_id': 'episode_a', 'server_clock_domain': 'server_clock_a', 'sample_seq': tick,
            }
        self.edges = []
        self.selected = []
        for number in (1, 2):
            life = {'episode_id': 'episode_a', 'dataset_entity_id': number,
                    'model_hash': 42, 'spawn_sample_seq': 0, 'despawn_sample_seq': 9999}
            edge = {
                'object_id': f'episode_a:entity:{number}', 'dataset_entity_id': str(number),
                'category': 'agent', 'reference_id': 'sample_a:ref0',
                'reference_bbox_xyxy': [1, 1, 6, 7],
                'target_projected_bboxes': [[1, 1, 6, 7] for _ in range(120)],
                'lifecycle': life, 'observed_at_ns': 1000, 'used_at_ns': 1010, 'used_until_ns': 2200,
                'reference_rgb': 'references/rgb.png', 'reference_whitebox': 'references/whitebox.png',
                'reference_crop': 'references/crop.png', 'reference_mask': 'references/mask.png',
                'embedding': 'references/embedding.npy',
            }
            self.edges.append(edge)
            self.selected.append({'label': f'O{number}', 'entity_id': str(number), 'category': 'agent',
                                  'lifecycle': copy.deepcopy(life),
                                  'target_bboxes': copy.deepcopy(edge['target_projected_bboxes'])})
        self.sample = {
            'sample_id': 'sample_a', 'episode_id': 'episode_a', 'agent_id': 'agent_000',
            'source_segment': 'segment_00000', 'frame_count': 120, 'fps': 24, 'training_ready': True,
            'conditions': {'prompt': 'prompt.txt', 'target_whitebox': 'whitebox.mp4'},
            'supervision': {'target_rgb': 'rgb.mp4', 'valid_pixels': 'valid.png'},
            'frame_state_index': 'frames.json', 'camera_records': 'cameras.json',
            'world_records': 'worlds.json', 'review': 'review.json', 'object_references': self.edges,
        }
        self.review = {'status': 'REVIEWED', 'pair_acceptable': True,
                       'objects': [{'label': 'O1', 'reference_acceptable': True},
                                   {'label': 'O2', 'reference_acceptable': True}]}
        for mock in (
            patch.object(finalizer, 'video_shape', side_effect=self.mock_video_shape),
            patch.object(finalizer.subprocess, 'run', return_value=SimpleNamespace(
                stdout=json.dumps({'streams': [{'duration': '5.0', 'avg_frame_rate': '24/1'}]}))),
            patch('memory.gta_release_media.cached_frame', side_effect=lambda *args: self.image.copy()),
        ):
            mock.start()
            self.addCleanup(mock.stop)

    def mock_video_shape(self, path, endpoints=None, fps=None):
        if endpoints is not None:
            endpoints.extend([self.image.copy(), self.image.copy()])
        return (120, 8, 12, 3)

    def write_fixture(self, prompt=None, source_review=None):
        # Rebuild the canonical prompt after identity changes so rejection must
        # come from the identity gate, not stale prompt text.
        documents = {
            self.root / 'sample.json': self.sample,
            self.root / 'frames.json': self.frames,
            self.root / 'cameras.json': self.cameras,
            self.root / 'worlds.json': self.worlds,
            self.root / 'review.json': self.review,
            self.review_root / 'decision.json': self.review if source_review is None else source_review,
            self.review_root / 'selected_objects.json': self.selected,
        }
        for path, value in documents.items():
            path.write_text(json.dumps(value), encoding='utf-8')
        (self.root / 'prompt.txt').write_text(
            reference_prompt(self.edges) if prompt is None else prompt, encoding='utf-8')

    def validate(self):
        return finalizer.validate_release_sample(self.root / 'sample.json', self.release)

    def test_valid_baseline_passes_full_gate(self):
        self.write_fixture()
        self.assertEqual(len(self.validate()['object_references']), 2)

    def test_world_composite_key_mutations_rejected(self):
        record = self.worlds['200']
        for field, value, message in (
            ('sample_seq', -1, 'World composite key mismatch'),
            ('server_clock_domain', 'wrong_clock', 'World composite key mismatch'),
            ('episode_id', 'wrong_episode', 'Wrong world state'),
        ):
            with self.subTest(field=field):
                original = record[field]
                record[field] = value
                self.write_fixture()
                with self.assertRaisesRegex(ValueError, message): self.validate()
                record[field] = original

    def test_camera_composite_key_mutations_rejected(self):
        record = self.cameras['100']
        for field, value in (('agent_id', 'agent_999'), ('client_sample_seq', -1),
                             ('client_clock_domain', 'wrong_clock')):
            with self.subTest(field=field):
                original = record[field]
                record[field] = value
                self.write_fixture()
                with self.assertRaisesRegex(ValueError, 'Camera composite key mismatch'): self.validate()
                record[field] = original

    def test_edge_namespace_and_lifecycle_mutations_rejected(self):
        edge = self.edges[0]
        for record, field, value, message in (
            (edge, 'object_id', 'wrong_episode:entity:1', 'Wrong object identity namespace'),
            (edge, 'dataset_entity_id', '999999', 'Wrong object identity namespace'),
            (edge['lifecycle'], 'dataset_entity_id', 999999, 'Wrong object lifecycle'),
            (edge['lifecycle'], 'episode_id', 'wrong_episode', 'Wrong object lifecycle'),
        ):
            with self.subTest(field=field, message=message):
                original = record[field]
                record[field] = value
                self.write_fixture()
                with self.assertRaisesRegex(ValueError, message): self.validate()
                record[field] = original

    def test_consistently_forged_entity_still_requires_approved_candidate(self):
        edge = self.edges[0]
        edge['object_id'] = 'episode_a:entity:999999'
        edge['dataset_entity_id'] = '999999'
        edge['lifecycle']['dataset_entity_id'] = 999999
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, 'Object has no approved visual identity reference'):
            self.validate()

    def test_each_object_requires_approval_even_when_pair_is_accepted(self):
        # Change both review copies: review equality must not hide the missing
        # per-object approval. O1 remains approved while exported O2 is rejected.
        self.review['objects'][1]['reference_acceptable'] = False
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, 'Object has no approved visual identity reference'):
            self.validate()

    def test_qualified_object_evidence_mutations_rejected(self):
        baseline = copy.deepcopy(self.edges[0])
        for field in ('category', 'lifecycle', 'target_projected_bboxes'):
            with self.subTest(field=field):
                self.edges[0] = copy.deepcopy(baseline)
                if field == 'category': self.edges[0][field] = 'vehicle'
                elif field == 'lifecycle': self.edges[0][field]['model_hash'] = 999
                else: self.edges[0][field][50] = [2, 2, 7, 7]
                self.write_fixture()
                with self.assertRaisesRegex(ValueError, 'Object association disagrees with qualified source evidence'):
                    self.validate()

    def test_duplicate_identity_rejected(self):
        self.edges.append(copy.deepcopy(self.edges[0]))
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, 'Duplicate reference identity'): self.validate()

    def test_changed_source_review_rejected(self):
        changed = copy.deepcopy(self.review)
        changed['objects'][0]['reference_acceptable'] = False
        self.write_fixture(source_review=changed)
        with self.assertRaisesRegex(ValueError, 'Review changed after export'): self.validate()

    def test_qa_annotations_do_not_change_canonical_prompt(self):
        original = reference_prompt(self.edges)
        for edge in self.edges:
            edge.update(target_action='QA_ACTION_SENTINEL', appearance='QA_EXTERIOR_SENTINEL',
                        state_change='QA_STATE_SENTINEL', state_change_review='QA_REVIEW_SENTINEL')
        self.review.update(target_action='QA_ACTION_SENTINEL', appearance='QA_EXTERIOR_SENTINEL')
        self.assertEqual(reference_prompt(self.edges), original)
        self.write_fixture()
        self.validate()

    def test_qa_prompt_injection_rejected(self):
        for text in ('Target action: QA_ACTION_SENTINEL', 'Exterior: QA_EXTERIOR_SENTINEL',
                     'Target state: QA_STATE_SENTINEL'):
            with self.subTest(text=text):
                self.write_fixture(prompt=reference_prompt(self.edges) + '\n' + text)
                with self.assertRaisesRegex(ValueError, 'Condition prompt contains unapproved annotation text'):
                    self.validate()


if __name__ == '__main__':
    unittest.main()
