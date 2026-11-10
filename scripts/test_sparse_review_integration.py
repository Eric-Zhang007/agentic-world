from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts import align_sparse_observations as runner
from scripts import test_observation_alignment_attachments as attachments
from whitebox.image_alignment import project_points
from whitebox.motion import validate_geometry


class SparseReviewIntegrationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.scene, observation = attachments.AttachmentTests().setup_scene()
        frames = []
        for index in range(2):
            bundle = self.root / ('bundle_%d' % index)
            bundle.mkdir()
            image = Image.new('RGB', (800, 600), (30 + index, 60, 90))
            image.save(bundle / 'input.png')
            mask = Image.new('L', image.size)
            mask.paste(255, (300, 200, 500, 400))
            mask.save(bundle / 'mask.png')
            perception = bundle / 'result.json'
            perception.write_text(json.dumps({'observations': [
                {'observation_id': 'wall_mask', 'mask_path': 'mask.png'}]}))
            frame = deepcopy(observation)
            frame.update(timestamp_sec=4 + index, image=str(bundle / 'input.png'),
                         perception=str(perception), mapping={'wall_mask': 'wall'})
            if index:
                pixels, _ = project_points([[3, 0, -1]], self.scene['camera'], 800, 600)
                frame['actor_roots'][0]['pixel'] = pixels[0].tolist()
            frames.append(frame)
        self.annotations = {'provenance': 'synthetic integration fixture', 'frames': frames}

    def test_sequence_reviews_candidates_without_committing(self):
        original = deepcopy((self.scene, self.annotations))
        input_bytes = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        output = self.root / 'review'
        starts = []
        align = runner.align_observation

        def record_start(scene, observation):
            starts.append(deepcopy(scene))
            return align(scene, observation)

        with patch.object(runner, 'align_observation', side_effect=record_start):
            reports = runner.run_sequence(self.scene, self.annotations, output)
        self.assertEqual(len(reports), 2)
        self.assertEqual(json.loads((output / 'report.json').read_text()), reports)
        self.assertEqual((self.scene, self.annotations), original)
        self.assertEqual(self.annotations['frames'][0]['static_landmarks'],
                         self.annotations['frames'][1]['static_landmarks'])
        previous = self.scene
        for index, report in enumerate(reports):
            step = output / ('step_%02d' % (index + 1))
            candidate = json.loads((step / 'scene.json').read_text())
            self.assertEqual(starts[index], previous)
            self.assertEqual(json.loads((step / 'scene_before.json').read_text()), previous)
            self.assertEqual(json.loads((step / 'report.json').read_text()), report)
            ids = {e['id']: e for e in candidate['entities']}
            np.testing.assert_allclose(ids['player']['position'], [2 + index, 0, -1], atol=1e-7)
            for entity in self.scene['entities']:
                if entity['id'] not in ('player', 'staff'):
                    self.assertEqual(ids[entity['id']], entity)
            visual = json.loads((step / 'visual_report.json').read_text())
            self.assertFalse(report['world_committed'])
            self.assertFalse(visual['world_committed'])
            self.assertEqual(report['visual_review'], {
                'counts': visual['counts'], 'geometry_errors': validate_geometry(candidate),
                'source_perception': self.annotations['frames'][index]['perception'],
                'perception_invoked': False})
            self.assertEqual(visual['geometry_errors'], validate_geometry(candidate))
            self.assertEqual(visual['counts']['mapped_observations'], 1)
            for name in ('overlay.jpg', 'visual_overlay.jpg'):
                with Image.open(step / name) as image:
                    self.assertEqual(image.size, (800, 600))
                    image.load()
            previous = candidate
        for path, content in input_bytes.items():
            self.assertEqual(path.read_bytes(), content)

    def test_missing_counterpart_rejected_before_any_alignment(self):
        for missing in ('perception', 'mapping'):
            with self.subTest(missing=missing):
                annotations = deepcopy(self.annotations)
                del annotations['frames'][1][missing]
                original = deepcopy((self.scene, annotations))
                output = self.root / ('missing_' + missing)
                with patch.object(runner, 'align_observation') as align:
                    with self.assertRaisesRegex(ValueError, 'together'):
                        runner.run_sequence(self.scene, annotations, output)
                    align.assert_not_called()
                self.assertFalse(output.exists())
                self.assertEqual((self.scene, annotations), original)

    def test_same_size_wrong_source_frame_rejected(self):
        annotations=deepcopy(self.annotations)
        annotations['frames'][0]['perception']=annotations['frames'][1]['perception']
        with self.assertRaisesRegex(ValueError, 'different source frame'):
            runner.run_sequence(self.scene,annotations,self.root/'wrong_frame')


if __name__ == '__main__':
    unittest.main()
