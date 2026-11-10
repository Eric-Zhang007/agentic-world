"""Isolated orchestration tests: no GPU, model download, database or paid API."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from whitebox.geometry_pipeline import propose


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / 'rgb.png'
        Image.new('RGB', (100, 80), 'red').save(self.image)
        self.reconstruction = self.root / 'cache.npz'
        self.reconstruction.write_bytes(b'compiler handles NPZ')
        self.perception = self.root / 'sam' / 'result.json'
        self.make_perception(self.perception)
        self.args = dict(image=str(self.image), reconstruction=str(self.reconstruction),
                         perception=str(self.perception), output=str(self.root / 'new'))
        self.commands = []

    def make_perception(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.open(self.image).save(path.parent / 'input.png')
        path.write_text(json.dumps({'observations': [{'category': 'road.'}]}))

    def fake_run(self, command, **kwargs):
        self.commands.append(command)
        self.assertEqual(kwargs['env']['HF_HUB_OFFLINE'], '1')
        self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], str(self.args.get('gpu', '2')))
        self.assertEqual(kwargs['env']['CUDA_DEVICE_ORDER'], 'PCI_BUS_ID')
        self.assertNotIn('shell', kwargs)
        output = Path(command[command.index('--output') + 1])
        if any('reconstruct_vggt.py' in x for x in command):
            self.assertIn('--local-files-only', command)
            output.write_bytes(b'new reconstruction')
        elif any('probe_perception.py' in x for x in command):
            self.make_perception(output / 'result.json')
        elif any('compile_geometry_assets.py' in x for x in command):
            output.mkdir()
            (output / 'scene.json').write_text(json.dumps({
                'camera': {'intrinsics': {'width': 100, 'height': 80}}, 'entities': []}))
            (output / 'report.json').write_text(json.dumps({
                'facade_regions': [{'coverage': .4}], 'scale_source': 'uncalibrated model units'}))
            (output / 'source_candidates.json').write_text('[]')
        else:
            output.mkdir()
            names = ['camera.png', 'overview.png', 'front.png', 'side.png']
            for name in names:
                Image.new('RGB', (100, 80)).save(output / name)
            (output / 'manifest.json').write_text(json.dumps({'images': [{'file': n} for n in names]}))
        return subprocess.CompletedProcess(command, 0)

    def run_mock(self, runner=None):
        with patch('whitebox.geometry_pipeline.subprocess.run', side_effect=runner or self.fake_run):
            return propose(self.args)

    def test_existing_display_forwarded_only_to_preview(self):
        self.args['use_display'] = True
        self.assertTrue(self.run_mock()['success'])
        self.assertIn('--use-display', self.commands[-1])
        self.assertTrue(all('--use-display' not in command for command in self.commands[:-1]))

    def test_cached_success(self):
        result = self.run_mock()
        self.assertTrue(result['success'], result)
        self.assertEqual(result['status'], 'PROPOSAL_READY')
        self.assertFalse(result['accepted'])
        self.assertFalse(result['database_written'])
        self.assertFalse(result['world_merge_ready'])
        self.assertEqual(len(self.commands), 2)
        self.assertEqual(result['stages']['reconstruction']['status'], 'cached')
        self.assertEqual(len(result['preview']['images']), 4)
        self.assertEqual([Path(p).name for p in result['review_images']], ['camera.png', 'overview.png'])
        self.assertTrue(all(Path(p).is_absolute() and Path(p).is_file() for p in result['review_images']))
        self.assertTrue(any('coverage incomplete' in s for s in result['warnings']))
        self.assertTrue(Path(result['evidence']['pipeline_report']).is_file())
        self.assertEqual(result['evidence']['preview_size'], [400, 320])

    def test_fresh_offline_pipeline(self):
        del self.args['reconstruction'], self.args['perception']
        self.args.update(reconstruction_python='vggt-python', perception_python='sam-python')
        result = self.run_mock()
        self.assertTrue(result['success'], result)
        self.assertEqual(len(self.commands), 4)
        self.assertEqual(self.commands[0][0], 'vggt-python')
        self.assertEqual(self.commands[1][0], 'sam-python')

    def test_gpu_selection_does_not_mutate_parent(self):
        self.args['gpu'] = '3'
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}):
            result = self.run_mock()
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '0')
        self.assertTrue(result['success'])
        self.assertEqual(result['stages']['compile']['environment']['CUDA_VISIBLE_DEVICES'], '3')

    def test_rgb_mismatch_even_when_cache_pair_agrees(self):
        Image.new('RGB', (100, 80), 'blue').save(self.image)
        result = self.run_mock()
        self.assertFalse(result['success'])
        self.assertEqual(result['error']['stage'], 'rgb_and_masks')
        self.assertEqual(self.commands, [])

    def test_missing_ground_no_retry_or_fallback(self):
        self.perception.write_text('{"observations": [{"category":"building"}]}')
        result = self.run_mock()
        self.assertIn('No road/ground/sidewalk', result['error']['message'])
        self.assertIsNone(result['scene_path'])
        self.assertEqual(self.commands, [])

    def test_mixed_ground_label_not_promoted(self):
        self.perception.write_text('{"observations": [{"category":"road sidewalk"}]}')
        result = self.run_mock()
        self.assertFalse(result['success'])
        self.assertEqual(result['evidence']['ground_segmentation']['exact_ground_categories'], [])
        self.assertFalse(result['evidence']['ground_segmentation']['exact_road_detected'])
        self.assertEqual(self.commands, [])

    def test_sidewalk_only_warns_road_unestablished(self):
        self.perception.write_text('{"observations": [{"category":"road sidewalk"}, {"category":"sidewalk"}]}')
        result = self.run_mock()
        self.assertTrue(result['success'], result)
        self.assertEqual(result['evidence']['ground_segmentation']['exact_ground_categories'], ['sidewalk'])
        self.assertTrue(any('No exact road' in x for x in result['warnings']))

    def test_compiler_failure_preserves_log(self):
        def fail(command, **kwargs):
            kwargs['stdout'].write('SAM and reconstruction RGB differ')
            return subprocess.CompletedProcess(command, 1)
        result = self.run_mock(fail)
        self.assertEqual(result['error']['stage'], 'compile')
        self.assertIn('RGB differ', result['error']['message'])
        self.assertNotIn('reconstruction_rgb_check', result['evidence'])
        self.assertIsNone(result['scene_path'])

    def test_preview_failure_keeps_candidate_not_success(self):
        def fail(command, **kwargs):
            if 'whitebox.godot.render' in command:
                raise subprocess.TimeoutExpired(command, 5)
            return self.fake_run(command, **kwargs)
        result = self.run_mock(fail)
        self.assertFalse(result['success'])
        self.assertEqual(result['error']['stage'], 'preview')
        self.assertTrue(Path(result['scene_path']).is_file())
        self.assertIsNone(result['preview'])
        self.assertEqual(result['review_images'], [])

    def test_invalid_args_and_existing_output(self):
        for extra in [{'frame': -1}, {'frame': True}, {'scale': float('nan')},
                      {'confidence_quantile': 1}, {'accept': True}, {'preview_scale': 0},
                      {'gpu': '0,2'}, {'device': 'cuda:2'}]:
            with self.subTest(extra=extra):
                result = propose({**self.args, **extra})
                self.assertEqual(result['error']['stage'], 'validation')
        out = Path(self.args['output'])
        out.mkdir()
        sentinel = out / 'proposal.json'
        sentinel.write_text('existing work')
        result = self.run_mock()
        self.assertFalse(result['success'])
        self.assertEqual(sentinel.read_text(), 'existing work')

    def test_cached_frame_forwarded(self):
        self.args['frame'] = 2
        result = self.run_mock()
        self.assertTrue(result['success'])
        command = self.commands[0]
        self.assertEqual(command[command.index('--frame') + 1], '2')

    def test_missing_preview_artifact_is_failure(self):
        def missing(command, **kwargs):
            proc = self.fake_run(command, **kwargs)
            if 'whitebox.godot.render' in command:
                (Path(command[command.index('--output') + 1]) / 'camera.png').unlink()
            return proc
        result = self.run_mock(missing)
        self.assertFalse(result['success'])
        self.assertEqual(result['error']['stage'], 'preview')

    def test_missing_compiler_artifact_is_failure(self):
        def missing(command, **kwargs):
            return subprocess.CompletedProcess(command, 0)
        result = self.run_mock(missing)
        self.assertFalse(result['success'])
        self.assertEqual(result['error']['stage'], 'compile')

    def test_review_images_must_be_in_manifest(self):
        def omit_camera(command, **kwargs):
            proc = self.fake_run(command, **kwargs)
            if 'whitebox.godot.render' in command:
                manifest = Path(command[command.index('--output') + 1]) / 'manifest.json'
                data = json.loads(manifest.read_text())
                data['images'][0]['file'] = 'overview.png'
                manifest.write_text(json.dumps(data))
            return proc
        result = self.run_mock(omit_camera)
        self.assertFalse(result['success'])
        self.assertEqual(result['error']['stage'], 'preview')
        self.assertEqual(result['review_images'], [])


if __name__ == '__main__':
    if '--real' not in sys.argv:
        unittest.main()
    else:
        import argparse

        parser = argparse.ArgumentParser(description='Run real cached compiler and renderer, no model inference')
        parser.add_argument('--real', action='store_true')
        parser.add_argument('--image', required=True)
        parser.add_argument('--reconstruction', required=True)
        parser.add_argument('--perception', required=True)
        parser.add_argument('--output', required=True)
        parser.add_argument('--frame', type=int, default=0)
        parser.add_argument('--plane-threshold', type=float, default=.02)
        parser.add_argument('--actor-height-prior', type=float)
        args = vars(parser.parse_args())
        args.pop('real')
        args = {key: value for key, value in args.items() if value is not None}
        result = propose(args)
        print(json.dumps({key: result.get(key) for key in
                          ('success', 'status', 'scene_path', 'preview', 'timing_seconds', 'error')}, indent=2))
        raise SystemExit(0 if result['success'] else 1)
