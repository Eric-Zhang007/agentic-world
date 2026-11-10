import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from renderer.comfy_h3 import ComfyH3Renderer
from scripts.run_comfy_control import main, encode_references
from argparse import Namespace


class RendererContractTests(unittest.TestCase):
    def test_native_video_and_image_reach_same_node(self):
        from unittest.mock import Mock
        node = Mock()
        node.execute.return_value.result = ('conditioning', 'latent')
        images = {'ref_image_1': object()}
        video = object()
        self.assertEqual(encode_references(node, 'clip', 'prompt', 1024, 768, 124,
                                          'vae', images, video), ('conditioning', 'latent'))
        self.assertIs(node.execute.call_args.kwargs['ref_videos']['ref_video_1'], video)
        self.assertIs(node.execute.call_args.kwargs['ref_images'], images)

    def test_native_video_forwarded_separately_from_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            def fake_run(command, **kwargs):
                self.assertIn('--reference-video', command)
                self.assertNotIn('--source', command)
                (Path(tmp)/'controlled.mp4').write_bytes(b'fixture')
                return subprocess.CompletedProcess(command, 0)
            with patch('renderer.comfy_h3.subprocess.run', side_effect=fake_run):
                ComfyH3Renderer().render('test', None, tmp, first_frame='first.png',
                                        reference_video='whitebox.mp4', controls=[])

    def test_appearance_images_forwarded_in_picture_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);images=[root/'hero.png',root/'staff.png']
            for path in images:path.write_bytes(b'transport fixture')
            def fake_run(command,**kwargs):
                actual=[command[i+1] for i,value in enumerate(command) if value=='--appearance-image']
                self.assertEqual(actual,[str(p.resolve()) for p in images])
                (root/'controlled.mp4').write_bytes(b'fixture')
                return subprocess.CompletedProcess(command,0)
            with patch('renderer.comfy_h3.subprocess.run',side_effect=fake_run):
                ComfyH3Renderer().render('test',None,root,first_frame='boundary.png',appearance_images=images)

    def test_appearance_limit_and_missing_asset_fail_before_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp,patch('renderer.comfy_h3.subprocess.run') as run:
            with self.assertRaises(ValueError):
                ComfyH3Renderer().render('test',None,tmp,appearance_images=['x']*9)
            with self.assertRaises(FileNotFoundError):
                ComfyH3Renderer().render('test',None,tmp,appearance_images=[Path(tmp)/'missing.png'])
            run.assert_not_called()

    def test_command_and_success_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            def fake_run(command, **kwargs):
                out = Path(command[command.index("--output") + 1])
                (out / "controlled.mp4").write_bytes(b"mock transport artifact")
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "3")
                self.assertIn("--control-video", command)
                self.assertIn("--reference-image", command)
                return subprocess.CompletedProcess(command, 0)
            with patch("renderer.comfy_h3.subprocess.run", side_effect=fake_run):
                result = ComfyH3Renderer().render(
                    "test", "source.mp4", tmp, first_frame="ref.png", control_video="canny.mp4", controls=[{'strength':1}])
            self.assertEqual(result.name, "controlled.mp4")
            self.assertEqual(json.loads((Path(tmp)/"generation_metadata.json").read_text())["returncode"], 0)

    def test_failure_preserves_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("renderer.comfy_h3.subprocess.run",
                       return_value=subprocess.CompletedProcess([], 7)):
                with self.assertRaises(RuntimeError):
                    ComfyH3Renderer().render("test", "source.mp4", tmp)
            record = json.loads((Path(tmp)/"generation_metadata.json").read_text())
            self.assertEqual(record["returncode"], 7)
            self.assertIn("error", record)

    def test_default_no_control_returns_consistent_case_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            def fake_run(command, **kwargs):
                (Path(tmp)/'controlled.mp4').write_bytes(b'no control artifact')
                return subprocess.CompletedProcess(command, 0)
            with patch('renderer.comfy_h3.subprocess.run', side_effect=fake_run):
                result = ComfyH3Renderer().render('test', None, tmp, strength=0)
            self.assertEqual(result.name, 'controlled.mp4')
            with self.assertRaises(FileExistsError):
                ComfyH3Renderer().render('test', None, tmp, strength=0)

    def test_explicit_zero_strength_case_keeps_controlled_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            def fake_run(command, **kwargs):
                self.assertIn('--cases', command)
                (Path(tmp)/'controlled.mp4').write_bytes(b'explicit case artifact')
                return subprocess.CompletedProcess(command, 0)
            with patch('renderer.comfy_h3.subprocess.run', side_effect=fake_run):
                result = ComfyH3Renderer().render('test', None, tmp, strength=0, controls=[])
            self.assertEqual(result.name, 'controlled.mp4')

    def test_explicit_terminal_guide_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            def fake_run(command,**kwargs):
                self.assertIn('--last-frame',command)
                self.assertIn('--anchor-first',command)
                out=Path(command[command.index('--output')+1]);(out/'controlled.mp4').write_bytes(b'fixture')
                return subprocess.CompletedProcess(command,0)
            with patch('renderer.comfy_h3.subprocess.run',side_effect=fake_run):
                ComfyH3Renderer().render('boundary test',None,tmp,first_frame='start.png',
                                         last_frame='end.png',anchor_first=True,controls=[])

    def test_timeout_preserves_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("renderer.comfy_h3.subprocess.run",
                       side_effect=subprocess.TimeoutExpired("worker", 1)):
                with self.assertRaises(subprocess.TimeoutExpired):
                    ComfyH3Renderer(timeout=1).render("test", "source.mp4", tmp)
            self.assertIn("error", json.loads((Path(tmp)/"generation_metadata.json").read_text()))

    def test_stale_rollout_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/"controlled.mp4").write_bytes(b"old")
            with self.assertRaises(FileExistsError):
                ComfyH3Renderer().render("test", "source.mp4", tmp)

    def test_alignment_rejected_before_loading_models(self):
        with self.assertRaisesRegex(ValueError, "Frame count"):
            main(Namespace(frames=72))
        with self.assertRaisesRegex(ValueError, "Canvas"):
            main(Namespace(frames=73, width=830, height=480))


if __name__ == "__main__":
    unittest.main()
