"""Prevent projection changes caused by stretching depth after rendering."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.run_video_reconcile_trial import read_depth_frames, render_proxy


class NativeDepthCanvasTests(unittest.TestCase):
    def test_wrong_canvas_is_rejected_instead_of_stretched(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'depth.png'
            Image.new('RGB', (960, 640)).save(path)
            with self.assertRaisesRegex(ValueError, 'not stretched'):
                read_depth_frames([path])

    def test_native_pixels_and_chronological_order_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp)/f'{i}.png' for i in range(2)]
            for value, path in zip((23, 178), paths):
                Image.new('RGB', (1024, 768), (value,)*3).save(path)
            frames = read_depth_frames(paths)
            self.assertEqual(frames.shape, (2, 768, 1024, 3))
            self.assertEqual(frames.dtype, np.uint8)
            self.assertTrue(np.all(frames[0] == 23))
            self.assertTrue(np.all(frames[1] == 178))

    def test_target_canvas_reaches_godot(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('scripts.run_video_reconcile_trial.subprocess.run') as run:
                render_proxy(Path(tmp), Path(tmp)/'depth', 'depth', width=1024, height=768)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index('--width')+1], '1024')
            self.assertEqual(command[command.index('--height')+1], '768')


if __name__ == '__main__':
    unittest.main()
