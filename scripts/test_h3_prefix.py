"""Local prefix contracts; no Comfy installation or model loading required."""
from argparse import Namespace
from fractions import Fraction
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from renderer.comfy_h3 import ComfyH3Renderer
from scripts.run_comfy_control import add_start_guide, main, read_prefix


class PrefixTests(unittest.TestCase):
    def decode(self, count, total=73, rate=Fraction(24), timestamps=None):
        frames = []
        for i in range(count):
            frame = Mock(pts=i + 48, time_base=Fraction(1, 24))
            if timestamps is not None:
                frame.pts = timestamps[i]
            frame.to_image.return_value = Image.new("RGB", (16, 8), (i, 0, 0))
            frames.append(frame)
        container = Mock()
        container.streams.video = [Namespace(average_rate=rate)]
        container.decode.return_value = iter(frames)
        opener = Mock()
        opener.return_value.__enter__ = Mock(return_value=container)
        opener.return_value.__exit__ = Mock(return_value=False)
        with patch.dict(sys.modules, {"av": Namespace(open=opener)}):
            return read_prefix("prefix.mp4", total, 32, 64)

    def test_valid_batches_preserve_all_frames_and_resize(self):
        for count in (5, 22, 39, 56):
            with self.subTest(count=count):
                pixels = self.decode(count)
                self.assertEqual(pixels.shape, (count, 64, 32, 3))
                self.assertEqual(pixels.dtype, np.uint8)
                self.assertEqual(pixels[:, 0, 0, 0].tolist(), list(range(count)))

    def test_invalid_counts_are_not_cropped(self):
        for count in (0, 1, 4, 6, 21, 23, 73, 74):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "frame count"):
                self.decode(count)

    def test_rate_and_timestamps(self):
        for rate in (None, Fraction(24000, 1001), Fraction(25)):
            with self.subTest(rate=rate), self.assertRaisesRegex(ValueError, "24 fps"):
                self.decode(5, rate=rate)
        for stamps in ([0, 1, 3, 4, 5], [0, 1, 1, 3, 4], [0, 1, None, 3, 4]):
            with self.subTest(stamps=stamps), self.assertRaisesRegex(ValueError, "timestamps"):
                self.decode(5, timestamps=stamps)

    def test_prefix_replaces_single_anchor(self):
        for anchor in (False, True):
            node = Mock()
            node.execute.return_value.result = ["guided"]
            prefix = np.zeros((5, 32, 32, 3))
            self.assertEqual(add_start_guide(node, "positive", "latent", "vae",
                                            "reference", anchor, prefix), "guided")
            node.execute.assert_called_once()
            self.assertEqual(node.execute.call_args.args, ("positive", "latent", 0))
            self.assertIs(node.execute.call_args.kwargs["image"], prefix)

    def test_legacy_anchor_behavior(self):
        node = Mock()
        node.execute.return_value.result = ["guided"]
        self.assertEqual(add_start_guide(node, "positive", "latent", "vae", "ref", False), "positive")
        node.execute.assert_not_called()
        add_start_guide(node, "positive", "latent", "vae", "ref", True)
        node.execute.assert_called_once_with("positive", "latent", 0, vae="vae", image="ref")

    def test_validation_precedes_comfy_and_output_creation(self):
        args = Namespace(frames=73, width=32, height=32, steps=1, source=None,
                         reference_image="ref.png", prefix_video="bad.mp4")
        with patch("scripts.run_comfy_control.read_prefix", side_effect=ValueError("bad prefix")):
            with self.assertRaisesRegex(ValueError, "bad prefix"):
                main(args)

    def test_old_namespace_needs_no_prefix_attribute(self):
        args = Namespace(frames=73, width=32, height=32, steps=1, source=None,
                         reference_image="ref.png", output=Mock())
        args.output.resolve.side_effect = RuntimeError("passed prefix validation")
        with patch("scripts.run_comfy_control.read_prefix") as decode:
            with self.assertRaisesRegex(RuntimeError, "passed prefix validation"):
                main(args)
            decode.assert_not_called()

    def test_renderer_forwards_prefix_and_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            def run(command, **kwargs):
                self.assertEqual(command[command.index("--prefix-video") + 1], str(Path("prefix.mp4").resolve()))
                self.assertEqual(command[command.index("--reference-image") + 1], str(Path("ref.png").resolve()))
                (Path(tmp) / "controlled.mp4").write_bytes(b"fixture")
                return subprocess.CompletedProcess(command, 0)
            with patch("renderer.comfy_h3.subprocess.run", side_effect=run):
                ComfyH3Renderer().render("test", None, tmp, first_frame="ref.png",
                                         prefix_video="prefix.mp4", anchor_first=True)


if __name__ == "__main__":
    unittest.main()
