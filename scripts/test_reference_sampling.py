import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import av
import numpy as np
from renderer.reference_video import read_reference


def make_video(path, count, fps):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for index in range(count):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), index, np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class ReferenceTests(unittest.TestCase):
    def test_short_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "short.mp4"
            make_video(path, 10, 10)
            with self.assertRaisesRegex(ValueError, "too short"):
                read_reference(path)

    def test_timestamp_resampling(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "three_seconds.mp4"
            make_video(path, 30, 10)
            frames, metadata = read_reference(path, width=64, height=48)
            self.assertEqual(len(frames), 72)
            self.assertEqual(metadata["selected_source_indices"][-1], 29)
            self.assertEqual(metadata["selected_source_indices"][0], 0)
            self.assertEqual(len(set(metadata["selected_source_indices"])), 30)

    def test_native_24fps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.mp4"
            make_video(path, 72, 24)
            _, metadata = read_reference(path, width=64, height=48)
            self.assertEqual(metadata["selected_source_indices"], list(range(72)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
