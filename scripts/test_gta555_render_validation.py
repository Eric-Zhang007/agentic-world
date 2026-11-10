"""Exercise real video decoding and rejection of incomplete render outputs."""
import json
from pathlib import Path
import tempfile
import unittest
import av
import numpy as np
from run_gta555_full_rebuild import verify_render, load_config


class RenderValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        folder = self.root / 'render/agent_000'
        folder.mkdir(parents=True)
        with av.open(str(folder / 'proxy.mp4'), 'w') as container:
            stream = container.add_stream('mpeg4', rate=24)
            stream.width = stream.height = 32
            stream.pix_fmt = 'yuv420p'
            for shade in (40, 120, 200):
                frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), shade, dtype=np.uint8), format='rgb24')
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        (folder / 'frames.jsonl').write_text('{}\n{}\n{}\n')
        self.manifest = {'complete': True, 'agents': {'agent_000': {
            'frames': 3, 'statuses': {'estimated_alignment': 2},
            'video': 'agent_000/proxy.mp4', 'frame_index': 'agent_000/frames.jsonl'}}}
        self.write_manifest()
        self.job = {'agent': 'agent_000', 'expected_frames': 3, 'usable_frames': 2}

    def write_manifest(self):
        (self.root / 'render/manifest.json').write_text(json.dumps(self.manifest))

    def tearDown(self):
        self.temp.cleanup()

    def test_real_decode(self):
        self.assertEqual(verify_render(self.root, self.job)['decoded_frames'], 3)

    def test_rejects_missing_metadata_frame(self):
        (self.root / 'render/agent_000/frames.jsonl').write_text('{}\n{}\n')
        with self.assertRaisesRegex(ValueError, 'Frame count mismatch'):
            verify_render(self.root, self.job)

    def test_rejects_changed_camera_coverage(self):
        self.job['usable_frames'] = 3
        with self.assertRaisesRegex(ValueError, 'Usable frame mismatch'):
            verify_render(self.root, self.job)

    def test_rejects_incomplete_render(self):
        self.manifest['complete'] = False
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'did not complete'):
            verify_render(self.root, self.job)

    def test_nested_config_is_not_ignored(self):
        path = self.root / 'config.yaml'
        path.write_text('workspace: /tmp/a\nsource_root: /tmp/b\noutputs: /tmp/c\nrender:\n  workers: 4\n')
        self.assertEqual(load_config(path)['render']['workers'], 4)


if __name__ == '__main__':
    unittest.main()
