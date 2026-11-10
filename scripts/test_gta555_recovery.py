import unittest
from pathlib import Path
from recover_gta555_source import destination_for


class RecoverySelectionTests(unittest.TestCase):
    def test_preserves_raw_camera_and_world(self):
        for path in ('0916/.spool/world_state.jsonl',
                     '0916/.spool/agents/agent_000/client.jsonl',
                     'worlds/gta5_scene_v3_20260908/collision/meshes/a.tri.gz'):
            self.assertEqual(destination_for('GTA_555/' + path), Path('data') / path)

    def test_keeps_frame_alignment(self):
        path = '0916/derived/segments/agent_000/segment_00000/frames.jsonl'
        self.assertEqual(destination_for('GTA_555/' + path), Path('data') / path)

    def test_omits_only_regenerable_derived_media(self):
        for name in ('proxy.mp4', 'scene_mesh.json', 'preview.png', 'render.log'):
            self.assertIsNone(destination_for('GTA_555/0916/derived/segments/' + name))
        self.assertIsNotNone(destination_for('GTA_555/0916/agents/agent_000/video_00000.mkv'))

    def test_does_not_replace_portable_metadata(self):
        for name in ('global_index.sqlite3', '_delivery/COMPLETE.json', 'evidence/jobs.zip'):
            self.assertIsNone(destination_for('GTA_555/' + name))


if __name__ == '__main__':
    unittest.main()
