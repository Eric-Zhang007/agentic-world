"""Real SQLite/images/video with mocked Godot; no paid API or H3."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory.paired_history import capture, catalog, select
from state.db import WorldDB
from whitebox.world import initialize, scene_from_db
from scripts.test_whitebox_world import fixture


class PairedHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = WorldDB(self.root/'world.sqlite')
        initialize(self.db, fixture())
        self.rgb = self.root/'initial.png'
        Image.new('RGB', (96, 64), (220, 20, 30)).save(self.rgb)
        self.render = patch('memory.paired_history._render', side_effect=self.fake_render).start()

    def tearDown(self):
        patch.stopall()
        self.db.connection.close()
        self.temp.cleanup()

    def fake_render(self, scene, output, size):
        self.assertEqual(json.loads(scene.read_text()), scene_from_db(self.db))
        output.mkdir()
        path = output/'camera.png'
        Image.new('RGB', size, (10, 200, 40)).save(path)
        return path

    def state(self):
        return (scene_from_db(self.db), self.db.get_meta('tick'),
                self.db.get_meta('whitebox_revision'), self.db.get_meta('whitebox_history'))

    def accepted(self):
        video = self.root/'accepted.mkv'
        with av.open(str(video), 'w') as out:
            stream = out.add_stream('ffv1', rate=24)
            stream.width, stream.height, stream.pix_fmt = 96, 64, 'bgr0'
            for color in ((40, 50, 60), (220, 20, 30)):
                frame = av.VideoFrame.from_image(Image.new('RGB', (96, 64), color))
                for packet in stream.encode(frame):
                    out.mux(packet)
            for packet in stream.encode():
                out.mux(packet)
        last = self.root/'last.png'
        with av.open(str(video)) as source:
            frames = list(source.decode(video=0))
            frames[-1].to_image().save(last)
        self.db.set_meta('whitebox_revision', 1)
        self.db.set_meta('tick', 2.)
        self.db.set_meta('whitebox_last_video', str(video))
        record = dict(provenance='AGENT_ACCEPTED_VIDEO_INTERPRETATION', revision_after=1,
                      end=2., artifacts=dict(video=str(video), last_frame=str(last)),
                      cameras=[scene_from_db(self.db)['camera']])
        self.db.set_meta('whitebox_history', [record])
        return record

    def test_initial_pair_state_and_idempotence(self):
        before = self.state()
        record = capture(self.db, self.root/'pair', initial_rgb=self.rgb)
        self.assertEqual(self.state(), before)
        self.assertTrue(record['id'].startswith('initial-'))
        self.assertEqual(record['provenance'], 'INFERRED')
        pixels = np.asarray(Image.open(record['history_path']))
        self.assertEqual(pixels.shape, (64, 192, 3))
        self.assertTrue((pixels[:, :96] == [220, 20, 30]).all())
        self.assertTrue((pixels[:, 96:] == [10, 200, 40]).all())
        self.assertEqual(capture(self.db, self.root/'unused'), record)
        self.assertEqual(self.render.call_count, 1)
        self.assertEqual(len(catalog(self.db)), 1)

    def test_accepted_endpoint_and_unchanged_state(self):
        self.accepted()
        before = self.state()
        record = capture(self.db, self.root/'accepted_pair')
        self.assertEqual(self.state(), before)
        self.assertEqual(record['verification']['decoded_frames'], 2)
        self.assertTrue(record['verification']['endpoint_pixels_equal'])
        self.assertTrue(record['id'].startswith('accepted-'))

    def test_stale_revision_rejected(self):
        self.accepted()
        self.db.set_meta('whitebox_revision', 2)
        with self.assertRaisesRegex(ValueError, 'Stale'):
            capture(self.db, self.root/'stale')
        self.assertEqual(catalog(self.db), [])
        self.render.assert_not_called()

    def test_uncommitted_video_rejected(self):
        self.accepted()
        self.db.set_meta('whitebox_last_video', 'pending.mp4')
        with self.assertRaisesRegex(ValueError, 'committed'):
            capture(self.db, self.root/'pending')
        self.assertEqual(catalog(self.db), [])

    def test_wrong_last_frame_rejected(self):
        record = self.accepted()
        Image.new('RGB', (96, 64), 'blue').save(record['artifacts']['last_frame'])
        with self.assertRaisesRegex(ValueError, 'last_frame'):
            capture(self.db, self.root/'wrong')
        self.assertEqual(catalog(self.db), [])

    def test_render_failure_retry(self):
        self.render.side_effect = RuntimeError('Godot failed')
        before = self.state()
        with self.assertRaises(RuntimeError):
            capture(self.db, self.root/'retry', self.rgb)
        self.assertEqual(catalog(self.db), [])
        self.assertEqual(self.state(), before)
        self.render.side_effect = self.fake_render
        capture(self.db, self.root/'retry', self.rgb)
        self.assertEqual(len(catalog(self.db)), 1)

    def test_world_change_during_render_rejected(self):
        def changed(scene, output, size):
            path = self.fake_render(scene, output, size)
            self.db.set_meta('tick', 99.)
            return path
        self.render.side_effect = changed
        with self.assertRaisesRegex(ValueError, 'World changed'):
            capture(self.db, self.root/'changed', self.rgb)
        self.assertEqual(catalog(self.db), [])

    def test_view_ranking_no_hard_rejection(self):
        record = capture(self.db, self.root/'pair', self.rgb)
        other = deepcopy(record)
        other.update(id='far', revision=9)
        other['camera'] = dict(position=[1000, 0, 0], target=[1001, 0, 0])
        self.db.set_meta('paired_history', [other, record])
        selected = select(self.db, record['camera'], 2)
        self.assertEqual(selected[0]['id'], record['id'])
        self.assertAlmostEqual(selected[0]['distance'], 0)
        self.assertAlmostEqual(selected[0]['angle'], 0, places=5)
        self.assertEqual(selected[1]['id'], 'far')
        self.assertIn('history_path', selected[1])
        self.assertEqual(select(self.db, record['camera'], 0), [])

    def test_same_revision_scene_change_not_hidden_by_cache(self):
        capture(self.db, self.root/'pair', self.rgb)
        header = self.db.get_meta('whitebox_header')
        header['terrain']['seed'] += 1
        self.db.set_meta('whitebox_header', header)
        with self.assertRaisesRegex(ValueError, 'different scene'):
            capture(self.db, self.root/'unused', self.rgb)
        self.assertEqual(len(catalog(self.db)), 1)

    def test_new_revision_cannot_overwrite_prior_output(self):
        original = capture(self.db, self.root/'pair', self.rgb)
        pixels = (self.root/'pair/history.png').read_bytes()
        self.accepted()
        with self.assertRaisesRegex(ValueError, 'another paired capture'):
            capture(self.db, self.root/'pair')
        self.assertEqual((self.root/'pair/history.png').read_bytes(), pixels)
        self.assertEqual(catalog(self.db), [original])


if __name__ == '__main__':
    unittest.main()
