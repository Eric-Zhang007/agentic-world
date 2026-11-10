"""Render state gates with real temporary DB/PNG data and no renderer execution."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from harness.pi.world_tools import dispatch, save
from renderer.memory_conditioning import image_bindings
from scripts.run_comfy_control import add_start_guide
from scripts.test_whitebox_world import fixture
from state.db import WorldDB
from whitebox.motion import simulate
from whitebox.world import initialize, scene_from_db
from PIL import Image


class MemoryGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.db_path = self.root / 'world.sqlite'
        self.db = WorldDB(self.db_path)
        self.addCleanup(self.db.connection.close)
        initialize(self.db, fixture())
        self.scene = scene_from_db(self.db)
        self.plan = self.root / 'plan'
        self.plan.mkdir()
        action = {'actor_id': 'player', 'phases': [
            {'type': 'wait', 'duration_sec': 3, 'criterion': 'Observe'}]}
        self.timeline, _, diagnostics = simulate(self.scene, action, 0)
        self.assertTrue(diagnostics['valid'])
        for name, value in [('scene', self.scene), ('action', action),
                            ('timeline', self.timeline),
                            ('status', {'revision': 0, 'world_start': 0})]:
            save(self.plan / (name + '.json'), value)
        video = self.root / 'whitebox.mp4'
        save(self.root / 'compiled_plan.json', {
            'plan': str(self.plan), 'scene': self.scene,
            'timeline': self.timeline, 'reference_video': str(video)})
        self.accepted = self.png('accepted.png', (10, 20, 30))
        self.same_pixels = self.png('copied_endpoint.png', (10, 20, 30))
        self.old = self.png('old.png', (30, 20, 10))
        self.set_meta(whitebox_last_preview=str(self.accepted))
        self.args = {'db': str(self.db_path), 'renderer': {}, 'request': {
            'reference_video': str(video), 'first_frame': str(self.same_pixels),
            'source_video': None, 'prompt': 'Wait.',
            'output_dir': str(self.root / 'render')}}
        self.selector = self.enterContext(patch('memory.paired_history.select', return_value=[]))
        self.renderer = self.enterContext(patch('renderer.comfy_h3.ComfyH3Renderer'))
        self.renderer.return_value.render.return_value = self.root / 'generated.mp4'

    def png(self, name, color, size=(16, 12)):
        path = self.root / name
        Image.new('RGB', size, color).save(path)
        return path

    def set_meta(self, **values):
        for key, value in values.items():
            self.db.set_meta(key, value)
        self.db.connection.commit()

    def initial(self, path):
        return [{'source': 'INITIAL_RGB', 'paths': {'rgb.png': str(path)}}]

    def rejected(self, message):
        with self.assertRaisesRegex(ValueError, message):
            dispatch('render', self.args)
        self.selector.assert_not_called()
        self.renderer.assert_not_called()
        self.assertFalse((self.root / 'render').exists())

    def test_identical_pixels_at_different_path_pass(self):
        before = scene_from_db(self.db)
        result = dispatch('render', self.args)
        self.assertEqual(result['video'], str(self.root / 'generated.mp4'))
        self.assertEqual(self.renderer.return_value.render.call_args.kwargs['first_frame'],
                         str(self.same_pixels))
        self.assertEqual(scene_from_db(self.db), before)
        self.assertEqual(self.db.get_meta('whitebox_revision'), 0)
        self.assertEqual(self.db.get_meta('tick'), 0)
        self.assertEqual(self.selector.call_args.args[1], self.timeline['frames'][-1]['camera'])
        self.assertEqual(self.selector.call_args.kwargs, {'limit': 1})

    def test_missing_db_rejected_after_valid_compile(self):
        del self.args['db']
        self.rejected('WORLD_REQUIRED')

    def test_compile_required_takes_priority_over_missing_db(self):
        del self.args['db']
        (self.root / 'compiled_plan.json').unlink()
        self.rejected('COMPILE_REQUIRED')

    def test_changed_compilation_rejected_before_missing_db(self):
        del self.args['db']
        save(self.root / 'compiled_plan.json', {
            'plan': str(self.plan), 'scene': {},
            'timeline': self.timeline,
            'reference_video': self.args['request']['reference_video']})
        self.rejected('RECOMPILE_REQUIRED')

    def test_prefix_rejected_in_checked_world(self):
        self.args['request']['prefix_video'] = str(self.root / 'old_prefix.mp4')
        self.rejected('FIRST_FRAME_CONFLICT')

    def test_hud_rewriting_rejected_in_checked_world(self):
        self.args['request']['hud_profile'] = str(self.root / 'hud.json')
        self.rejected('FIRST_FRAME_CONFLICT')

    def test_anchor_defaults_true_without_mutating_request(self):
        dispatch('render', self.args)
        self.assertIs(self.renderer.return_value.render.call_args.kwargs['anchor_first'], True)
        self.assertNotIn('anchor_first', self.args['request'])

    def test_explicit_true_anchor_passes(self):
        self.args['request']['anchor_first'] = True
        dispatch('render', self.args)
        self.assertIs(self.renderer.return_value.render.call_args.kwargs['anchor_first'], True)

    def test_explicit_false_anchor_rejected(self):
        self.args['request']['anchor_first'] = False
        self.rejected('FIRST_FRAME_REQUIRED')

    def test_missing_accepted_and_initial_rgb_rejected(self):
        self.set_meta(whitebox_last_preview=None, paired_history=[])
        self.rejected('INITIAL_RGB_REQUIRED')

    def test_noninitial_history_cannot_supply_initial_rgb(self):
        self.set_meta(whitebox_last_preview=None, paired_history=[{
            'source': 'ACCEPTED_VIDEO_ENDPOINT',
            'paths': {'rgb.png': str(self.same_pixels)}}])
        self.rejected('INITIAL_RGB_REQUIRED')

    def test_stale_revision_rejected_even_if_scene_matches(self):
        self.set_meta(whitebox_revision=1)
        self.rejected('WORLD_CHANGED: plan revision or time is stale')

    def test_stale_tick_rejected_even_if_scene_matches(self):
        self.set_meta(tick=3.0)
        self.rejected('WORLD_CHANGED: plan revision or time is stale')

    def test_old_first_frame_rejected(self):
        self.args['request']['first_frame'] = str(self.old)
        self.rejected('FIRST_FRAME_MISMATCH')

    def test_one_pixel_difference_rejected(self):
        with Image.open(self.same_pixels) as source:
            changed = source.copy()
        changed.putpixel((0, 0), (11, 20, 30))
        changed.save(self.same_pixels)
        self.rejected('FIRST_FRAME_MISMATCH')

    def test_dimension_difference_rejected_without_resizing(self):
        self.args['request']['first_frame'] = str(self.png('small.png', (10, 20, 30), (8, 6)))
        self.rejected('FIRST_FRAME_MISMATCH')

    def test_initial_rgb_fallback_accepts_same_pixels(self):
        self.set_meta(whitebox_last_preview=None, paired_history=self.initial(self.accepted))
        dispatch('render', self.args)
        self.renderer.return_value.render.assert_called_once()

    def test_initial_rgb_fallback_rejects_wrong_pixels(self):
        self.set_meta(whitebox_last_preview=None, paired_history=self.initial(self.old))
        self.rejected('FIRST_FRAME_MISMATCH')

    def test_accepted_endpoint_takes_priority_over_initial_rgb(self):
        self.set_meta(paired_history=self.initial(self.old))
        dispatch('render', self.args)
        self.renderer.return_value.render.assert_called_once()

    def test_initial_frame_cannot_replace_later_accepted_endpoint(self):
        self.set_meta(paired_history=self.initial(self.old))
        self.args['request']['first_frame'] = str(self.old)
        self.rejected('FIRST_FRAME_MISMATCH')


class WorkerAnchorTests(unittest.TestCase):
    def test_history_is_picture_one_but_anchor_is_current_first(self):
        first, history = object(), object()
        bindings = image_bindings(first, history)
        self.assertIs(bindings['ref_image_1'], history)
        self.assertIs(bindings['ref_image_2'], first)
        node = Mock()
        node.execute.return_value = SimpleNamespace(result=['guided'])
        self.assertEqual(add_start_guide(node, 'positive', 'latent', 'vae', first, True), 'guided')
        node.execute.assert_called_once_with('positive', 'latent', 0, vae='vae', image=first)

    def test_disabled_anchor_does_not_add_guide(self):
        node = Mock()
        self.assertEqual(add_start_guide(node, 'positive', 'latent', 'vae', object(), False), 'positive')
        node.execute.assert_not_called()

    def test_prefix_currently_takes_priority_over_first_anchor(self):
        # Characterize the legacy override; this is not a pixel-consistency check.
        node = Mock()
        node.execute.return_value = SimpleNamespace(result=['guided'])
        first, prefix = object(), object()
        add_start_guide(node, 'positive', 'latent', 'vae', first, True, prefix)
        node.execute.assert_called_once_with('positive', 'latent', 0, vae='vae', image=prefix)


if __name__ == '__main__':
    unittest.main()
