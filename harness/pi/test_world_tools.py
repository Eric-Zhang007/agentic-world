"""Replay a real saved video on a disposable world; no gateway calls."""
import json
from pathlib import Path
import tempfile
import unittest
from world_tools import ROOT, dispatch, read, save


class WorldToolsTest(unittest.TestCase):
    def test_real_partial_accept_and_resume(self):
        old = ROOT/'outputs/posture_live_20261010/segment_00'
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            db = str(out/'world.sqlite')
            dispatch('fork', {'source': str(ROOT/'outputs/video_reconcile_guard_v2_20261009/world.sqlite'), 'db': db})
            before = dispatch('status', {'db': db})
            self.assertEqual(before['scene'], read(old/'plan/scene.json'))
            planned=dispatch('plan', {'db': db, 'action': str(old/'plan/action.json'), 'output': str(out/'plan')})
            timeline=read(out/'plan/timeline.json')
            self.assertEqual(planned['video_frames'],len(timeline['frames'])+1)
            self.assertEqual(planned['video_frames']%17,5)
            self.assertEqual(planned['duration_sec'],len(timeline['frames'])/timeline['fps'])
            self.assertEqual(planned['diagnostics'],read(out/'plan/diagnostics.json'))
            self.assertEqual(planned['endpoint']['camera'],read(out/'plan/endpoint.json')['camera'])
            evidence = dispatch('inspect_video', {'plan': str(out/'plan'),
                'video': str(old/'render/controlled.mp4'), 'output': str(out/'evidence')})
            self.assertTrue(evidence['frames'])
            self.assertEqual(len(evidence['review_images']),3)
            self.assertTrue(all(Path(p).is_file() for p in evidence['review_images']))
            args = {'db': db, 'plan': str(out/'plan'), 'evidence': str(out/'evidence/evidence.json'),
                    'review': str(old/'review/review.json'), 'output': str(out/'accepted')}
            rejected = dict(read(old/'review/review.json'), decision='REGENERATE')
            save(out/'rejected.json', rejected)
            self.assertEqual(dispatch('commit', dict(args, review=str(out/'rejected.json')))['status'], 'REJECTED')
            self.assertEqual(dispatch('status', {'db': db}), before)
            result = dispatch('commit', args)
            self.assertFalse(result['action_complete'])
            after = dispatch('status', {'db': db})
            tiger = next(e for e in after['scene']['entities'] if e['id']=='tiger_guard')
            self.assertEqual(tiger['posture_weights']['crouched'], 1.)
            self.assertEqual(after['history_length'], before['history_length']+1)
            self.assertEqual(dispatch('commit', args)['status'], 'ALREADY_COMMITTED')
            self.assertEqual(dispatch('status', {'db': db}), after)

            action = read(old/'review/next_plan/action.json')
            action['goal'] = after['pending_action']['continuation_goal']
            save(out/'action.json', action)
            dispatch('plan', {'db': db, 'action': str(out/'action.json'), 'output': str(out/'next')})
            self.assertEqual(read(out/'next/scene.json'), after['scene'])
            rejected = dict(read(old/'review/review.json'), decision='REGENERATE')
            save(out/'rejected.json', rejected)
            with self.assertRaises(ValueError):
                dispatch('commit', dict(args, review=str(out/'rejected.json')))
            self.assertEqual(dispatch('status', {'db': db}), after)

    def test_missing_world_does_not_create_database(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'missing.sqlite'
            with self.assertRaises(ValueError):
                dispatch('status', {'db': str(path)})
            self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
