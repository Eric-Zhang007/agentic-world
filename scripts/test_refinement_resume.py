import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.refine_scene_from_evidence import cached_call,preview


class RefinementResumeTests(unittest.TestCase):
    def test_saved_review_is_reused_without_new_model_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'review.json'
            path.write_text(json.dumps({'consistent':False,'evidence':['original rejection']}))
            with patch('scripts.refine_scene_from_evidence.json_call') as model:
                result=cached_call(path,{},'system',{})
                model.assert_not_called()
            self.assertFalse(result['consistent'])

    def test_model_failure_does_not_create_fake_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'review.json'
            with patch('scripts.refine_scene_from_evidence.json_call',side_effect=TimeoutError('gateway')):
                with self.assertRaises(TimeoutError):cached_call(path,{},'system',{})
            self.assertFalse(path.exists())

    def test_changed_scene_cannot_reuse_old_preview(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            (folder/'views').mkdir();(folder/'views/camera.png').touch()
            (folder/'scene.json').write_text(json.dumps({'revision':'old'}))
            with self.assertRaises(ValueError):preview({'revision':'new'},folder)


if __name__=='__main__':unittest.main()
