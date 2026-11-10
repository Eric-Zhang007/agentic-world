import json
from pathlib import Path
import tempfile
import unittest

from scripts.compile_saved_rollout_prompt import plan_prefix_frames


class SavedPromptTests(unittest.TestCase):
    def test_historical_prefix_survives_saved_plan_compilation(self):
        with tempfile.TemporaryDirectory() as folder:
            plan=Path(folder)
            for prefix,ticks in [(1,123),(22,102)]:
                (plan/'alignment.json').write_text(json.dumps({'prefix_frames':prefix}))
                self.assertEqual(plan_prefix_frames(plan,{'frames':[None]*ticks}),prefix)
            for prefix in [None,True,0,2]:
                (plan/'alignment.json').write_text(json.dumps({'prefix_frames':prefix}))
                with self.assertRaises(ValueError):plan_prefix_frames(plan,{'frames':[None]*123})


if __name__=='__main__':unittest.main()
