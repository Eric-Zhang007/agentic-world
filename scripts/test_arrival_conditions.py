import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_arrival_trial import review_conditions


class ConditionPurposeTests(unittest.TestCase):
    def test_wait_is_not_reviewed_as_another_arrival(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = root/'plan'; plan.mkdir()
            preview = root/'preview'; preview.mkdir()
            (preview/'frame_000000.png').touch()
            scene = {'entities': [{'id':'player','kind':'actor','position':[0,0,0]}]}
            values = {'scene.json':scene, 'candidate_world.json':scene,
                      'numerical_report.json':{'valid':True},
                      'action.json':{'phases':[{'type':'wait','duration_sec':5.125}]}}
            for name, value in values.items():
                (plan/name).write_text(json.dumps(value))
            contract = {'kind':'arrival','actor_id':'player','target':[0,0,0],'radius':.18}
            with patch('scripts.run_arrival_trial.json_call', return_value={'usable':True}) as call:
                review_conditions({},plan,'reference.png','annotation.jpg',preview,contract)
            payload = call.call_args.args[2]
            self.assertEqual(payload['phase_purpose'], 'HOLD_AT_ACCEPTED_ENDPOINT')
            self.assertTrue(payload['initial_target_distance']['inside'])
            self.assertEqual(len(call.call_args.args[3]), 6)


if __name__ == '__main__':
    unittest.main()
