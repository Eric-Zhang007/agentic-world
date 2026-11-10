import tempfile
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from agent.interaction_evidence import validate_interval, review_arrival


class EvidenceTests(unittest.TestCase):
    def test_invalid_or_excessive_requests(self):
        for start, end in [(-1, 1), (0, 3), (2, 1), (0, float('nan')), (1, 6), (False, 1)]:
            with self.assertRaises(ValueError):
                validate_interval({'start_sec': start, 'end_sec': end, 'question': 'Did it return?'}, 5)

    def test_tool_is_optional(self):
        response = {'role': 'assistant', 'content': '{"arrival":"CONTRADICTED",'
                    '"retained_displacement":"CONTRADICTED","evidence":["No shift"]}'}
        with tempfile.TemporaryDirectory() as tmp, \
                patch('agent.interaction_evidence.image_content', return_value={'type':'text','text':'image'}), \
                patch('agent.interaction_evidence.AgentClient') as client, \
                patch('agent.interaction_evidence.inspect_motion') as inspect:
            client.return_value.complete.return_value = response
            result = review_arrival({}, 'video.mp4', {}, [{'timestamp':0,'path':'a'}, {'timestamp':3,'path':'b'}],
                                    Path(tmp)/'result')
            inspect.assert_not_called()
            self.assertEqual(result['inspection_calls'], 0)
            self.assertFalse(result['world_modified'])

    def test_requested_interval_and_images_are_returned_to_agent(self):
        request = {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'inspect_1', 'type': 'function', 'function': {'name': 'inspect_motion_interval',
             'arguments': json.dumps({'start_sec': .5, 'end_sec': 1.5, 'question': 'Did it return?'})}}]}
        final = {'role': 'assistant', 'content': '{"arrival":"UNOBSERVED",'
                 '"retained_displacement":"UNOBSERVED","evidence":["Occluded"]}'}
        evidence = {'start_sec': .5, 'end_sec': 1.5, 'analysis_width': 640, 'pair_count':24,
                    'measured_pairs': 23, 'motion_peaks': [], 'limitations': ['No metric 3D'],
                    'frames': [{'path':'dense.jpg','timestamp':.5}]}
        with tempfile.TemporaryDirectory() as tmp, \
                patch('agent.interaction_evidence.image_content', return_value={'type':'text','text':'image'}), \
                patch('agent.interaction_evidence.AgentClient') as client, \
                patch('agent.interaction_evidence.inspect_motion', return_value=evidence) as inspect:
            client.return_value.complete.side_effect = [request, final]
            output = Path(tmp)/'result'
            result = review_arrival({}, 'video.mp4', {}, [{'timestamp':0,'path':'a'}, {'timestamp':3,'path':'b'}], output)
            self.assertEqual(result['inspection_calls'], 1)
            self.assertEqual(inspect.call_args.args[2:4], (.5, 1.5))
            messages = client.return_value.complete.call_args.args[0]
            self.assertTrue(any(m.get('tool_call_id') == 'inspect_1' for m in messages))
            self.assertTrue((output/'inspection_01.json').exists())
            self.assertEqual(len(json.loads((output/'tool_trace.json').read_text())), 3)


if __name__ == '__main__':
    unittest.main()
