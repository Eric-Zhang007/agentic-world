import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from whitebox.interaction import arrival_distance, check_arrival_review


class ArrivalTests(unittest.TestCase):
    def setUp(self):
        self.scene = {'entities': [{'id': 'player', 'kind': 'actor', 'position': [.4, 0, 0]}]}
        self.contract = {'kind': 'arrival', 'actor_id': 'player', 'target': [.5, 0, 0], 'radius': .15}

    def test_coarse_inside_does_not_prove_visual_arrival(self):
        self.assertTrue(arrival_distance(self.scene, self.contract)['inside'])
        with self.assertRaises(ValueError):
            check_arrival_review(self.scene, self.contract, {'action_complete': True,
                'interaction_assessment': {'status': 'UNOBSERVED', 'evidence': ['Occluded']}})

    def test_partial_progress_is_allowed(self):
        self.scene['entities'][0]['position'] = [.2, 0, 0]
        result = check_arrival_review(self.scene, self.contract, {'action_complete': False,
            'interaction_assessment': {'status': 'CONTRADICTED', 'evidence': ['Stopped short']}})
        self.assertFalse(result['inside'])

    def test_false_completion_is_rejected(self):
        self.scene['entities'][0]['position'] = [0, 0, 0]
        with self.assertRaises(ValueError):
            check_arrival_review(self.scene, self.contract, {'action_complete': True,
                'interaction_assessment': {'status': 'SUPPORTED', 'evidence': ['Moved']}})

    def test_supported_arrival(self):
        result = check_arrival_review(self.scene, self.contract, {'action_complete': True,
            'interaction_assessment': {'status': 'SUPPORTED', 'evidence': ['Feet retained near landmark']}})
        self.assertTrue(result['inside'])

    def test_invalid_radius(self):
        self.contract['radius'] = float('nan')
        with self.assertRaises(ValueError):
            arrival_distance(self.scene, self.contract)


if __name__ == '__main__':
    unittest.main()
