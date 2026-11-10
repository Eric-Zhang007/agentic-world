import unittest
import numpy as np
from memory.gta_visibility import geometry_evidence, visibility_evidence


class VisibilityTests(unittest.TestCase):
    def obj(self, position):
        return {'dataset_entity_id': 1, 'position': position,
                'orientation_quaternion': [0, 0, 0, 1],
                'model_bounds_min': [-1, -1, -1], 'model_bounds_max': [1, 1, 1]}

    def test_inside_not_invisible(self):
        self.assertEqual(geometry_evidence(self.obj([0,0,0]), np.zeros(3), np.eye(3)), 'CAMERA_INSIDE_OBB')

    def test_projectable_not_visible(self):
        result = visibility_evidence(self.obj([0,0,5]), np.zeros(3), np.eye(3), [])
        self.assertEqual(result['geometry_status'], 'PROJECTABLE_OBB')
        self.assertEqual(result['rgb_visibility'], 'UNKNOWN')
        self.assertFalse(result['appearance_reference_approved'])

    def test_near_crossing(self):
        self.assertEqual(geometry_evidence(self.obj([2,0,.5]), np.zeros(3), np.eye(3)), 'NEAR_PLANE_INTERSECTION')

    def test_behind_and_outside(self):
        self.assertEqual(geometry_evidence(self.obj([0,0,-5]), np.zeros(3), np.eye(3)), 'BEHIND_NEAR_PLANE')
        self.assertEqual(geometry_evidence(self.obj([100,0,5]), np.zeros(3), np.eye(3)), 'OUTSIDE_IMAGE')

    def test_occupancy_not_automatic_occlusion(self):
        obj = self.obj([0,0,5]); obj['current_vehicle_network_id'] = 9
        result = visibility_evidence(obj, np.zeros(3), np.eye(3), [{'network_id':9,'dataset_entity_id':2}])
        self.assertTrue(result['in_vehicle'])
        self.assertEqual(result['carrier_entity_ids'], ['2'])
        self.assertEqual(result['rgb_visibility'], 'UNKNOWN')

    def test_invalid(self):
        obj = self.obj([0,0,5]); obj['position'][0] = float('nan')
        self.assertEqual(geometry_evidence(obj, np.zeros(3), np.eye(3)), 'INVALID_GEOMETRY')


if __name__ == '__main__':
    unittest.main()
