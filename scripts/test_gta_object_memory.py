import unittest

import numpy as np

from scripts.prepare_gta_object_memory import camera_geometry, coarse_visibility, project_box
from scripts.index_gta_memory_segments import valid_runs
from memory.gta_validation import camera_exclusion, validate_reference_identity


def box(z):
    return {'position': [0, 0, z], 'orientation_quaternion': [0, 0, 0, 1],
            'model_bounds_min': [-1, -1, -1], 'model_bounds_max': [1, 1, 1]}


class GeometryTests(unittest.TestCase):
    def test_missing_camera_splits_runs(self):
        rows = [{'status': 'estimated_alignment', 'camera_interpolation': {'alpha': 0, 'lower_client_sample_seq': 0, 'upper_client_sample_seq': 1},
                 'proxy_frame_index': i, 'rgb_decode_index': i, 'server_clock_domain': 's',
                 'client_clock_domain': 'c', 'server_time_estimate_ns': i+1} for i in range(5)]
        rows[2]['status'] = 'missing_camera'
        runs, excluded = valid_runs(rows)
        self.assertEqual([(r['start_frame'],r['end_frame_exclusive']) for r in runs],[(0,2),(3,5)])
        self.assertEqual(excluded, {'missing_camera': 1})

    def test_clock_change_splits_runs(self):
        rows = [{'status': 'estimated_alignment', 'camera_interpolation': {'alpha': 0, 'lower_client_sample_seq': 0, 'upper_client_sample_seq': 1},
                 'proxy_frame_index': i, 'rgb_decode_index': i, 'server_clock_domain': str(i),
                 'client_clock_domain': 'c', 'server_time_estimate_ns': i+1} for i in range(2)]
        runs, _ = valid_runs(rows)
        self.assertEqual(len(runs),2)

    def test_failed_interpolation_rejected(self):
        self.assertEqual(camera_exclusion({'status':'estimated_alignment', 'camera_interpolation':
            {'method':'nearest_real_camera_after_invalid_interpolation'}}), 'unsupported_camera_interpolation')

    def test_nonfinite_alpha_rejected(self):
        self.assertEqual(camera_exclusion({'status':'estimated_alignment', 'camera_interpolation':
            {'lower_client_sample_seq':0,'upper_client_sample_seq':1,'alpha':float('nan')}}),'invalid_camera_alpha')

    def test_replaced_entity_cannot_reuse_reference(self):
        with self.assertRaises(ValueError):
            validate_reference_identity([{'interpolated_entities':[{'dataset_entity_id':2,'model_hash':4}]}],1)

    def test_future_reference_rejected(self):
        rows=[{'interpolated_entities':[{'dataset_entity_id':1,'model_hash':4}],
               'frame_metadata':{'server_clock_domain':'a','server_time_estimate_ns':t}} for t in [5,4]]
        with self.assertRaises(ValueError): validate_reference_identity(rows,1)

    def test_ray_basis_projects_source_calibration_points(self):
        rays = [{'uv': uv, 'direction': np.array(direction).tolist()}
                for uv, direction in [([.5, .5], [0, 0, 1]),
                                      ([.75, .5], [.5, 0, 1]),
                                      ([.5, .75], [0, .5, 1])]]
        sample = {'rendered_camera_position': [0, 0, 0], 'projection_rays': {'rays': rays}}
        origin, basis = camera_geometry(sample, sample, .3)
        for ray in rays:
            q = np.linalg.solve(basis, np.array(ray['direction'])*10-origin)
            np.testing.assert_allclose(q[:2]/q[2]+.5, ray['uv'], atol=1e-7)

    def test_near_object_occludes_far_object(self):
        owner, hits = coarse_visibility([box(4), box(8)], np.zeros(3), np.eye(3), 64, 64)
        self.assertGreater(hits[1], 0)
        self.assertEqual(int((owner == 1).sum()), 0)
        self.assertGreater(int((owner == 0).sum()), 0)

    def test_behind_camera_and_near_plane_rejected(self):
        for z in [-5, 0]:
            self.assertIsNone(project_box(box(z), np.zeros(3), np.eye(3), 100, 100))

    def test_projected_center(self):
        bounds = project_box(box(5), np.zeros(3), np.eye(3), 100, 100)
        np.testing.assert_allclose(bounds, [25, 25, 75, 75])


if __name__ == '__main__':
    unittest.main()
