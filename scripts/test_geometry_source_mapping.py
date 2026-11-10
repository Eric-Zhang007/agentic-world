"""Regression tests for explicit geometry-to-observation provenance."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compile_geometry_assets import geometry_review_mapping


def entity(identity, source):
    return {'id': identity, 'geometry_evidence': {'source_observation_id': source}}


def perception(*identities):
    return {'observations': [{'observation_id': identity} for identity in identities]}


class GeometrySourceMappingTests(unittest.TestCase):
    def assert_mapping(self, scene, observations, expected):
        before = copy.deepcopy((scene, observations))
        self.assertEqual(geometry_review_mapping(scene, observations), expected)
        self.assertEqual((scene, observations), before)

    def assert_rejected(self, scene, observations, message):
        before = copy.deepcopy((scene, observations))
        with self.assertRaisesRegex(ValueError, message):
            geometry_review_mapping(scene, observations)
        self.assertEqual((scene, observations), before)

    def test_explicit_sources_override_position_and_return_lists(self):
        self.assert_mapping(
            {'entities': [entity('mesh-a', 'obs-b'), entity('mesh-b', 'obs-a')]},
            perception('obs-a', 'obs-b', 'unused'),
            {'obs-b': ['mesh-a'], 'obs-a': ['mesh-b']})

    def test_split_parts_grouped_in_scene_order(self):
        self.assert_mapping(
            {'entities': [entity('wall-right', 'wall'), entity('road', 'ground'),
                          entity('wall-left', 'wall')]},
            perception('wall', 'ground'),
            {'wall': ['wall-right', 'wall-left'], 'ground': ['road']})

    def test_missing_and_none_sources_skipped(self):
        self.assert_mapping(
            {'entities': [{'id': 'missing-evidence'},
                          {'id': 'missing-source', 'geometry_evidence': {}},
                          entity('none-source', None), entity('mapped', 'obs')]},
            perception('obs'), {'obs': ['mapped']})

    def test_no_index_or_matching_id_guessing(self):
        self.assert_mapping(
            {'entities': [{'id': 'obs-0'}, {'id': 'obs-1',
                                          'source_observation_id': 'obs-1'}]},
            perception('obs-0', 'obs-1'), {})

    def test_unknown_source_rejected_after_valid_entity(self):
        self.assert_rejected(
            {'entities': [entity('valid', 'obs'), entity('invalid', 'unknown')]},
            perception('obs'), 'Unknown source observation ID')

    def test_duplicate_observation_ids_rejected_even_without_entities(self):
        self.assert_rejected({'entities': []}, perception('obs', 'obs'),
                             'Duplicate observation or entity IDs')

    def test_duplicate_entity_ids_rejected(self):
        for entities in ([entity('same', 'a'), entity('same', 'b')],
                         [{'id': 'same'}, {'id': 'same'}]):
            with self.subTest(entities=entities):
                self.assert_rejected({'entities': entities}, perception('a', 'b'),
                                     'Duplicate observation or entity IDs')

    def test_empty_inputs(self):
        self.assert_mapping({'entities': []}, perception(), {})


if __name__ == '__main__':
    unittest.main()
