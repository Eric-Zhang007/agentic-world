import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_config
from scripts.test_cases import fixture
from matching.matcher import match
from perception.quality import conflicting_labels, reference_assessability
from state.models import ObservationObject
from verifier.discrepancy import discrepancy


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")

    def test_overlapping_different_labels_not_identity(self):
        objects, camera, observations, _, _ = fixture("A", self.config)
        observations.append(ObservationObject("wrong", "wrong", "tree", observations[1].bbox_xyxy,
                                               .7, 0, 0))
        result = match(objects, observations, camera, self.config["matching"])
        self.assertEqual(len(result.observation_conflicts), 1)
        self.assertFalse(any(m["world_id"] == "person_01" for m in result.matched))
        self.assertTrue(result.ambiguous_matches)
        self.assertEqual(len(result.matched), 2)

    def test_reference_miss_is_not_render_failure_proof(self):
        _, _, _, result, _ = fixture("C", self.config)
        quality = reference_assessability(result)
        report = discrepancy("test", [{"frame_id": 0, "matching": result}],
                             self.config["matching"], quality)
        self.assertEqual(len(report["missing_with_unreliable_reference_detection"]), 3)
        self.assertEqual(report["missing_after_reliable_reference_detection"], [])

    def test_spatially_separate_labels_remain_independent(self):
        _, _, obs, _, _ = fixture("A", self.config)
        self.assertEqual(conflicting_labels(obs), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
