"""Deterministic contract fixtures plus optional real GPT decision calls.

These controlled observations test A/B/C logic; they are not claimed to be H3 detections.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.decision import decide
from agent.world_agent import WorldTools
from common import load_config, save_json
from compiler.projection import project_object
from matching.matcher import match
from memory.observation_store import ObservationStore
from scripts.test_h3 import scene
from state.db import WorldDB
from state.models import ObservationObject
from verifier.discrepancy import discrepancy
from verifier.reconcile import apply_reconciliation
from perception.frame_sampler import sample_frames


def fixture(case, config):
    objects, camera = scene()
    observations = []
    for index, obj in enumerate(objects):
        if not obj.state.get("matchable", True):
            continue
        p = project_object(obj, camera)
        observations.append(ObservationObject(
            f"test_{index}", f"track_{index}", obj.category, p["bbox"], 0.95, 0, 0,
            source="TEST_FIXTURE"))
    if case in ("B", "C"):
        observations.append(ObservationObject("dog", "dog_track", "dog", [5, 5, 35, 35],
                                              0.95, 0, 0, source="TEST_FIXTURE"))
    if case == "C":
        observations = [observations[-1]]
        for index, category in enumerate(("truck", "bus", "bicycle", "building")):
            observations.append(ObservationObject(
                f"unexpected_{index}", f"new_{index}", category, [30, 30, 100, 100],
                0.95, 0, 0, source="TEST_FIXTURE"))
    result = match(objects, observations, camera, config["matching"]).to_dict()
    report = discrepancy(case, [{"frame_id": 0, "matching": result}], config["matching"])
    return objects, camera, observations, result, report


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")

    def test_a_matches(self):
        _, _, _, result, report = fixture("A", self.config)
        self.assertEqual(len(result["matched"]), 3)
        self.assertEqual(report["summary"]["missing_ratio"], 0)

    def test_b_dog_stays_unmatched(self):
        _, _, _, result, _ = fixture("B", self.config)
        self.assertEqual([o["category"] for o in result["unmatched_observations"]], ["dog"])
        self.assertEqual(len(result["matched"]), 3)

    def test_c_deviation(self):
        _, _, _, result, report = fixture("C", self.config)
        self.assertEqual(len(result["matched"]), 0)
        self.assertEqual(report["summary"]["missing_ratio"], 1)
        self.assertEqual(report["summary"]["unmatched_ratio"], 1)

    def test_no_camera(self):
        objects, _, observations, _, _ = fixture("A", self.config)
        result = match(objects, observations, None, self.config["matching"])
        self.assertEqual(len(result.matched), 3)

    def test_missing_modalities_do_not_fabricate_scores(self):
        _, _, _, result, _ = fixture("A", self.config)
        self.assertNotIn("visual_similarity", result["matched"][0])
        self.assertNotIn("depth_error", result["matched"][0])

    def test_reconciliation_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            world = WorldDB(Path(directory) / "world.sqlite")
            obj = scene()[0][1]
            world.save(obj, "test setup")
            before = world.snapshot()
            edit = {"op": "update", "object_id": obj.object_id, "position": [1, 2, 3]}
            with self.assertRaises(ValueError):
                apply_reconciliation(world, {"action": "RECONCILE_WORLD", "edits": [edit]})
            self.assertEqual(before, world.snapshot())
            world.add_evidence("e1", "MANUAL", obj.object_id, {"allowed_changes": {"position": [1, 2, 3]}})
            edit["evidence_ids"] = ["e1"]
            apply_reconciliation(world, {"action": "RECONCILE_WORLD", "edits": [edit]})
            self.assertEqual(world.get_object(obj.object_id).position, [1, 2, 3])

    def test_generated_spawn_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            world = WorldDB(Path(directory) / "world.sqlite")
            tools = WorldTools(world, ObservationStore(Path(directory) / "observations.sqlite"))
            with self.assertRaises(ValueError):
                tools.spawn_object({"object_id": "dog", "category": "dog"}, [], "generated dog")
            self.assertIsNone(world.get_object("dog"))

    def test_identity_ambiguity_not_accepted(self):
        objects, camera, observations, _, _ = fixture("A", self.config)
        twin = deepcopy(objects[1])
        twin.object_id = "car_twin"
        result = match([objects[1], twin], [observations[0]], camera, self.config["matching"])
        self.assertEqual(len(result.matched), 0)
        self.assertTrue(result.ambiguous_matches)

    def test_no_observations_all_missing(self):
        objects, camera = scene()
        result = match(objects, [], camera, self.config["matching"])
        self.assertEqual(len(result.missing_world_objects), 3)
        self.assertEqual(result.matched, [])

    def test_no_world_all_unmatched(self):
        _, camera, observations, _, _ = fixture("A", self.config)
        result = match([], observations, camera, self.config["matching"])
        self.assertEqual(len(result.unmatched_observations), 3)

    def test_invalid_second_edit_no_partial_write(self):
        with tempfile.TemporaryDirectory() as directory:
            world = WorldDB(Path(directory) / "world.sqlite")
            obj = scene()[0][1]
            world.save(obj, "setup")
            world.add_evidence("e1", "MANUAL", obj.object_id, {"allowed_changes": {"position": [1, 2, 3]}})
            before = world.snapshot()
            edits = [{"op": "update", "object_id": obj.object_id, "position": [1, 2, 3],
                      "evidence_ids": ["e1"]},
                     {"op": "update", "object_id": "made_up", "position": [0, 0, 0]}]
            with self.assertRaises(ValueError):
                apply_reconciliation(world, {"action": "RECONCILE_WORLD", "edits": edits})
            self.assertEqual(before, world.snapshot())

    def test_generated_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            world = WorldDB(Path(directory) / "world.sqlite")
            with self.assertRaises(ValueError):
                world.add_evidence("render", "GENERATED", "tree", {"allowed_changes": {"position": [1, 2, 3]}})

    def test_frame_sampling_real_smoke(self):
        path = Path(__file__).resolve().parents[1] / "outputs/h3_smoke/generated.mp4"
        if not path.exists():
            self.skipTest("H3 smoke media not present on this machine")
        with tempfile.TemporaryDirectory() as directory:
            frames = sample_frames(path, directory, interval=5)
            self.assertEqual(len(frames), 3)
            self.assertEqual(frames[0]["frame_id"], 0)
            self.assertEqual(frames[-1]["frame_id"], 71)

    def test_unsupported_conditions_rejected(self):
        from renderer.minimax_h3 import H3Renderer
        with self.assertRaises(ValueError):
            H3Renderer(self.config["renderer"]).render("test", depth_sequence=["depth.png"], output_dir="unused")


def agent_cases(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    decisions = {}
    for case in ("A", "B", "C"):
        with tempfile.TemporaryDirectory() as directory:
            objects, camera, _, matching, report = fixture(case, config)
            world = WorldDB(Path(directory) / "world.sqlite")
            world.set_meta("camera", camera)
            for obj in objects:
                world.save(obj, "controlled A/B/C fixture setup")
            before = world.snapshot()
            decision = decide(config["agent"], world, "Preserve the specified static scene", report)
            save_json(output / f"case_{case}.json", {"fixture_not_h3_output": True, "matching": matching,
                                                    "report": report, "decision": decision,
                                                    "world_unchanged": world.snapshot() == before})
            assert world.get_object("dog") is None
            decisions[case] = decision["action"]
    assert decisions["A"] == "ACCEPT", decisions
    assert decisions["B"] != "RECONCILE_WORLD", decisions
    assert decisions["C"] == "REGENERATE", decisions
    print(decisions, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", action="store_true")
    parser.add_argument("--output", default="outputs/contract_cases")
    args = parser.parse_args()
    if args.agent:
        agent_cases(load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml"), args.output)
    else:
        unittest.main(argv=[sys.argv[0]], verbosity=2)
