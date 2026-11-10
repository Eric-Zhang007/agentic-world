"""Small renderer integration fixture; uses only Python's standard library."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

from render import validate


def fixture():
    scene = {
        "schema_version": 1, "seed": 17,
        "terrain": {"size": [18, 18], "resolution": 32, "seed": 17,
                    "height_amplitude": 1.3, "flatten_radius": 5},
        "camera": {"position": [6, 4.5, 8], "target": [0, 1, 0]},
        "entities": [
            {"id": "actor", "kind": "actor", "position": [0, 0, 0],
             "rotation_y": 0, "size": [0.65, 1.9, 0.4], "color": [0.85, 0.3, 0.23]},
            {"id": "box", "kind": "box", "position": [-2, 0.5, 0],
             "rotation_y": 0.3, "size": [1.2, 1, 1], "color": [0.24, 0.48, 0.8]},
            {"id": "stairs", "kind": "stairs", "position": [2, 0.7, -1],
             "rotation_y": 0, "size": [1.8, 1.4, 2.8], "steps": 7,
             "color": [0.72, 0.73, 0.76]},
            {"id": "staff", "kind": "staff", "position": [0, 1, 0],
             "rotation_y": 0, "size": [0.07, 2, 0.07], "color": [0.85, 0.68, 0.2],
             "weapon": {"a": [-0.8, 1.2, 0.5], "b": [1.2, 1.2, 0.5]}},
        ]}
    frames = []
    for i in range(24):
        wave = math.sin(i / 23 * math.pi)
        joints = {
            "pelvis": [0, 0.95, 0], "chest": [0, 1.4, 0], "head": [0, 1.75, 0],
            "left_shoulder": [-0.32, 1.45, 0], "right_shoulder": [0.32, 1.45, 0],
            "left_elbow": [-0.55, 1.15 + wave * 0.35, 0.2],
            "right_elbow": [0.55, 1.15 + wave * 0.35, 0.2],
            "left_hand": [-0.6, 1.2 + wave * 0.6, 0.5],
            "right_hand": [0.6, 1.2 + wave * 0.6, 0.5],
            "left_hip": [-0.15, 0.95, 0], "right_hip": [0.15, 0.95, 0],
            "left_knee": [-0.2, 0.55, 0.08], "right_knee": [0.2, 0.55, 0.08],
            "left_foot": [-0.2, 0.08, 0.1], "right_foot": [0.2, 0.08, 0.1],
        }
        frames.append({"time": i / 24, "entities": [
            {"id": "actor", "position": [0, 0, 0], "rotation_y": 0, "pose": joints},
            {"id": "staff", "weapon": {"a": [-1, 1.2 + wave*0.6, 0.5],
                                          "b": [1, 1.2 + wave*0.6, 0.5]}},
        ], "camera": scene["camera"]})
    return scene, {"fps": 24, "frames": frames}


class ContractTests(unittest.TestCase):
    def test_fixture(self):
        validate(*fixture())

    def test_invalid_kind(self):
        scene, timeline = fixture()
        scene["entities"][0]["kind"] = "unknown"
        with self.assertRaises(ValueError): validate(scene, timeline)

    def test_unknown_timeline_entity(self):
        scene, timeline = fixture()
        timeline["frames"][0]["entities"][0]["id"] = "missing"
        with self.assertRaises(ValueError): validate(scene, timeline)

    def test_duplicate_ids(self):
        scene, timeline = fixture()
        scene["entities"][1]["id"] = "actor"
        with self.assertRaises(ValueError): validate(scene, timeline)

    def test_bad_pose(self):
        scene, timeline = fixture()
        timeline["frames"][0]["entities"][0]["pose"]["head"] = [0, float("nan"), 0]
        with self.assertRaises(ValueError): validate(scene, timeline)

    def test_bad_resolution(self):
        scene, timeline = fixture()
        scene["terrain"]["resolution"] = 0
        with self.assertRaises(ValueError): validate(scene, timeline)

    def test_bad_camera(self):
        scene, timeline = fixture()
        scene["camera"]["target"] = scene["camera"]["position"]
        with self.assertRaises(ValueError): validate(scene, timeline)

    def test_nonmonotonic_time(self):
        scene, timeline = fixture()
        timeline["frames"][1]["time"] = 0
        with self.assertRaises(ValueError): validate(scene, timeline)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mp4", action="store_true")
    parser.add_argument("--static", action="store_true", help="Verify identical states produce identical PNG bytes")
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        scene, timeline = fixture()
        if args.static:
            first = timeline["frames"][0]
            timeline["frames"] = [first, dict(first, time=1/24)]
        for name, data in [("scene", scene), ("timeline", timeline)]:
            (args.output / f"{name}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        command = [sys.executable, str(Path(__file__).with_name("render.py")),
                   "--scene", str(args.output / "scene.json"),
                   "--timeline", str(args.output / "timeline.json"),
                   "--output", str(args.output / "renders")]
        if args.mp4:
            command.append("--mp4")
        subprocess.run(command, check=True)
        manifest = json.loads((args.output / "renders/manifest.json").read_text())
        expected = 6 if args.static else (28 if args.mp4 else 13)
        assert len(manifest["images"]) == expected, manifest
        assert all(item["luminance_range"] >= 0.08 for item in manifest["images"])
        for sample in manifest["terrain_samples"]:
            x, z = sample["x"], sample["z"]
            expected_height = 1.3 * max(0, min(1, (math.hypot(x, z)-5)/3)) * (
                0.5*math.sin(0.31*x+17*0.013)*math.cos(0.27*z-17*0.017)
                + 0.25*math.sin(0.61*(x+z)+17*0.007))
            assert math.isclose(sample["height"], expected_height, abs_tol=1e-9), sample
        renders = args.output / "renders"
        if args.static:
            assert (renders / "frame_000000.png").read_bytes() == (renders / "frame_000001.png").read_bytes()
        else:
            assert (renders / "frame_000000.png").read_bytes() != (renders / "frame_000012.png").read_bytes()
        assert len({(renders / f"{view}.png").read_bytes() for view in ("overview", "front", "side", "camera")}) == 4
        print(f"PASS: {expected} nonblank renders")


if __name__ == "__main__":
    main()
