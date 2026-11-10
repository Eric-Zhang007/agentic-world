"""Real model perception only; missing weights are an error, never synthetic detections."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy.optimize import linear_sum_assignment

from matching.costs import iou
from state.models import ObservationObject


class PerceptionPipeline:
    def __init__(self, config):
        self.config = config
        self.previous = []
        self.next_track = 0

    def detect_frame(self, frame, output_dir, namespace, source="GENERATED"):
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, HF_HOME="/mnt/hdd/agentic-world-models", HF_HUB_OFFLINE="1")
        command = [self.config.get("python", "/home/star/agentic_world/.venv/bin/python"),
                   str(Path(__file__).resolve().parents[1] / "scripts/probe_perception.py"),
                   "--image", str(Path(frame["path"]).resolve()), "--output", str(output),
                   "--device", self.config["device"],
                   "--threshold", str(self.config["threshold"]),
                   "--text", ". ".join(self.config["categories"]) + "."]
        if self.config.get("hud_profile"):
            command.extend(["--hud-profile", self.config["hud_profile"]])
        with (output / "perception.log").open("w") as log:
            subprocess.run(command, env=env, check=True, stdout=log,
                           stderr=subprocess.STDOUT, timeout=600)
        data = json.loads((output / "result.json").read_text())
        observations = []
        for index, raw in enumerate(data["observations"]):
            features = np.load(raw["embedding_path"])[raw["embedding_index"]]
            observations.append(ObservationObject(
                obs_id=f"{namespace}:f{frame['frame_id']}:o{index}", track_id=None,
                category=raw["category"].strip().rstrip("."), bbox_xyxy=raw["bbox_xyxy"],
                confidence=raw["confidence"], frame_id=frame["frame_id"],
                timestamp=frame["timestamp"], mask=raw["mask_path"],
                visual_embedding=features.tolist(),
                source=source+":GroundingDINO+SAM2.1+DINOv2"))
        assigned = set()
        if self.previous and observations:
            cost = np.ones((len(self.previous), len(observations)))
            for i, old in enumerate(self.previous):
                for j, new in enumerate(observations):
                    if old.category != new.category:
                        continue
                    similarity = float(np.dot(old.visual_embedding, new.visual_embedding))
                    cost[i, j] = 0.5*(1-iou(old.bbox_xyxy, new.bbox_xyxy)) + 0.5*(1-similarity)
            rows, cols = linear_sum_assignment(np.c_[cost, np.full((len(self.previous), len(self.previous)), 0.65)])
            for i, j in zip(rows, cols):
                if j < len(observations) and cost[i, j] < 0.65:
                    observations[j].track_id = self.previous[i].track_id
                    assigned.add(j)
        for j, obj in enumerate(observations):
            if j not in assigned:
                obj.track_id = f"{namespace}:track_{self.next_track:04d}"
                self.next_track += 1
        self.previous = observations
        return observations
