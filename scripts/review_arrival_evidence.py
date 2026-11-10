"""Inspect an arrival rollout with optional agent-selected temporal tools."""
import argparse
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.interaction_evidence import review_arrival
from perception.frame_sampler import sample_frames


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--reference', type=Path)
    p.add_argument('--annotation', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1]/'configs/default.yaml')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    # Prototype clips are short: preserve a true middle sample, including 5.125s clips.
    frames = sample_frames(a.video, a.output/'samples', interval=60.)
    result = review_arrival(yaml.safe_load(a.config.read_text())['agent'], a.video,
        json.loads(a.contract.read_text()), frames, a.output/'agent', a.reference, a.annotation)
    print(json.dumps(result, indent=2))
