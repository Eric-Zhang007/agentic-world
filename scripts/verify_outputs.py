"""Read actual output files and verify media, persistence and model evidence."""
import argparse
import json
from pathlib import Path

import av
import numpy as np


def verify(root):
    root = Path(root)
    status = json.loads((root / "status.json").read_text())
    before = json.loads((root / "world_before.json").read_text())
    after = json.loads((root / "world_after.json").read_text())
    before_ids = {o["object_id"] for o in before["objects"]}
    after_ids = {o["object_id"] for o in after["objects"]}
    assert before_ids == after_ids
    before_facts = [{k: v for k, v in o.items() if k != "appearance_embedding"} for o in before["objects"]]
    after_facts = [{k: v for k, v in o.items() if k != "appearance_embedding"} for o in after["objects"]]
    attempts = []
    for directory in sorted(root.glob("attempt_*")):
        decision = json.loads((directory / "agent_decision.json").read_text())
        report = json.loads((directory / "discrepancy.json").read_text())
        metadata_path = directory / "generation_metadata.json"
        request_path = directory / "generation_request.json"
        record = {"attempt": directory.name, "decision": decision["action"],
                  "summary": report["summary"]}
        if request_path.exists():
            request = json.loads(request_path.read_text())
            record.update(seed=request["seed"], steps=request["steps"])
        video = directory / "generated.mp4"
        if video.exists():
            with av.open(str(video)) as container:
                stream = container.streams.video[0]
                fps = float(stream.average_rate)
                frames = list(container.decode(stream))
                assert len(frames) == 72 and fps == 24
                sample = np.asarray(frames[len(frames)//2].to_image())
                assert sample.std() > 1
                record.update(frames=len(frames), fps=fps,
                              nonblank_std=float(sample.std()))
        if metadata_path.exists():
            record["generation_seconds"] = json.loads(metadata_path.read_text())["seconds"]
        observations = json.loads((directory / "observations.json").read_text())
        assert len(observations) >= 3
        for entry in observations:
            for obj in entry["objects"]:
                assert obj["source"].startswith("GENERATED:GroundingDINO")
                assert Path(obj["mask"]).exists()
                assert len(obj["visual_embedding"]) == 384
                assert np.isfinite(obj["visual_embedding"]).all()
        assert all(o["category"] != "dog" for o in after["objects"])
        attempts.append(record)
    if len(attempts) > 1:
        seeds = [a["seed"] for a in attempts]
        assert len(seeds) == len(set(seeds))
    assert not (status["retry_exhausted"] and status["accepted"])
    result = {"success": True, "status": status, "world_ids_unchanged": True,
              "world_snapshot_identical": before == after,
              "authoritative_object_facts_unchanged": before_facts == after_facts,
              "attempts": attempts}
    (root / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    verify(args.root)
