"""Small visible-person check; not a benchmark or a camera calibration."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import linear_sum_assignment
from common import save_json
from matching.costs import iou


def evaluate(diagnostic, annotations, mode="joint"):
    results = json.loads(Path(diagnostic).read_text())
    labels = json.loads(Path(annotations).read_text())
    evaluated = []
    for truth in labels["images"]:
        entry = next(r for r in results if Path(r["source"]).name == truth["file"])
        predicted = [r for r in entry["runs"][mode] if r["category"] == "person"]
        expected = truth["objects"]
        overlaps = np.zeros((len(expected), len(predicted)))
        for i, gt in enumerate(expected):
            for j, pred in enumerate(predicted):
                overlaps[i, j] = iou(gt["bbox"], pred["bbox"])
        rows, cols = linear_sum_assignment(1-overlaps)
        pairs = [{"annotation_id": expected[i]["annotation_id"], "iou": float(overlaps[i, j]),
                  "confidence": predicted[j]["confidence"]} for i, j in zip(rows, cols) if overlaps[i, j] >= .5]
        evaluated.append({"file": truth["file"], "tp": len(pairs), "fp": len(predicted)-len(pairs),
                          "fn": len(expected)-len(pairs), "pairs": pairs})
    tp, fp, fn = (sum(r[k] for r in evaluated) for k in ("tp", "fp", "fn"))
    return {"mode": mode, "annotation_source": labels["annotation_source"], "person_only": True,
            "images": evaluated, "tp": tp, "fp": fp, "fn": fn,
            "precision": tp/max(1, tp+fp), "recall": tp/max(1, tp+fn),
            "mean_iou": float(np.mean([p["iou"] for r in evaluated for p in r["pairs"]])) if tp else None,
            "limitation": "Two inspected frames, approximate boxes; no held-out set or 3D ground truth."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic", required=True)
    parser.add_argument("--annotations", default="configs/gta_person_annotations.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = [evaluate(args.diagnostic, args.annotations, mode) for mode in ("joint", "reversed", "single_person")]
    save_json(args.output, result)
    print(json.dumps(result, indent=2))
