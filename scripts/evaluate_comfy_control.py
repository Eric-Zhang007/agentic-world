"""Decode actual outputs and make an aligned visual/edge smoke-test report."""
import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw


def decode(path):
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(stream)]
        return np.stack(frames), float(stream.average_rate)


def main(root):
    control, fps = decode(root / "control_canny.mp4")
    source, _ = decode(root / "source_resampled.mp4")
    clips = {"source": source, "canny": control}
    results = {"fps": fps, "warning": (
        "Edge distances are diagnostics, not semantic correctness or world-state accuracy. "
        "Canny conditions are derived from source RGB, not compiled calibrated world geometry.")}
    for label in ("controlled", "baseline"):
        if not (root / f"{label}.mp4").exists():
            continue
        images, rate = decode(root / f"{label}.mp4")
        if len(images) != len(control) or images.shape[1:] != control.shape[1:]:
            raise ValueError("Output/control dimensions or frame counts differ")
        clips[label] = images
        distances, precision, recalls = [], [], []
        for image, target in zip(images, control):
            edges = cv2.Canny(image, 100, 200) > 0
            truth = target.mean(axis=2) > 127
            if not truth.any() or not edges.any():
                distances.append(None)
                precision.append(0.0)
                recalls.append(0.0)
                continue
            dt_target = cv2.distanceTransform((~truth).astype("uint8"), cv2.DIST_L2, 3)
            dt_image = cv2.distanceTransform((~edges).astype("uint8"), cv2.DIST_L2, 3)
            distances.append(float((dt_target[edges].mean() + dt_image[truth].mean()) / 2))
            precision.append(float((dt_target[edges] <= 3).mean()))
            recalls.append(float((dt_image[truth] <= 3).mean()))
        p, r = float(np.mean(precision)), float(np.mean(recalls))
        finite = [v for v in distances if v is not None]
        results[label] = {
            "frames": len(images), "fps": rate, "width": images.shape[2], "height": images.shape[1],
            "pixel_std": float(images.std()), "unique_frames": int(sum(
                np.any(a != b) for a, b in zip(images, images[1:]))) + 1,
            "symmetric_edge_distance_px": float(np.mean(finite)) if finite else None,
            "edge_precision_3px": p, "edge_recall_3px": r,
            "edge_f1_3px": 2*p*r/(p+r) if p+r else 0.0,
        }
    if "controlled" in clips and "baseline" in clips:
        results["controlled_baseline_pixel_mae"] = float(np.abs(
            clips["controlled"].astype("float32") - clips["baseline"]).mean())
    indices = [0, len(control)//2, len(control)-1]
    w, h = 416, 240
    canvas = Image.new("RGB", (w*3, (h+24)*len(clips)), "#eeeeee")
    draw = ImageDraw.Draw(canvas)
    for row, (label, images) in enumerate(clips.items()):
        for col, index in enumerate(indices):
            y = row*(h+24)
            draw.text((col*w+6, y+5), f"{label} / frame {index}", fill="black")
            canvas.paste(Image.fromarray(images[index]).resize((w, h)), (col*w, y+24))
    canvas.save(root / "comparison.jpg", quality=94)
    (root / "evaluation.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    main(p.parse_args().root)
