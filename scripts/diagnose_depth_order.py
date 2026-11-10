"""Local urban diagnostic; no inference or world writes."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perception.depth_order import compare_depth_order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nearer-is-larger', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    out = ROOT / 'outputs/local_depth_order_20261109'
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'outputs/gta_pi_loop_20261102/urban/input.png'
    depth_path = ROOT / 'outputs/local_fresh_depth_20261109/00_urban/relative_depth.npy'
    scene_path = ROOT / 'outputs/local_landmark_fit_20261109/urban/scene.json'
    # Visually selected interior patches in the 1280 x 720 source RGB.
    regions = {'player': [530, 365, 580, 440], 'npc': [775, 335, 795, 355],
               'hydrant': [393, 430, 408, 460], 'lamp_base': [275, 450, 320, 475]}
    depth = np.load(depth_path, allow_pickle=False)
    image = Image.open(source).convert('RGB')
    if image.size != (depth.shape[1], depth.shape[0]):
        raise ValueError('Source and depth dimensions must match; no implicit resizing')
    report = compare_depth_order(depth, json.loads(scene_path.read_text()), regions,
                                 nearer_is_larger=args.nearer_is_larger)
    report.update(source=str(source), depth=str(depth_path), scene=str(scene_path),
                  annotation='Manual interior rectangles from visual RGB inspection, not object masks or detections',
                  boxes=regions)
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (out / 'annotations.json').write_text(json.dumps({'image_size': image.size, 'regions': regions,
        'evidence': report['annotation']}, indent=2) + '\n')
    draw = ImageDraw.Draw(image)
    for name, box in regions.items():
        draw.rectangle([box[0], box[1], box[2]-1, box[3]-1], outline='magenta', width=2)
        draw.text((box[0], box[1]-14), name, fill='magenta', stroke_width=1, stroke_fill='black')
    image.save(out / 'manual_regions.png')
    print(json.dumps(report['pairs'], indent=2))


if __name__ == '__main__':
    main()
