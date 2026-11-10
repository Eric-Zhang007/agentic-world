"""Preserve old scene IDs and compare incoming reconstruction region by region."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from whitebox.geometry_local import assess_region
from scripts.evaluate_geometry_alignment import replay_preprocessing


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1]/'outputs/geometry_init/urban')
    p.add_argument('--registration', default='alignment_three_frames')
    p.add_argument('--old', default='reconstruction/single.npz')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); root = args.root
    if args.output.exists():
        raise ValueError('Choose new output directory')
    with np.load(root/args.old) as data:
        old = data['points'][0]
        rgb = data['images'][0].transpose(1, 2, 0)
        record = json.loads(str(data['metadata']))['preprocessing']['images'][0]
    with np.load(root/args.registration/'registration.npz') as reg:
        if not reg['accepted'].item():
            raise ValueError('Global registration did not pass')
        incoming = reg['aligned_joint_points'][int(reg['anchor_index'])]
        eligible = reg['eligible']
        registration_report = json.loads(str(reg['report_json']))
    threshold = registration_report['thresholds']['inlier_threshold']
    perception = json.loads((root/'perception_target/result.json').read_text())
    scene_path = root/'coarse_assets/scene.json'
    scene_bytes = scene_path.read_bytes()
    ids = {e['id'] for e in json.loads(scene_bytes)['entities']}
    overlay = (rgb*255).astype('uint8')
    colors = {'CONSISTENT_KEEP_OLD': [40, 210, 100], 'LOCAL_OFFSET_REVIEW': [240, 200, 40],
              'SHAPE_CONFLICT_KEEP_OLD': [240, 70, 70], 'INSUFFICIENT': [150, 150, 150]}
    rows = []
    for i, observation in enumerate(perception['observations']):
        category = observation['category'].strip().split()
        if set(category) & {'person', 'car'}:
            continue
        mask_path = root/'perception_target'/Path(observation['mask_path'].replace('\\', '/')).name
        with Image.open(mask_path) as mask:
            region = replay_preprocessing(mask.convert('L'), record, Image.Resampling.NEAREST) > 0
        region &= eligible
        result = assess_region(old[region], incoming[region], threshold=threshold)
        result.update(source_mask_id=f'geo_{i:03d}', category=observation['category'],
                      old_entity_id=f'geo_{i:03d}' if f'geo_{i:03d}' in ids else None)
        rows.append(result)
        overlay[region] = (.55*overlay[region]+.45*np.asarray(colors[result['status']])).astype('uint8')
    args.output.mkdir(parents=True)
    im = Image.fromarray(overlay); im = im.resize((1036, 588))
    board = Image.new('RGB', (1036, 640), '#202326'); board.paste(im, (0, 0))
    draw = ImageDraw.Draw(board)
    draw.text((10, 596), 'Green: keep old | Yellow: candidate local offset, review | Red: shape conflict, keep old', fill='white')
    draw.text((10, 616), 'No proposed offsets are applied. No world history is modified.', fill='white')
    board.save(args.output/'regional_review.png')
    payload = {'counts': dict(Counter(r['status'] for r in rows)), 'regions': rows,
               'source_scene': str(scene_path), 'world_modified': False,
               'registration': args.registration, 'global_threshold_retained': threshold,
               'scope': 'Local translation diagnostic with held-out anchor pixels; no hidden surface completion',
               'world_merge_ready': False}
    (args.output/'report.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    assert scene_bytes == scene_path.read_bytes()
    print(json.dumps(payload['counts']))


if __name__ == '__main__':
    main()
