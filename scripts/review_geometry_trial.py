"""Build visual and projection evidence for the RGB-only geometry trial."""
import argparse
from itertools import product
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1]/'outputs/geometry_init/urban')
    args = p.parse_args(); root = args.root
    scene = json.loads((root/'coarse_assets/scene.json').read_text())
    report = json.loads((root/'coarse_assets/report.json').read_text())
    with np.load(root/'reconstruction/single.npz') as d:
        image = (d['images'][0].transpose(1, 2, 0)*255).astype('uint8')
        depth = d['depth'][0, :, :, 0]
        intrinsic, extrinsic = d['intrinsic'][0], d['extrinsic'][0]
    masks = np.load(root/'coarse_assets/masks.npz')
    matrix = np.linalg.inv(np.asarray(report['model_to_scene']))
    overlay = Image.fromarray(image); draw = ImageDraw.Draw(overlay)
    metrics = []
    for entity in scene['entities']:
        position, size = np.asarray(entity['position']), np.asarray(entity['size'])
        corners = np.asarray(list(product(*zip(-size/2, size/2))))
        if entity['kind'] == 'actor':
            corners[:, 1] += size[1]/2
        c, s = np.cos(entity['rotation_y']), np.sin(entity['rotation_y'])
        rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        world = corners @ rotation.T + position
        model = world @ matrix[:3, :3].T + matrix[:3, 3]
        camera = model @ extrinsic[:, :3].T + extrinsic[:, 3]
        if (camera[:, 2] <= 0).any():
            metrics.append({'id': entity['id'], 'status': 'crosses_camera_plane'})
            continue
        projected = camera @ intrinsic.T
        uv = projected[:, :2]/projected[:, 2:]
        lo, hi = uv.min(axis=0), uv.max(axis=0)
        yy, xx = np.where(masks[entity['id']])
        target_lo, target_hi = np.array([xx.min(), yy.min()]), np.array([xx.max()+1, yy.max()+1])
        intersection = np.maximum(0, np.minimum(hi, target_hi)-np.maximum(lo, target_lo)).prod()
        union = (hi-lo).prod()+(target_hi-target_lo).prod()-intersection
        row = {'id': entity['id'], 'category': entity['category'], 'status': 'projected',
               'bbox_iou_with_sam': float(intersection/union),
               'center_error_processed_px': float(np.linalg.norm((lo+hi-target_lo-target_hi)/2)),
               'expected_bbox_processed_px': [*lo.tolist(), *hi.tolist()],
               'sam_bbox_processed_px': [*target_lo.tolist(), *target_hi.tolist()]}
        metrics.append(row)
        draw.rectangle([*lo.tolist(), *hi.tolist()], outline='#ffca4b', width=1)
        draw.text(tuple(lo), entity['id'], fill='white')
    overlay.save(root/'evaluation/projected_assets.png')
    depth_vis = np.log(np.maximum(depth, 1e-8))
    lo, hi = np.quantile(depth_vis, [.02, .95])
    pixels = ((depth_vis-lo)/(hi-lo)*255).clip(0, 255).astype('uint8')
    Image.fromarray(cv2.cvtColor(cv2.applyColorMap(255-pixels, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)).save(root/'evaluation/predicted_depth.png')
    panels = [
        ('RGB input (no camera / geometry supplied)', root/'coarse_assets/input.png'),
        ('VGGT single-image depth (relative scale)', root/'evaluation/predicted_depth.png'),
        ('GroundingDINO + SAM2 segmentation', root/'perception_target/overlay.png'),
        ('Dataset proxy (visible mismatch; NOT aligned GT)', root/'evaluation/target_whitebox.png'),
        ('Coarse asset proposal (unknown areas unfilled)', root/'coarse_assets/camera.png'),
        ('Projected candidate bounds on RGB', root/'evaluation/projected_assets.png'),
    ]
    sheet = Image.new('RGB', (1554, 648), '#202326'); draw = ImageDraw.Draw(sheet)
    for i, (label, path) in enumerate(panels):
        im = Image.open(path).convert('RGB'); im.thumbnail((518, 294))
        x, y = (i % 3)*518, (i//3)*324
        sheet.paste(im, (x+(518-im.width)//2, y))
        draw.text((x+8, y+300), label, fill='white')
    sheet.save(root/'comparison.jpg', quality=95)
    more = [('Original target RGB', root/'coarse_assets/input.png'),
            ('Single RGB -> coarse assets', root/'coarse_assets/camera.png'),
            ('Three RGB views -> coarse assets', root/'multiview_assets/camera.png'),
            ('Historical RGB view', root/'input/reference_rgb.png'),
            ('Frozen single-view assets, transferred camera', root/'novel_view/camera.png'),
            ('Single vs multi-view anchor alignment', root/'alignment/before_after_error.png')]
    board = Image.new('RGB', (1554, 648), '#202326'); pen = ImageDraw.Draw(board)
    for i, (label, path) in enumerate(more):
        im = Image.open(path).convert('RGB'); im.thumbnail((518, 294))
        x, y = i%3*518, i//3*324
        board.paste(im, (x+(518-im.width)//2, y+(294-im.height)//2))
        pen.text((x+8, y+300), label, fill='white')
    board.save(root/'multiview_comparison.jpg', quality=95)
    result = {'scope': 'Projection consistency against SAM, not independent geometry truth',
              'image_size': [518, 294], 'objects': metrics,
              'median_bbox_iou': float(np.median([m['bbox_iou_with_sam'] for m in metrics if m['status']=='projected'])),
              'median_center_error_px': float(np.median([m['center_error_processed_px'] for m in metrics if m['status']=='projected']))}
    (root/'evaluation/asset_projection.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'objects'}, indent=2))


if __name__ == '__main__':
    main()
