"""Offline joint -> single VGGT registration using positive SAM static regions.

Run: python -B scripts/evaluate_geometry_alignment.py
Exit codes: 0 registration accepted, 2 geometric rejection, 1 invalid/missing inputs.
Registration acceptance does not imply world merge readiness.
No model inference, network access, DB writes, or coarse-world modification.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from whitebox.geometry_alignment import AlignmentError, fit_anchor_sim3

STATIC_CLASSES = frozenset(('building', 'road', 'sidewalk', 'tree', 'pole', 'wall'))
DYNAMIC_CLASSES = frozenset(('person', 'car'))
# Fixed before fitting; normalized rectangles cover all four HUD corners.
HUD_RECTANGLES = ((0., 0., .12, .065), (.87, 0., 1., .065),
                  (0., .765, .195, 1.), (.88, .93, 1., 1.))
BASE = ROOT / 'outputs/geometry_init/urban'


def load_reconstruction(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in
                ('points', 'depth', 'depth_conf', 'images', 'metadata')}
    data['metadata'] = json.loads(str(data['metadata'].item()))
    points = data['points']
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError('points must have shape [N,H,W,3]')
    n, h, w, _ = points.shape
    expected = {'depth': (n, h, w, 1), 'depth_conf': (n, h, w), 'images': (n, 3, h, w)}
    for key, shape in expected.items():
        if data[key].shape != shape:
            raise ValueError(f'{key} has incompatible shape: {data[key].shape}')
    if not np.isfinite(data['images']).all() or np.any((data['images'] < 0) | (data['images'] > 1)):
        raise ValueError('images must be finite RGB in [0,1]')
    return data


def replay_preprocessing(image, record, resampling):
    """Replay metadata resize/crop/pad; nearest neighbor for categorical masks."""
    if list(image.size) != record['original_wh']:
        raise ValueError('SAM original image/mask dimensions do not match metadata')
    rw, rh = record['resized_wh']
    ow, oh = record['original_wh']
    width, height = record['output_wh']
    left, top, right, bottom = record['padding_ltrb']
    crop_top = record['crop_top']
    ch = height - top - bottom
    if (min(left, top, right, bottom, crop_top) < 0 or ch <= 0
            or rw + left + right != width or crop_top + ch > rh):
        raise ValueError('Inconsistent preprocessing crop/padding metadata')
    sx, sy = rw / ow, rh / oh
    expected = [[sx, 0, (sx - 1) / 2 + left],
                [0, sy, (sy - 1) / 2 + top - crop_top], [0, 0, 1]]
    if not np.allclose(record['original_to_processed_pixel_centers'], expected, atol=1e-12, rtol=0):
        raise ValueError('Pixel-center warp disagrees with resize/crop/pad metadata')
    resized = image.resize((rw, rh), resampling)
    cropped = resized.crop((0, crop_top, rw, crop_top + ch))
    canvas = Image.new(image.mode, (width, height), (255, 255, 255) if image.mode == 'RGB' else 0)
    canvas.paste(cropped, (left, top))
    return np.asarray(canvas)


def check_anchor_rgb(single, joint, index, perception_path, perception):
    old_rgb, new_rgb = single['images'][0], joint['images'][index]
    if old_rgb.shape != new_rgb.shape or not np.array_equal(old_rgb, new_rgb):
        raise ValueError('Anchor RGB mismatch: single[0] and joint[anchor-index] must be identical')
    old_record = single['metadata']['preprocessing']['images'][0]
    new_record = joint['metadata']['preprocessing']['images'][index]
    for key in ('original_wh', 'resized_wh', 'crop_top', 'padding_ltrb',
                'output_wh', 'original_to_processed_pixel_centers'):
        if old_record[key] != new_record[key]:
            raise ValueError('Anchor preprocessing metadata mismatch: ' + key)
    if old_record['output_wh'] != [old_rgb.shape[2], old_rgb.shape[1]]:
        raise ValueError('Metadata output size differs from reconstruction')
    if perception['image_size'] != old_record['original_wh']:
        raise ValueError('Perception image_size differs from anchor original size')
    image_path = perception_path.parent / Path(perception['image'].replace('\\', '/')).name
    with Image.open(image_path) as image:
        processed = replay_preprocessing(image.convert('RGB'), old_record, Image.Resampling.BICUBIC)
    expected = np.rint(old_rgb.transpose(1, 2, 0) * 255).astype(np.uint8)
    if not np.array_equal(processed, expected):
        raise ValueError('SAM source RGB does not reproduce the reconstruction anchor RGB')
    return old_record, dict(single_joint_exact=True, sam_processed_exact=True,
                            max_single_joint_difference=0., sam_image=str(image_path.resolve()))


def build_static_mask(perception_path, perception, record):
    width, height = record['original_wh']
    positive = np.zeros((height, width), dtype=bool)
    dynamic = np.zeros_like(positive)
    observations = []
    for observation in perception['observations']:
        # Composite labels such as 'building wall' are accepted only when all
        # words are approved static classes. Dynamic labels always take priority.
        words = set(observation['category'].strip().lower().split())
        kind = 'dynamic' if words & DYNAMIC_CLASSES else (
            'static' if words and words <= STATIC_CLASSES else 'ignored')
        if kind == 'ignored':
            continue
        path = perception_path.parent / Path(observation['mask_path'].replace('\\', '/')).name
        with Image.open(path) as image:
            mask = np.asarray(image.convert('L')) > 0
        if mask.shape != positive.shape:
            raise ValueError('SAM mask is not in the original image grid: ' + str(path))
        if kind == 'dynamic':
            dynamic |= mask
        else:
            positive |= mask
        observations.append(dict(id=observation.get('observation_id', path.stem),
                                 category=observation['category'], kind=kind,
                                 mask=str(path.resolve()), pixels=int(mask.sum())))
    hud = np.zeros_like(positive)
    for x0, y0, x1, y1 in HUD_RECTANGLES:
        hud[int(y0 * height):int(np.ceil(y1 * height)),
            int(x0 * width):int(np.ceil(x1 * width))] = True
    def warp(mask):
        return replay_preprocessing(Image.fromarray(mask.astype(np.uint8) * 255),
                                    record, Image.Resampling.NEAREST) > 0
    positive, dynamic, hud = (warp(mask) for mask in (positive, dynamic, hud))
    static = positive & ~dynamic & ~hud
    if not static.any():
        raise ValueError('No positive segmented static support remains')
    return static, dynamic, hud, dict(
        policy='union(positive approved static masks) minus all person/car masks minus HUD',
        approved_classes=sorted(STATIC_CLASSES), dynamic_classes=sorted(DYNAMIC_CLASSES),
        hud_rectangles_normalized=HUD_RECTANGLES, warp='metadata resize/crop/pad, nearest neighbor',
        positive_pixels=int(positive.sum()), dynamic_pixels=int(dynamic.sum()),
        hud_pixels=int(hud.sum()), static_pixels=int(static.sum()), observations=observations)


def error_statistics(error, mask):
    values = error[mask]
    if not len(values):
        return {'count': 0}
    return dict(count=int(len(values)), rmse=float(np.sqrt(np.mean(values ** 2))),
                median=float(np.median(values)), p95=float(np.percentile(values, 95)),
                maximum=float(values.max()))


def residual_breakdown(depth, eligible, inliers, before, after, median_depth,
                       threshold, accepted, mask_report, record):
    """All-eligible diagnostics, with overlapping SAM instances kept separate.

    Bins use old/single depth only, independently of residual or fit acceptance.
    Relative errors are per-pixel error / single camera-z depth, dimensionless.
    """
    def summary(mask):
        support = mask & eligible
        count = int(support.sum())
        result = dict(eligible_count=count, inlier_count=int((support & inliers).sum()),
                      inlier_ratio=float(inliers[support].mean()) if count else None,
                      median_depth=float(np.median(depth[support])) if count else None,
                      before=error_statistics(before, support), after=None,
                      after_error_over_pixel_depth=None,
                      after_p95_over_global_median_depth=None,
                      after_fraction_above_inlier_threshold=None)
        if accepted:
            result['after'] = error_statistics(after, support)
            relative = np.zeros_like(after)
            np.divide(after, depth, out=relative, where=support)
            result['after_error_over_pixel_depth'] = error_statistics(relative, support)
            if count:
                result['after_p95_over_global_median_depth'] = result['after']['p95'] / median_depth
                result['after_fraction_above_inlier_threshold'] = float((after[support] > threshold).mean())
        return result

    bins = dict(near=depth <= median_depth,
                mid=(depth > median_depth) & (depth <= 2 * median_depth),
                far=depth > 2 * median_depth)
    instances, categories = [], {}
    for observation in mask_report['observations']:
        if observation['kind'] != 'static':
            continue
        with Image.open(observation['mask']) as image:
            mask = replay_preprocessing(image.convert('L'), record, Image.Resampling.NEAREST) > 0
        category = observation['category'].strip().lower()
        categories.setdefault(category, np.zeros_like(eligible))
        categories[category] |= mask
        instances.append(dict(id=observation['id'], category=category, mask=observation['mask'],
                              processed_mask_pixels=int(mask.sum()), **summary(mask)))
    return dict(
        population='All eligible static pixels, including RANSAC outliers; dynamic/HUD excluded',
        depth_bin_policy='near: 0 < z <= m; mid: m < z <= 2m; far: z > 2m; m=single static median depth',
        depth_bin_boundaries=[median_depth, 2 * median_depth],
        relative_error_definition='Euclidean 3D error / single camera-z depth at each pixel; dimensionless',
        instance_overlap_policy='Overlapping masks counted in each instance; category union deduplicates pixels within each exact label',
        all_eligible=summary(eligible),
        depth_bins={name: summary(mask) for name, mask in bins.items()},
        static_instances=instances,
        static_categories={name: summary(mask) for name, mask in sorted(categories.items())})


def save_error_figure(path, rgb, static, eligible, inliers, before, after, threshold, accepted, anchor_index):
    import cv2
    from PIL import Image, ImageDraw
    mask_view = np.zeros((*static.shape, 3), dtype=np.uint8)
    mask_view[static] = [55, 100, 175]
    mask_view[inliers] = [50, 195, 90]
    maximum = max(threshold, float(np.percentile(before[eligible], 95)))
    if accepted:
        maximum = max(maximum, float(np.percentile(after[eligible], 95)))
    panels = [(f'Target RGB: single[0] == joint[{anchor_index}]', (rgb*255).astype('uint8')),
              ('Static: blue; registration inliers: green', mask_view)]
    for error, title in ((before, 'Before: all eligible static'), (after, 'After: all eligible static')):
        valid = eligible & np.isfinite(error)
        pixels = (np.nan_to_num(error)/max(maximum, 1e-12)*255).clip(0, 255).astype('uint8')
        heat = cv2.cvtColor(cv2.applyColorMap(pixels, cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)
        heat[~valid] = 208
        panels.append((title, heat))
    board = Image.new('RGB', (1040, 730), '#202326'); draw = ImageDraw.Draw(board)
    draw.text((10, 8), f'Registration {"ACCEPTED" if accepted else "REJECTED"}; world merge NOT READY; threshold {threshold:.6g}', fill='white')
    for i, (title, pixels) in enumerate(panels):
        im = Image.fromarray(pixels); im.thumbnail((518, 294))
        x, y = i%2*520, 40+i//2*330
        board.paste(im, (x, y)); draw.text((x+5, y+299), title, fill='white')
    draw.text((10, 710), f'Both heatmaps: black=0, pale yellow>={maximum:.5g}; model units, not meters; gray=excluded', fill='white')
    board.save(path)


def evaluate(args):
    single, joint = load_reconstruction(args.single), load_reconstruction(args.joint)
    if len(single['points']) < 1 or len(joint['points']) < 2:
        raise ValueError('Expected old reconstruction with anchor at index 0 and incoming >=2 frames')
    if not 0 <= args.anchor_index < len(joint['points']):
        raise ValueError('anchor-index must be within joint frame range')
    perception = json.loads(args.perception.read_text(encoding='utf-8'))
    if perception.get('success') is not True:
        raise ValueError('SAM result is not successful')
    record, rgb_check = check_anchor_rgb(single, joint, args.anchor_index, args.perception, perception)
    static, dynamic, hud, mask_report = build_static_mask(args.perception, perception, record)
    old, new = single['points'][0], joint['points'][args.anchor_index]
    old_conf, new_conf = single['depth_conf'][0], joint['depth_conf'][args.anchor_index]
    depth = single['depth'][0, ..., 0]
    depth_valid = static & np.isfinite(depth) & (depth > 0)
    if not depth_valid.any():
        raise ValueError('No finite positive single static depth')
    median_depth = float(np.median(depth[depth_valid]))
    threshold, max_rmse = median_depth * .05, median_depth * .03
    eligible = (depth_valid & np.isfinite(old).all(-1) & np.isfinite(new).all(-1)
                & np.isfinite(old_conf) & np.isfinite(new_conf) & (old_conf > 0) & (new_conf > 0))
    if not eligible.any():
        raise ValueError('No eligible finite static correspondences')
    report = dict(schema_version=2, created_utc=datetime.now(timezone.utc).isoformat(),
                  inputs={k: str(getattr(args, k).resolve()) for k in ('single', 'joint', 'perception')},
                  direction='joint model world -> single model world', anchor_index=args.anchor_index,
                  joint_frame_count=len(joint['points']),
                  old_frame_count=len(single['points']),
                  world_merge_ready=False,
                  world_merge_reason='Local geometry has not been validated. A global Sim3 inlier consensus does not establish far-scene alignment or authorize world merging.',
                  local_validation_status='not_performed',
                  units='single VGGT predicted-scale world units; not calibrated meters',
                  anchor_rgb=rgb_check, static_mask=mask_report,
                  thresholds=dict(median_single_static_depth=median_depth,
                                  median_depth_pixel_count=int(depth_valid.sum()),
                                  inlier_threshold=threshold, max_rmse=max_rmse,
                                  inlier_factor=.05, rmse_factor=.03,
                                  min_inliers=20, min_inlier_ratio=.5,
                                  ransac_iterations=512, seed=0),
                  scope='Same-anchor internal consistency, not held-out reference or metric-world accuracy')
    before = np.linalg.norm(new.astype(float) - old, axis=-1)
    after = np.full(static.shape, np.nan)
    inliers = np.zeros_like(static)
    arrays = dict(static_mask=static, dynamic_mask=dynamic, hud_mask=hud, eligible=eligible,
                  before_error=before, single_anchor=old, joint_anchor=new)
    try:
        result = fit_anchor_sim3(new, old, valid_static_mask=depth_valid,
                                new_confidence=new_conf, old_confidence=old_conf,
                                inlier_threshold=threshold, max_rmse=max_rmse)
    except AlignmentError as exc:
        report.update(status='rejected', reason=str(exc), fit_metrics=exc.metrics)
        accepted = False
    else:
        accepted = True
        inliers = result.inliers
        aligned = result.apply(joint['points'])
        after = np.linalg.norm(aligned[args.anchor_index] - old, axis=-1)
        arrays.update(transform=result.transform, aligned_joint_points=aligned)
        report.update(status='accepted', fit_metrics=result.metrics,
                      transform=result.transform.tolist())
    report['errors'] = dict(before_all_eligible=error_statistics(before, eligible),
                            after_all_eligible=error_statistics(after, eligible) if accepted else None,
                            before_final_inliers=error_statistics(before, inliers),
                            after_final_inliers=error_statistics(after, inliers))
    report['registration_accepted'] = accepted
    report['residual_diagnostics'] = residual_breakdown(
        depth, eligible, inliers, before, after, median_depth, threshold,
        accepted, mask_report, record)
    report['registration_contract'] = (
        'accepted scalar bool is registration-only; world_merge_ready is false pending local validation; '
        'transform [4,4] and aligned_joint_points [N,H,W,3] exist only on acceptance; '
        'inliers/static_mask/dynamic_mask/hud_mask/eligible [H,W] bool; errors [H,W] in single units; '
        'after_error is NaN when rejected. No camera or coarse-world update.')
    args.output.mkdir(parents=True, exist_ok=True)
    arrays.update(accepted=np.asarray(accepted), world_merge_ready=np.asarray(False),
                  anchor_index=np.asarray(args.anchor_index), single_depth=depth,
                  inliers=inliers, after_error=after,
                  report_json=np.asarray(json.dumps(report, allow_nan=False)))
    np.savez_compressed(args.output / 'registration.npz', **arrays)
    save_error_figure(args.output / 'before_after_error.png',
                      single['images'][0].transpose(1, 2, 0), static, eligible,
                      inliers, before, after, threshold, accepted, args.anchor_index)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], output=str(args.output.resolve()),
                         world_merge_ready=False,
                         thresholds=report['thresholds'], fit_metrics=report['fit_metrics'],
                         errors=report['errors']), indent=2))
    return 0 if accepted else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--single', '--old', dest='single', type=Path, default=BASE / 'reconstruction/single.npz',
                        help='Old reconstruction, with common anchor at index 0; may contain multiple frames')
    parser.add_argument('--joint', type=Path, default=BASE / 'reconstruction/joint.npz')
    parser.add_argument('--perception', type=Path, default=BASE / 'perception_target/result.json')
    parser.add_argument('--anchor-index', type=int, default=1,
                        help='Joint frame matching single[0], zero-based (e.g. 0 for target-first joint3)')
    parser.add_argument('--output', type=Path, default=BASE / 'alignment')
    return evaluate(parser.parse_args(argv))


if __name__ == '__main__':
    sys.exit(main())
