"""Conservative dynamic-occlusion ground completion, without GT or compiler writes.

API: complete_ground(points, masks, extrinsic, intrinsic, model_to_scene,
                     valid=None, config=GroundConfig()) -> GroundCompletion.
Points are HxWx3 model coordinates; masks are processed-image boolean arrays.
Extrinsic maps model to camera (OpenCV); scene uses Y up. All geometric limits
use scene units, so callers must supply a meaningful model_to_scene scale.
Records may contain mask (eroded/confidence-filtered support) and raw_mask
(complete SAM silhouette). Non-ground categories use raw_mask when provided.
Pass the compiler finite bitmap as valid=finite. Without raw_mask, non-ground
mask must be complete and must NOT be confidence-filtered.
Run: python -m whitebox.geometry_ground --urban outputs/geometry_init/urban
"""
from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class GroundConfig:
    max_gap_px: int = 72
    silhouette_margin_px: int = 5
    support_radius_px: int = 7
    min_support: int = 30
    max_residual: float = .08
    max_slope_deg: float = 18.
    max_extent: float = 5.
    max_area: float = 5.
    max_boundary_jump: float = .25


@dataclass
class GroundCompletion:
    points: np.ndarray
    visible_mask: np.ndarray
    inferred_mask: np.ndarray
    evidence: list

    def report(self):
        """JSON-ready coverage and per-region decision evidence."""
        return dict(visible_pixels=int(self.visible_mask.sum()),
                    inferred_pixels=int(self.inferred_mask.sum()),
                    accepted_regions=sum(e['status'] == 'INFERRED' for e in self.evidence),
                    evidence=self.evidence)


def complete_ground(points, masks, extrinsic, intrinsic, model_to_scene,
                    *, valid=None, config=GroundConfig()):
    """Return scene points and disjoint visible/INFERRED masks; inputs unchanged.

    Unknown pixels beyond the bounded silhouette margin are never candidates. Rows
    require ground support on both sides, with no intervening unknown/static
    pixels outside the configured silhouette margin. Each connected candidate
    fits its own local ground plane. Rejected
    regions retain their original points and have a reason in evidence.
    """
    p = np.asarray(points, float)
    if p.ndim != 3 or p.shape[-1] != 3:
        raise ValueError('points must be HxWx3')
    h, w = p.shape[:2]
    matrix = np.asarray(model_to_scene, float)
    ext, intr = np.asarray(extrinsic, float), np.asarray(intrinsic, float)
    if matrix.shape != (4, 4) or ext.shape != (3, 4) or intr.shape != (3, 3):
        raise ValueError('Expected 4x4 scene, 3x4 extrinsic and 3x3 intrinsic matrices')
    if not all(np.isfinite(a).all() for a in (matrix, ext, intr)):
        raise ValueError('Camera and scene matrices must be finite')
    if any(not np.isfinite(v) or v <= 0 for v in asdict(config).values()):
        raise ValueError('Completion limits must be finite and positive')
    finite = np.isfinite(p).all(axis=2)
    if valid is not None:
        if np.shape(valid) != (h, w):
            raise ValueError('valid shape differs from point map')
        finite &= np.asarray(valid, bool)
    ground = {}
    blocked = np.zeros((h, w), bool)
    dynamic = []
    for i, item in enumerate(masks):
        cat = item['category'].strip().rstrip('.')
        key = 'raw_mask' if cat not in ('road', 'ground', 'sidewalk') and 'raw_mask' in item else 'mask'
        mask = np.asarray(item[key], bool)
        if mask.shape != (h, w):
            raise ValueError('mask shape differs from point map')
        if cat in ('road', 'ground', 'sidewalk'):
            ground[cat] = ground.get(cat, np.zeros((h, w), bool)) | mask
        elif cat in ('person', 'car'):
            dynamic.append((str(item.get('id', i)), cat, mask))
        else:
            blocked |= mask
    occupied = blocked.copy()
    for _, _, mask in dynamic:
        occupied |= mask
    ground = {k: v & finite & ~occupied for k, v in ground.items()}
    visible = np.zeros((h, w), bool)
    for mask in ground.values():
        visible |= mask
    world = p @ matrix[:3, :3].T + matrix[:3, 3]
    inferred = np.zeros((h, w), bool)
    evidence = []
    camera = -np.linalg.solve(ext[:, :3], ext[:, 3])
    origin = matrix[:3, :3] @ camera + matrix[:3, 3]
    yy, xx = np.indices((h, w))
    rays = np.stack((xx, yy, np.ones_like(xx)), -1) @ np.linalg.inv(intr).T
    rays = rays @ np.linalg.inv(ext[:, :3]).T @ matrix[:3, :3].T
    for dynamic_index, (entity, category, silhouette) in enumerate(dynamic):
        evidence_start = len(evidence)
        margin = config.silhouette_margin_px
        envelope = cv2.dilate(silhouette.astype('uint8'), np.ones((2*margin+1, 2*margin+1), 'uint8')) > 0
        other_dynamic = np.zeros((h, w), bool)
        for other_index, (_, _, other_mask) in enumerate(dynamic):
            if other_index != dynamic_index:
                other_dynamic |= other_mask
        allowed = envelope & ~blocked & ~visible & ~inferred & ~other_dynamic
        for surface, support in ground.items():
            candidate = np.zeros((h, w), np.uint8)
            for y in np.flatnonzero(allowed.any(axis=1)):
                xs = np.flatnonzero(support[y])
                for left, right in zip(xs[:-1], xs[1:]):
                    if 1 < right-left <= config.max_gap_px and allowed[y, left+1:right].all():
                        candidate[y, left+1:right] = 1
            count, labels = cv2.connectedComponents(candidate, connectivity=8)
            for label in range(1, count):
                region = labels == label
                if not (region & silhouette).any():
                    continue
                rec = dict(entity=entity, category=category, surface=surface,
                           pixels=int(region.sum()), silhouette_pixels=int((region & silhouette).sum()),
                           margin_pixels=int((region & ~silhouette).sum()), status='REJECTED')
                region_y, region_x = np.where(region)
                rec['bbox_xyxy'] = [int(region_x.min()), int(region_y.min()),
                                    int(region_x.max()+1), int(region_y.max()+1)]
                evidence.append(rec)
                if not (region[:-1, :-1] & region[1:, :-1] & region[:-1, 1:] & region[1:, 1:]).any():
                    rec['reason'] = 'no_surface_cells'
                    continue
                radius = config.support_radius_px
                ring = cv2.dilate(region.astype('uint8'), np.ones((2*radius+1, 2*radius+1), 'uint8')) > 0
                local = ring & support
                cloud = world[local]
                rec['support_pixels'] = len(cloud)
                if len(cloud) < config.min_support:
                    rec['reason'] = 'insufficient_support'
                    continue
                center = cloud.mean(0)
                _, singular, vt = np.linalg.svd(cloud-center, full_matrices=False)
                normal = vt[-1]
                if normal[1] < 0:
                    normal = -normal
                residual = np.abs((cloud-center) @ normal)
                slope = float(np.degrees(np.arccos(np.clip(normal[1], -1, 1))))
                rec.update(plane_normal=normal.tolist(), plane_center=center.tolist(),
                           residual_p95=float(np.quantile(residual, .95)), slope_deg=slope)
                if singular[1] < 1e-6 or rec['residual_p95'] > config.max_residual or slope > config.max_slope_deg:
                    rec['reason'] = 'plane_quality'
                    continue
                directions = rays[region]
                denominator = directions @ normal
                if np.any(np.abs(denominator) < 1e-8):
                    rec['reason'] = 'parallel_ray'
                    continue
                t = ((center-origin) @ normal) / denominator
                fill = origin + t[:, None]*directions
                extent = float(np.linalg.norm(np.ptp(fill[:, [0, 2]], axis=0)))
                hull = cv2.convexHull(fill[:, [0, 2]].astype('float32'))
                area = float(cv2.contourArea(hull))
                rec.update(extent=extent, area=area)
                if np.any(t <= 0) or extent > config.max_extent or area > config.max_area:
                    rec['reason'] = 'size_or_behind_camera'
                    continue
                proposed = world.copy()
                proposed[region] = fill
                jumps = []
                for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                    ry, rx = np.where(region)
                    sy, sx = ry+dy, rx+dx
                    inside = (sy >= 0) & (sy < h) & (sx >= 0) & (sx < w)
                    ry, rx, sy, sx = ry[inside], rx[inside], sy[inside], sx[inside]
                    hit = visible[sy, sx]
                    jumps.extend(np.linalg.norm(proposed[ry[hit], rx[hit]]-world[sy[hit], sx[hit]], axis=1))
                rec['boundary_jump_max'] = float(max(jumps, default=float('inf')))
                if rec['boundary_jump_max'] > config.max_boundary_jump:
                    rec['reason'] = 'boundary_discontinuity'
                    continue
                world[region] = fill
                inferred |= region
                allowed &= ~region
                rec.update(status='INFERRED', reason='local_dynamic_occlusion')
        if len(evidence) == evidence_start:
            evidence.append(dict(entity=entity, category=category, pixels=0,
                                 status='REJECTED', reason='no_bracketed_ground_support'))
    return GroundCompletion(world, visible, inferred, evidence)


def ground_mesh(result, *, max_edge=.5):
    """Full-resolution triangles; never bridge unknown cells. Tags per triangle."""
    if not np.isfinite(max_edge) or max_edge <= 0:
        raise ValueError('max_edge must be positive and finite')
    mask = result.visible_mask | result.inferred_mask
    cell = mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]
    y, x = np.where(cell)
    q = np.stack([result.points[y, x], result.points[y, x+1],
                  result.points[y+1, x], result.points[y+1, x+1]], axis=1)
    tags = np.stack([result.inferred_mask[y, x], result.inferred_mask[y, x+1],
                     result.inferred_mask[y+1, x], result.inferred_mask[y+1, x+1]], axis=1)
    triangles = q[:, [[0, 1, 2], [1, 3, 2]]].reshape(-1, 3, 3)
    inferred = tags[:, [[0, 1, 2], [1, 3, 2]]].any(axis=2).reshape(-1)
    edges = np.linalg.norm(triangles-np.roll(triangles, 1, axis=1), axis=2)
    keep = np.isfinite(triangles).all(axis=(1, 2)) & (edges.max(axis=1) <= max_edge)
    area_vector = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    keep &= np.linalg.norm(area_vector, axis=1) > 0
    triangles, inferred = triangles[keep], inferred[keep]
    reverse = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])[:, 1] < 0
    triangles[reverse] = triangles[reverse][:, [0, 2, 1]]
    return triangles.astype('<f4'), inferred


def run_urban(root, output):
    """Load existing RGB/SAM/VGGT and scene calibration only; no GT inputs."""
    report = json.loads((root/'surface_assets/report.json').read_text())
    perception = json.loads((root/'perception_target/result.json').read_text())
    with np.load(root/'reconstruction/single.npz', allow_pickle=False) as data:
        points, conf = data['points'][0], data['depth_conf'][0]
        image = data['images'][0].transpose(1, 2, 0)
        ext, intr = data['extrinsic'][0], data['intrinsic'][0]
        prep = json.loads(str(data['metadata']))['preprocessing']['images'][0]
    h, w = points.shape[:2]
    if list(perception['image_size']) != prep['original_wh']:
        raise ValueError('SAM and reconstruction source dimensions differ')
    source = Image.open(root/'perception_target/input.png').convert('RGB')
    source = source.resize(tuple(prep['resized_wh']), Image.Resampling.BICUBIC)
    source = source.crop((0, prep['crop_top'], source.width, min(source.height, prep['crop_top']+h)))
    canvas = Image.new('RGB', (w, h), 'white')
    canvas.paste(source, tuple(prep['padding_ltrb'][:2]))
    if np.max(np.abs(np.asarray(canvas)/255.-image)) > .01:
        raise ValueError('SAM source RGB does not match reconstruction')
    masks = []
    for i, item in enumerate(perception['observations']):
        path = root/'perception_target'/Path(item['mask_path']).name
        mask = np.asarray(Image.open(path).convert('L')) > 0
        warped = cv2.warpPerspective(mask.astype('uint8'), np.asarray(prep['original_to_processed_pixel_centers']),
                                     (w, h), flags=cv2.INTER_NEAREST) > 0
        eroded = cv2.erode(warped.astype('uint8'), np.ones((3, 3), 'uint8')) > 0
        masks.append(dict(id=f'geo_{i:03d}', category=item['category'], mask=eroded, raw_mask=warped))
    valid = np.isfinite(conf) & np.isfinite(points).all(2)
    valid &= conf >= np.quantile(conf[valid], report['confidence_quantile'])
    config = GroundConfig()
    result = complete_ground(points, masks, ext, intr, report['model_to_scene'], valid=valid, config=config)
    triangles, tags = ground_mesh(result)
    output.mkdir(parents=True, exist_ok=False)
    colors = np.broadcast_to(np.where(tags[:, None, None], np.array([1., .35, .1]),
                                      np.array([.15, .7, .85])), triangles.shape)
    np.concatenate((triangles, colors), axis=2).astype('<f4').tofile(output/'ground.bin')
    triangles.tofile(output/'ground_collision.bin')
    np.savez_compressed(output/'completion.npz', points=result.points, visible_mask=result.visible_mask,
                        inferred_mask=result.inferred_mask, triangle_inferred=tags)
    Image.fromarray(result.inferred_mask.astype('uint8')*255).save(output/'inferred_mask.png')
    overlay = (image*255).clip(0, 255).astype('uint8')
    overlay[result.visible_mask] = [38, 179, 217]
    overlay[result.inferred_mask] = [255, 89, 26]
    preview = Image.fromarray(overlay).resize((w*2, h*2))
    ImageDraw.Draw(preview).text((8, 8), 'CYAN: visible ground   ORANGE: INFERRED', fill='white', stroke_width=2, stroke_fill='black')
    preview.save(output/'completion_overlay.png')
    summary = dict(config=asdict(config), scale_source=report['scale_source'],
                   inputs=['reconstruction/single.npz', 'perception_target/result.json', 'surface_assets/report.json'],
                   triangles=len(triangles), inferred_triangles=int(tags.sum()), **result.report())
    (output/'evidence.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urban', type=Path, default=Path('outputs/geometry_init/urban'))
    parser.add_argument('--output', type=Path, default=Path('outputs/geometry_init/urban/ground_completion'))
    args = parser.parse_args()
    run_urban(args.urban, args.output)
