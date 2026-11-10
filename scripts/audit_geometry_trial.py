"""Audit source pairing and recorded actor projections without image warping."""
import argparse
import copy
import json
from pathlib import Path
import sys
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from scripts.prepare_geometry_trial import (
    camera_record, decode_frames, read_image, read_json, save_image,
)
from memory.gta_release import Records
from memory.gta_release_media import rgb_file_index
from scripts.prepare_gta_object_memory import camera_geometry, project_box


def difference(a, b):
    if a.shape != b.shape:
        return {"equal": False, "shapes": [list(a.shape), list(b.shape)]}
    delta = a.astype(float) - b.astype(float)
    return {"equal": bool(np.array_equal(a, b)),
            "mse": float(np.mean(delta ** 2)), "max_abs": float(np.abs(delta).max())}


def bbox(mask):
    y, x = np.nonzero(mask)
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1] if len(x) else None


def blue_actor(image):
    b, g, r = image.astype(float).transpose(2, 0, 1)
    mask = ((b > 70) & (b > 1.8 * (g + 1)) & (b > 1.8 * (r + 1))).astype('uint8')
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        raise ValueError('No blue proxy actor')
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == index


def delivery_impact():
    source = read_json(ROOT / 'outputs/gta_pi_loop_20261102/urban/source.json')
    sample_dir = Path(source['sample']).parent
    sid = sample_dir.name
    episode = read_json(sample_dir / 'sample.json')['episode_id']
    base = sample_dir.parents[3]
    delivery = base / 'unified_delivery'
    archive = delivery / 'GTA_memory_training_unified.zip'
    splits = {}
    for split in ('train', 'validation', 'test'):
        path = delivery / 'package_metadata/manifests' / f'{split}.jsonl'
        rows = [(i, json.loads(line)) for i, line in enumerate(path.read_text().splitlines(), 1)]
        splits[split] = {'path': str(path),
                         'episode_records': sum(r.get('episode_id') == episode for _, r in rows),
                         'sample_matches': [{'line': i, 'record': r} for i, r in rows if r.get('sample_id') == sid]}
    media = {}
    # Read only the two requested PNG members, without extracting the archive.
    with zipfile.ZipFile(archive) as bundle:
        for name in ('target_rgb.png', 'target_whitebox.png'):
            member = f'samples/{sid}/{name}'
            archived = bundle.read(member)
            original = (sample_dir / name).read_bytes()
            decoded = cv2.imdecode(np.frombuffer(archived, dtype='uint8'), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError(f'Invalid archived PNG: {member}')
            media[name] = {'archive_member': member, 'source_path': str(sample_dir / name),
                           'bytes': len(archived), 'byte_identical': archived == original,
                           'pixel_comparison': difference(decoded, read_image(sample_dir / name))}
    audit_path = base / 'audited_mixture/exclusion_ledger.jsonl'
    audit = [{'line': i, 'record': json.loads(line)}
             for i, line in enumerate(audit_path.read_text().splitlines(), 1)
             if json.loads(line).get('sample_id') == sid]
    return {
        'sample_id': sid, 'episode_id': episode, 'archive': str(archive),
        'final_splits': splits, 'archived_target_media': media,
        'prior_audit': {'path': str(audit_path), 'matches': audit,
                        'packaged_audit_member': f'provenance/cumulative_audit/scene_audit/{sid}.json'},
        'upload_record_path': str(delivery / 'HF_UPLOAD_STATUS.json'),
        'upload_record': read_json(delivery / 'HF_UPLOAD_STATUS.json'),
        'delivery_manifest_path': str(delivery / 'DELIVERY.json'),
        'remote_checked_this_audit': False,
        'confirmed_impact': 'This exact sample and its unchanged target RGB/proxy are in the final local ZIP identified by the upload record.',
        'scope_limit': '164 same-episode training records are included; this is membership, not evidence that all 164 or the whole delivery have the same mismatch. Recipient download or use is not established.',
        'source_camera_vs_visual_truth': {
            'verified': 'Frame 480 source RGB and original proxy match the training PNGs exactly; FOREGROUND BLUE SUBJECT ONLY: recorded actor projection reproduces its proxy bbox within one pixel, mask IoU 0.9807914834529045. This does not verify the second red actor against RGB.',
            'actor_difference': 'FOREGROUND BLUE SUBJECT ONLY: rigid 13-part arms-spread proxy, unavailable measured body pose; RGB detection center and proxy bbox center differ by 1.98883056640625 px. This result does not apply to the second red other_agent.',
            'camera_limit': 'Agreement with recorded rays verifies record-to-proxy projection, not independent calibration against RGB. Actual capture rectangle/crop remains unverified; adjacent target camera/subject records are identical.',
            'geometry_limit': 'Physical collision geometry is not full visual geometry. static_session_match_verified=false; individual building mismatch causes have not been independently established.',
            'review_limit': 'Prior alignment_ok=true and accepted=true reflect sampled coarse background-memory review, not pixel-aligned visual ground truth.',
            'new_render_limit': 'Audit region render is a new static-only rendering of the existing 100-unit region with recorded camera, versus the original 190-unit proxy region. It is not a repaired GT or a replacement delivery asset.',
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional audit-only artifact directory')
    parser.add_argument('--render-region', action='store_true')
    parser.add_argument('--delivery-only', action='store_true', help='Append delivery evidence to an existing audit.json')
    args = parser.parse_args()
    if args.delivery_only:
        if args.output is None:
            parser.error('--delivery-only requires --output')
        path = args.output / 'audit.json'
        report = read_json(path)
        report['delivery_impact'] = delivery_impact()
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(json.dumps(report['delivery_impact'], indent=2))
        return
    if args.render_region and args.output is None:
        parser.error('--render-region requires --output')
    source = read_json(ROOT / 'outputs/gta_pi_loop_20261102/urban/source.json')
    sample_dir = Path(source['sample']).parent
    sample = read_json(sample_dir / 'sample.json')
    target = read_json(sample_dir / 'target_state.json')
    reference = read_json(sample_dir / 'reference_state.json')
    frame_path = Path(target['source_manifest']).with_name('frames.jsonl')
    rows = [json.loads(line) for line in frame_path.read_text().splitlines()]
    frames = {row['proxy_frame_index']: row for row in rows}
    target_rows = read_json(sample_dir / 'target_frame_states.json')
    source_root = Path(target['source_root'])
    base = sample_dir.parents[3]
    receipt = read_json(base / 'record_adapter' / (sample['episode_id'] + '.json'))
    proxy = Path(target['media_paths']['proxy_video'])
    proxy_dir = proxy.parent
    compiler = read_json(proxy_dir / 'compiler_manifest.json')
    method = base / 'source/method/source/src'
    sys.path.insert(0, str(method))
    from wm_engine.processing.client_samples import _interpolate_camera
    from wm_engine.proxy.geometry import Camera, Rasterizer, box_vertices, quaternion_matrix
    from wm_engine.proxy.humanoid import humanoid_parts

    wanted = [reference['frame_index'], target['frame_index']]
    proxy_images = decode_frames(proxy, wanted)
    rgb_images = decode_frames(Path(target['media_paths']['rgb_video']),
                               [rgb_file_index(frames[i]) for i in wanted])
    report = {
        'sample_json': source['sample'], 'frames_jsonl': str(frame_path),
        'proxy_video': str(proxy), 'record_receipt': receipt,
        'rebuild_proxy_exists': (source_root / read_json(Path(target['source_manifest']))['proxy_video']).exists(),
        'old_vs_rebuild_metadata_byte_equal': {
            name: (proxy_dir / name).read_bytes() == (frame_path.parent / name).read_bytes()
            for name in ('frames.jsonl', 'manifest.json', 'compiler_manifest.json')},
        'target_frame_states_equal_source': target_rows == [frames[i] for i in range(480, 600)],
        'geometry_contract': {k: compiler[k] for k in (
            'geometry_fidelity', 'static_geometry', 'static_session_match_verified',
            'scene_limits', 'body_proxy', 'alignment')},
        'pairs': {},
    }
    region_dir = ROOT / 'outputs/gta_asset_alignment/urban_region'
    region = read_json(region_dir / 'region.json')
    records = Records(Path(receipt['index']))
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
    try:
        for side, state in [('reference', reference), ('target', target)]:
            row = frames[state['frame_index']]
            rgb = read_image(sample_dir / f'{side}_rgb.png')
            whitebox = read_image(sample_dir / f'{side}_whitebox.png')
            h, w = rgb.shape[:2]
            info = row['camera_interpolation']
            brackets = [records.camera(state['agent_id'], row['client_clock_domain'], info[k])
                        for k in ('lower_client_sample_seq', 'upper_client_sample_seq')]
            camrow = _interpolate_camera(brackets[0]['camera'], brackets[1]['camera'], info['alpha'])
            camera = Camera(camrow, w, h)
            exported = camera_record(records, state, row, w, h, np.array(region['origin_gta']), receipt)
            nearest = next(c for c in brackets if c['client_sample_seq'] == info['nearest_client_sample_seq'])
            subject = nearest['subject_observation']
            if subject['body_pose']['available']:
                raise ValueError('Audit actor construction assumes the documented rigid fallback')
            envelope = box_vertices(subject['position'], [-.325, -.20, -.95], [.325, .20, .85],
                                    quaternion_matrix(subject['orientation_quaternion']))
            parts = humanoid_parts(envelope)
            vertices = np.concatenate([v for _, v in parts])
            def projected_bounds(cam, points=vertices):
                projected = cam.project(cam.transform(points))
                return np.r_[projected.min(axis=0), projected.max(axis=0)].tolist()
            actor_render = Rasterizer(camera)
            for _, vertices_part in parts:
                actor_render.box(vertices_part, 'agent', human_facing_cues=True)
            measured = blue_actor(whitebox)
            simulated = blue_actor(actor_render.image)
            equal_fy = copy.copy(camera)
            equal_fy.fx = equal_fy.fy
            equal_fx = copy.copy(camera)
            equal_fx.fy = equal_fx.fx
            origin, basis = camera_geometry(*brackets, info['alpha'])
            entity = next(e for e in nearest['scene_observation']['entities']
                          if str(e.get('dataset_entity_id')) == str(subject['dataset_entity_id']))
            bracket_results = []
            for c in brackets:
                sub = c['subject_observation']
                box = box_vertices(sub['position'], [-.325, -.20, -.95], [.325, .20, .85],
                                   quaternion_matrix(sub['orientation_quaternion']))
                points = np.concatenate([v for _, v in humanoid_parts(box)])
                bracket_results.append({
                    'seq': c['client_sample_seq'], 'time_ns': c['server_time_estimate_ns'],
                    'render_frame_id': c['render_frame_id'],
                    'camera_position': c['rendered_camera_position'], 'subject_position': sub['position'],
                    'subject_only_change_bbox': projected_bounds(camera, points),
                    'camera_only_change_bbox': projected_bounds(Camera(c['camera'], w, h)),
                    'paired_camera_subject_bbox': projected_bounds(Camera(c['camera'], w, h), points),
                })
            uv = np.array([[.05, .05], [.95, .05], [.5, .5], [.05, .95], [.95, .95]])
            points = origin + np.column_stack([uv - .5, np.ones(len(uv))]) @ basis.T * 10
            renderer_pixels = camera.project(camera.transform(points))
            old_frame = json.loads((proxy_dir / 'frames.jsonl').read_text().splitlines()[state['frame_index']])
            item = {
                'state_frame_index': state['frame_index'], 'state_time_ns': state['time_ns'],
                'frame': row, 'old_proxy_frame_equals_rebuild': old_frame == row,
                'rgb_png_vs_source': difference(rgb, rgb_images[rgb_file_index(row)]),
                'whitebox_png_vs_source_proxy': difference(whitebox, proxy_images[state['frame_index']]),
                'exported_whitebox_vs_training': difference(read_image(ROOT / f'outputs/geometry_init/urban/evaluation/{side}_whitebox.png'), whitebox),
                'training_clip_first_frames': {
                    kind: difference(read_image(sample_dir / f'target_{kind}.png'),
                                     decode_frames(sample_dir / f'target_{kind}.mp4', [0])[0])
                    for kind in ('rgb', 'whitebox')
                } if side == 'target' else None,
                'recorded_render_resolution': camrow['render_resolution'], 'image_size_wh': [w, h],
                'K_pixel_edges': exported['K_pixel_edges'],
                'original_renderer_fx_fy': [camera.fx, camera.fy],
                'original_renderer_vs_ray_pixel_error': float(np.max(np.abs(renderer_pixels - uv * [w, h]))),
                'subject_entity_id': subject['dataset_entity_id'], 'subject_position': subject['position'],
                'subject_pose_available': subject['body_pose']['available'],
                'model_bounds_projected_bbox': project_box(entity, origin, basis, w, h),
                'proxy_blue_bbox_xyxy_exclusive': bbox(measured),
                'recorded_actor_render_bbox_xyxy_exclusive': bbox(simulated),
                'recorded_actor_projected_bbox': projected_bounds(camera),
                'actor_mask_iou': float(np.count_nonzero(measured & simulated) / np.count_nonzero(measured | simulated)),
                'equal_focal_fx_equals_fy_bbox': projected_bounds(equal_fy),
                'equal_focal_fy_equals_fx_bbox': projected_bounds(equal_fx),
                'brackets': bracket_results,
                'camera': exported,
            }
            report['pairs'][side] = item
            if side == 'target':
                perception = ROOT / 'outputs/geometry_init/urban/perception_target'
                result = read_json(perception / 'result.json')
                if not np.array_equal(read_image(perception / 'target_rgb.png'), rgb):
                    raise ValueError('Existing RGB perception uses a different image')
                people = [o for o in result['observations'] if o['category'] == 'person']
                own = max(people, key=lambda o: o['bbox_xyxy'][3] - o['bbox_xyxy'][1])
                mask = read_image(perception / Path(own['mask_path']).name)[:, :, 0] > 0
                item['rgb_actor_detection'] = {
                    'source': str(perception / 'result.json'), 'observation_id': own['observation_id'],
                    'bbox': own['bbox_xyxy'], 'mask_bbox': bbox(mask),
                    'interpretation': 'Existing model detection, not annotated ground truth; tallest detected person',
                    'proxy_minus_rgb_bbox_center_x': (bbox(measured)[0] + bbox(measured)[2] - own['bbox_xyxy'][0] - own['bbox_xyxy'][2]) / 2,
                }
                item['canvas_evidence'] = {
                    'recorded_aspect': 3072 / 1920, 'rgb_aspect': w / h,
                    'fx_over_fy': camera.fx / camera.fy,
                    'aspect_ratio_change': (w / h) / (3072 / 1920),
                    'capture_code': str(method / 'wm_engine/capture/windows_fivem.py'),
                    'capture_code_behavior': 'gdigrab window capture followed by scale=width:height; no crop in command builder',
                    'actual_capture_rectangle_verified': False,
                }
            if args.output:
                save_image(args.output / f'{side}_recorded_actor.png', actor_render.image)
                save_image(args.output / f'{side}_original_proxy.png', proxy_images[state['frame_index']])
            if args.render_region:
                packed = np.fromfile(region_dir / 'surfaces.bin', dtype='<f4').reshape(-1, 3, 6)
                # The region stores Godot-local coordinates; invert that documented transform.
                local = packed[:, :, :3]
                gta = local[:, :, [0, 2, 1]].astype(float)
                gta[:, :, 1] *= -1
                gta += region['origin_gta']
                raster = Rasterizer(camera)
                transformed = camera.transform(gta.reshape(-1, 3)).reshape(-1, 3, 3)
                # Frustum rejection only, preserving triangles intersecting the near plane.
                z = transformed[:, :, 2]
                x = camera.fx * transformed[:, :, 0]
                y = camera.fy * transformed[:, :, 1]
                visible = ~(np.all(z < camera.near, axis=1) |
                            np.all(x < -w / 2 * z, axis=1) | np.all(x > w / 2 * z, axis=1) |
                            np.all(y < -h / 2 * z, axis=1) | np.all(y > h / 2 * z, axis=1))
                for i in np.flatnonzero(visible):
                    raster.polygon(gta[i], np.rint(packed[i, 0, 3:][::-1] * 255).astype('uint8'))
                save_image(args.output / f'{side}_region_recordcamera.png', raster.image)
                labels = ['RGB (training source)', 'Original dataset proxy (not aligned GT)',
                          'NEW region + recorded camera (static only)']
                canvas = np.full((h + 40, w * 3, 3), 255, dtype='uint8')
                for i, (label, image) in enumerate(zip(labels, [rgb, whitebox, raster.image])):
                    canvas[40:, i*w:(i+1)*w] = image
                    cv2.putText(canvas, label, (i*w+10, 27), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 1)
                save_image(args.output / f'{side}_source_region_comparison.jpg', canvas)
                item['new_render'] = {
                    'path': str(args.output / f'{side}_region_recordcamera.png'),
                    'source_geometry': str(region_dir / 'surfaces.bin'), 'region_radius': region['radius'],
                    'original_proxy_region_radius': read_json(proxy_dir / row['scene_index'])['region_radius'],
                    'triangles_total': len(gta), 'triangles_in_frustum': int(visible.sum()),
                    'renderer': str(method / 'wm_engine/proxy/geometry.py'),
                    'meaning': 'NEW static region render, not original proxy, not RGB ground truth; actors omitted',
                }
    finally:
        records.close()
    if args.output:
        report['delivery_impact'] = delivery_impact()
        (args.output / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({**{k: v for k, v in report.items() if k not in ('record_receipt', 'pairs')},
                      'pairs': {side: {k: v for k, v in item.items() if k not in ('frame', 'camera')}
                                for side, item in report['pairs'].items()}}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
