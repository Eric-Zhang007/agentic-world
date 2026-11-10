"""Build a portable, reviewable object-memory pilot from existing GTA pairs."""
import argparse
import json
import subprocess
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation, Slerp
from memory.gta_validation import camera_exclusion


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def selected_records(path, keys, column, domain=None):
    result = {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
        for row in batch.to_pylist():
            key = row[column]
            if key in keys and (domain is None or row['client_clock_domain'] == domain):
                if key in result:
                    raise ValueError(f'Ambiguous record {path}:{key}')
                result[key] = json.loads(row['payload_json'])
        if len(result) == len(keys):
            break
    return result


def camera_geometry(lower, upper, alpha):
    origin = (1-alpha)*np.array(lower['rendered_camera_position']) + alpha*np.array(upper['rendered_camera_position'])
    a = lower['projection_rays']['rays']
    b = upper['projection_rays']['rays']
    rays = []
    for first, second in zip(a, b):
        assert first['uv'] == second['uv']
        direction = (1-alpha)*np.array(first['direction']) + alpha*np.array(second['direction'])
        rays.append(direction / np.linalg.norm(direction))
    assert [r['uv'] for r in a] == [[.5, .5], [.75, .5], [.5, .75]]
    forward = rays[0]
    right = (rays[1]/np.dot(rays[1], forward)-forward)/.25
    down = (rays[2]/np.dot(rays[2], forward)-forward)/.25
    basis = np.column_stack([right, down, forward])
    if not np.isfinite(basis).all() or abs(np.linalg.det(basis)) < 1e-8:
        raise ValueError('Invalid projection-ray basis')
    return origin, basis


def interpolate_entities(lower, upper, alpha):
    newer = {str(e['dataset_entity_id']): e for e in upper['scene_observation']['entities']
             if e.get('dataset_entity_id') is not None}
    result = []
    for old in lower['scene_observation']['entities']:
        oid = str(old.get('dataset_entity_id'))
        new = newer.get(oid)
        if new is None or old.get('model_hash') != new.get('model_hash'):
            continue
        fields = ('position', 'orientation_quaternion', 'model_bounds_min', 'model_bounds_max')
        if any(old.get(k) is None or new.get(k) is None for k in fields):
            continue
        obj = dict(old)
        obj['position'] = ((1-alpha)*np.array(old['position'])+alpha*np.array(new['position'])).tolist()
        rotations = Rotation.from_quat([old['orientation_quaternion'], new['orientation_quaternion']])
        obj['orientation_quaternion'] = Slerp([0, 1], rotations)([alpha]).as_quat()[0].tolist()
        result.append(obj)
    return result


def project_box(obj, origin, basis, width, height):
    corners = np.array(list(product(*zip(obj['model_bounds_min'], obj['model_bounds_max']))))
    points = Rotation.from_quat(obj['orientation_quaternion']).apply(corners) + obj['position']
    q = np.linalg.solve(basis, (points-origin).T).T
    if np.any(q[:, 2] <= .05):
        return None
    uv = q[:, :2]/q[:, 2:3] + .5
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    if np.any(hi <= 0) or np.any(lo >= 1):
        return None
    return (np.r_[np.maximum(lo, 0)*[width, height], np.minimum(hi, 1)*[width, height]]).tolist()


def coarse_visibility(objects, origin, basis, width=320, height=180):
    """Ray/OBB depth order; these are proxy masks, never RGB segmentation truth."""
    yy, xx = np.mgrid[:height, :width]
    rays = np.stack([(xx+.5)/width-.5, (yy+.5)/height-.5, np.ones_like(xx)], axis=-1) @ basis.T
    nearest = np.full((height, width), np.inf)
    owner = np.full((height, width), -1, dtype=np.int32)
    hits = []
    for index, obj in enumerate(objects):
        rot = Rotation.from_quat(obj['orientation_quaternion']).as_matrix()
        local_origin = (origin-np.array(obj['position'])) @ rot
        local_rays = rays @ rot
        lo, hi = np.array(obj['model_bounds_min']), np.array(obj['model_bounds_max'])
        parallel = np.abs(local_rays) < 1e-10
        safe = np.where(parallel, 1., local_rays)
        t0, t1 = (lo-local_origin)/safe, (hi-local_origin)/safe
        near = np.where(parallel, -np.inf, np.minimum(t0, t1)).max(axis=-1)
        far = np.where(parallel, np.inf, np.maximum(t0, t1)).min(axis=-1)
        outside = np.any(parallel & ((local_origin < lo) | (local_origin > hi)), axis=-1)
        hit = ~outside & (far >= np.maximum(near, .05))
        distance = np.where(hit, np.maximum(near, .05), np.inf)
        closer = distance < nearest
        owner[closer] = index
        nearest[closer] = distance[closer]
        hits.append(int(hit.sum()))
    return owner, hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--agent', default='agent_000')
    parser.add_argument('--segment', default='segment_00001')
    parser.add_argument('--reference-frame', type=int, default=0)
    parser.add_argument('--start-frame', type=int, default=120)
    parser.add_argument('--frames', type=int, default=120)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output directory; existing packages are immutable')
    root = args.episode
    segment = root/'derived/segments'/args.agent/args.segment
    manifest = read(segment/'manifest.json')
    rows = [json.loads(line) for line in (segment/'frames.jsonl').read_text(encoding='utf-8').splitlines()]
    lookup = {f['proxy_frame_index']: f for f in rows}
    indices = [args.reference_frame, *range(args.start_frame, args.start_frame+args.frames)]
    chosen = [lookup[i] for i in indices]
    assert args.reference_frame < args.start_frame
    domain = chosen[0]['client_clock_domain']
    server_domain = chosen[0]['server_clock_domain']
    for frame in chosen:
        if camera_exclusion(frame):
            raise ValueError('Clip crosses missing/unsupported camera interval; do not bridge the gap')
        assert frame['client_clock_domain'] == domain and frame['server_clock_domain'] == server_domain
    times = [f['server_time_estimate_ns'] for f in chosen]
    assert all(a < b for a, b in zip(times, times[1:]))
    target_decode = [f['rgb_decode_index'] for f in chosen[1:]]
    assert target_decode == list(range(target_decode[0], target_decode[0]+args.frames))
    keys = {f['camera_interpolation'][k] for f in chosen
            for k in ('lower_client_sample_seq', 'upper_client_sample_seq')}
    cameras = selected_records(root/f'agents/{args.agent}/camera.parquet', keys, 'client_sample_seq', domain)
    ticks = {f['world_capture_tick'] for f in chosen}
    states = selected_records(root/'server/world_state.parquet', ticks, 'sample_seq')
    if set(cameras) != keys or set(states) != ticks:
        raise ValueError('Missing camera/world joins; sample rejected')
    out = args.output
    (out/'references').mkdir(parents=True)
    (out/'review').mkdir()
    write(out/'source_manifest.json', manifest)
    write(out/'appearance_intervals.json', read(segment/'appearance.json'))
    write(out/'camera_samples.json', cameras)
    write(out/'world_states.json', states)
    scene_names = sorted({f['scene_index'] for f in chosen})
    (out/'scenes').mkdir()
    for name in scene_names:
        write(out/'scenes'/name, read(segment/name))
    observations = []
    cards = []
    rgb = cv2.VideoCapture(str(root/manifest['rgb_video']))
    proxy = cv2.VideoCapture(str(root/manifest['proxy_video']))
    for number, frame in enumerate(chosen):
        interp = frame['camera_interpolation']
        lower, upper = (cameras[interp[k]] for k in ('lower_client_sample_seq', 'upper_client_sample_seq'))
        origin, basis = camera_geometry(lower, upper, interp['alpha'])
        objects = interpolate_entities(lower, upper, interp['alpha'])
        observation = {'proxy_frame_index': frame['proxy_frame_index'], 'frame_metadata': frame,
                       'camera_origin': origin.tolist(), 'ray_basis': basis.tolist(),
                       'interpolated_entities': objects}
        if number == 0 or (number-1) % 24 == 0 or number == len(chosen)-1:
            rgb.set(cv2.CAP_PROP_POS_FRAMES, frame['rgb_decode_index'])
            proxy.set(cv2.CAP_PROP_POS_FRAMES, frame['proxy_frame_index'])
            ok, image = rgb.read(); ok2, whitebox = proxy.read()
            if not ok or not ok2:
                raise ValueError('Incomplete/unreadable video')
            height, width = image.shape[:2]
            owners, hits = coarse_visibility(objects, origin, basis)
            candidates = []
            marked = image.copy()
            for index, obj in enumerate(objects):
                box = project_box(obj, origin, basis, width, height)
                if box is None:
                    continue
                oid = str(obj['dataset_entity_id'])
                mask = (owners == index).astype(np.uint8)*255
                mask_path = f'review/{frame["proxy_frame_index"]:06d}_{oid}_coarse.png'
                cv2.imwrite(str(out/mask_path), mask)
                candidates.append({'object_id': f'{manifest["episode_id"]}:entity:{oid}',
                    'dataset_entity_id': oid, 'category': obj['entity_type'], 'model_hash': obj['model_hash'],
                    'bbox_xyxy': box, 'coarse_mask': mask_path, 'mask_resolution': [320, 180],
                    'dynamic_obb_visible_fraction': float((owners == index).sum()/max(1, hits[index])),
                    'static_occlusion_checked': False, 'visibility_status': 'NEEDS_REVIEW',
                    'state_compatibility': 'UNKNOWN', 'current_vehicle_network_id': obj.get('current_vehicle_network_id')})
                x0,y0,x1,y1 = map(round, box)
                cv2.rectangle(marked, (x0,y0), (x1,y1), (0,255,255), 2)
                cv2.putText(marked, oid[-5:], (x0,max(18,y0)), cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),1)
            observation['geometry_candidates'] = candidates
            cv2.imwrite(str(out/f'review/rgb_{frame["proxy_frame_index"]:06d}.jpg'), image)
            cv2.imwrite(str(out/f'review/projected_{frame["proxy_frame_index"]:06d}.jpg'), marked)
            card = np.concatenate([cv2.resize(marked,(640,360)),cv2.resize(whitebox,(640,360))],axis=1)
            cv2.putText(card,f'frame {frame["proxy_frame_index"]} / projected OBBs, NOT masks',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),1)
            cards.append(card)
            if number == 0:
                cv2.imwrite(str(out/'references/rgb.png'), image)
                cv2.imwrite(str(out/'references/whitebox.png'), whitebox)
        observations.append(observation)
    rgb.release(); proxy.release()
    write(out/'observations.json', observations)
    cv2.imwrite(str(out/'review/contact_sheet.jpg'), np.concatenate(cards,axis=0))
    for kind, source, begin in [('rgb', manifest['rgb_video'],target_decode[0]),
                                ('whitebox',manifest['proxy_video'],args.start_frame)]:
        subprocess.run(['ffmpeg','-v','error','-n','-i',str(root/source),'-vf',
            f'trim=start_frame={begin}:end_frame={begin+args.frames},setpts=PTS-STARTPTS',
            '-fps_mode','passthrough','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',
            str(out/f'target_{kind}.mp4')],check=True)
        cap=cv2.VideoCapture(str(out/f'target_{kind}.mp4'));count=0
        while cap.read()[0]: count+=1
        cap.release()
        assert count==args.frames,(kind,count)
    write(out/'package.json', {'status':'PENDING_MANUAL_REVIEW','training_ready':False,
        'episode_id':manifest['episode_id'],'agent_id':args.agent,'segment':args.segment,
        'reference_time_ns':times[0],'target_start_time_ns':times[1],'target_end_time_ns':times[-1],
        'target_frames':args.frames,'source_root':str(root), 'reference_edges':[],
        'missing_camera_policy':'reject_clip_do_not_bridge',
        'static_geometry_available':(root.parent/'worlds').exists(),
        'mask_semantics':'dynamic oriented bounding box ray intersections, not actual RGB masks',
        'scope':'one pilot, not full dataset processing'})
    print(out)


if __name__ == '__main__':
    main()
