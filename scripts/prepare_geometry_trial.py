"""Export urban RGB inputs and separate, ray-derived evaluation cameras."""
import argparse
import json
from pathlib import Path
import shutil
import sys

# Keep direct script execution from creating caches alongside source modules.
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from scipy.linalg import rq

from memory.gta_release import Records
from memory.gta_release_media import open_video, rgb_file_index
from scripts.prepare_gta_object_memory import camera_geometry


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot write image: {path}")


def decode_frames(path, indices):
    wanted = set(indices)
    frames = {}
    cap = open_video(path)
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open source video: {path}")
        for index in range(max(wanted) + 1):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"Source video ended before frame {index}: {path}")
            if index in wanted:
                frames[index] = image
    finally:
        cap.release()
    return frames


def validate_frame(state, frame, source_frame):
    if frame != source_frame:
        raise ValueError('Training frame metadata differs from source frames.jsonl')
    if any(state[key] != frame[key] for key in ('server_clock_domain', 'client_clock_domain')):
        raise ValueError('State and frame clock domains differ')
    if frame['proxy_frame_index'] == state['frame_index'] and state['time_ns'] != frame['server_time_estimate_ns']:
        raise ValueError('State and frame timestamps differ')


def proxy_provenance(sample_dir, side, state, frame, evaluation):
    proxy = Path(state['media_paths']['proxy_video'])
    source_dir = Path(state['source_manifest']).parent
    for name in ('manifest.json', 'frames.jsonl', 'compiler_manifest.json'):
        if (source_dir / name).read_bytes() != (proxy.parent / name).read_bytes():
            raise ValueError(f'Proxy provenance mismatch: {name}')
    image = read_image(sample_dir / f'{side}_whitebox.png')
    original = decode_frames(proxy, [frame['proxy_frame_index']])[frame['proxy_frame_index']]
    if not np.array_equal(image, original):
        raise ValueError(f'{side} whitebox PNG differs from original proxy frame')
    compiler = read_json(proxy.parent / 'compiler_manifest.json')
    return {
        'path': str(evaluation / f'{side}_whitebox.png'),
        'source': str(sample_dir / f'{side}_whitebox.png'),
        'kind': 'original_dataset_proxy', 'aligned_rgb_gt': False,
        'label': 'Dataset proxy (NOT aligned GT)',
        'proxy_video': str(proxy), 'proxy_frame_index': frame['proxy_frame_index'],
        'rgb_file_frame_index': rgb_file_index(frame),
        'server_time_estimate_ns': frame['server_time_estimate_ns'],
        'frames_jsonl': str(source_dir / 'frames.jsonl'),
        'proxy_frames_jsonl': str(proxy.with_name('frames.jsonl')),
        'png_matches_original_proxy_frame': True, 'metadata_byte_identical': True,
        'geometry_fidelity': compiler['geometry_fidelity'],
        'static_session_match_verified': compiler['static_session_match_verified'],
        'body_proxy': compiler['body_proxy'], 'scene_limits': compiler['scene_limits'],
    }


def camera_record(records, state, frame, width, height, origin, receipt):
    interpolation = frame["camera_interpolation"]
    brackets = [records.camera(state["agent_id"], frame["client_clock_domain"],
                              interpolation[key])
                for key in ("lower_client_sample_seq", "upper_client_sample_seq")]
    if any(c["episode_id"] != state["episode_id"] or
           c["server_clock_domain"] != frame["server_clock_domain"] for c in brackets):
        raise ValueError("Camera episode or clock domain mismatch")
    position, basis = camera_geometry(*brackets, interpolation["alpha"])
    gta_to_godot = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
    center = gta_to_godot @ (position - origin)
    basis_local = gta_to_godot @ basis
    pixels = np.array([[width, 0., width / 2 - .5],
                       [0., height, height / 2 - .5], [0., 0., 1.]])

    # RQ retains ray-derived skew/aspect instead of treating the ray basis as R.
    intrinsic, rotation = rq(pixels @ np.linalg.inv(basis_local))
    signs = np.diag(np.where(np.diag(intrinsic) < 0, -1., 1.))
    intrinsic = intrinsic @ signs
    rotation = signs @ rotation
    intrinsic /= intrinsic[2, 2]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-10) or not np.isclose(
            np.linalg.det(rotation), 1., atol=1e-10):
        raise ValueError("Ray decomposition did not produce a proper camera rotation")
    translation = -rotation @ center
    intrinsic_edges = intrinsic.copy()
    intrinsic_edges[:2, 2] += .5
    godot_transform = np.eye(4)
    godot_transform[:3, :3] = rotation.T @ np.diag([1., -1., -1.])
    godot_transform[:3, 3] = center

    # Check both conventions on rays across the image at several depths.
    uv = np.array([(u, v) for u in (.05, .25, .5, .75, .95)
                   for v in (.05, .25, .5, .75, .95)])
    rays = np.column_stack([uv - .5, np.ones(len(uv))])
    points = np.concatenate([position + (rays @ basis.T) * depth
                             for depth in (3., 10., 80.)])
    local = (points - origin) @ gta_to_godot.T
    projected = (local @ rotation.T + translation) @ intrinsic.T
    projected = projected[:, :2] / projected[:, 2:3]
    expected = np.tile(uv * [width, height] - .5, (3, 1))
    error = float(np.max(np.abs(projected - expected)))
    if error > 1e-6:
        raise ValueError(f"Ray/KRT reprojection mismatch: {error} pixels")

    source_keys = ("proxy_frame_index", "rgb_frame_index", "rgb_decode_index",
                   "rgb_segment_frame_index", "rgb_video_pts_ns", "rgb_encoded_pts_ns",
                   "server_time_estimate_ns", "server_clock_domain", "client_clock_domain",
                   "client_alignment_delta_ns", "client_clock_uncertainty_ns", "status",
                   "camera_projection", "camera_resampling", "camera_interpolation",
                   "source_timestamp_provenance", "world_capture_tick", "scene_index")
    return {
        "K": intrinsic.tolist(), "R": rotation.tolist(), "t": translation.tolist(),
        "K_pixel_edges": intrinsic_edges.tolist(),
        "camera_center_godot": center.tolist(),
        "godot_camera_to_world": godot_transform.tolist(),
        "ray_origin_gta": position.tolist(), "ray_basis_gta": basis.tolist(),
        "ray_basis_godot": basis_local.tolist(),
        "estimated_alignment": True, "alignment_quality": "estimated",
        "max_ray_reprojection_error_px": error,
        "source": {
            "episode_id": state["episode_id"], "agent_id": state["agent_id"],
            "point_id": state["point_id"], "source_manifest": state["source_manifest"],
            "frames_jsonl": str(Path(state["source_manifest"]).with_name("frames.jsonl")),
            "rgb_video": state["media_paths"]["rgb_video"],
            "source_video_frame_index": rgb_file_index(frame),
            "record_index": receipt["index"], "record_sources": receipt["sources"],
            "frame": {key: frame[key] for key in source_keys if key in frame},
            "client_brackets": [{
                "client_sample_seq": c["client_sample_seq"],
                "camera": {key: c[key] for key in (
                    "rendered_camera_position", "rendered_camera_rotation", "rendered_camera_fov",
                    "projection_rays", "render_resolution", "near_clip", "far_clip") if key in c}
            } for c in brackets],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/geometry_init/urban')
    args = parser.parse_args()
    source_path = ROOT / "outputs/gta_pi_loop_20261102/urban/source.json"
    source = read_json(source_path)
    sample_path = Path(source["sample"])
    sample_dir = sample_path.parent
    sample = read_json(sample_path)
    target = read_json(sample_dir / "target_state.json")
    reference = read_json(sample_dir / "reference_state.json")
    target_frames = read_json(sample_dir / "target_frame_states.json")
    region_path = ROOT / "outputs/gta_asset_alignment/urban_region/region.json"
    region = read_json(region_path)
    origin = np.array(region["origin_gta"], dtype=float)
    base = sample_dir.parents[3]
    receipt_path = base / "record_adapter" / (sample["episode_id"] + ".json")
    receipt = read_json(receipt_path)
    reference_frame_path = Path(reference["source_manifest"]).with_name("frames.jsonl")
    with reference_frame_path.open(encoding="utf-8") as stream:
        reference_frame = next(json.loads(line) for line in stream
                               if json.loads(line)["proxy_frame_index"] == reference["frame_index"])
    if len(target_frames) != 120 or target_frames[0]["proxy_frame_index"] != target["frame_index"]:
        raise ValueError("Unexpected urban target frame mapping")
    if reference["media_paths"]["rgb_video"] != target["media_paths"]["rgb_video"]:
        raise ValueError("This urban export expects reference and target in the same source video")
    selected = [("reference_rgb.png", reference, reference_frame, None, "reference_rgb.png"),
                ("target_rgb.png", target, target_frames[0], 0, "target_rgb.png")]
    selected += [(f"frame_{i:03d}.png", target, target_frames[i], i, None)
                 for i in (30, 60, 90, 119)]
    source_video = Path(target["media_paths"]["rgb_video"])
    decoded = decode_frames(source_video, [rgb_file_index(row[2]) for row in selected])
    output = args.output.resolve()
    input_dir = output / "input"
    evaluation = output / "evaluation"
    source_frames = {json.loads(line)['proxy_frame_index']: json.loads(line)
                     for line in Path(target['source_manifest']).with_name('frames.jsonl').read_text().splitlines()}
    for frame in target_frames:
        validate_frame(target, frame, source_frames[frame['proxy_frame_index']])
    whitebox = {side: proxy_provenance(sample_dir, side, state, frame, evaluation)
                for side, state, frame in [('reference', reference, reference_frame),
                                           ('target', target, target_frames[0])]}
    images, cameras = [], []
    records = Records(Path(receipt["index"]))
    try:
        for name, state, frame, offset, copy_name in selected:
            validate_frame(state, frame, source_frames[frame['proxy_frame_index']])
            image = read_image(sample_dir / copy_name) if copy_name else decoded[rgb_file_index(frame)]
            h, w = image.shape[:2]
            camera = camera_record(records, state, frame, w, h, origin, receipt)
            camera.update({"image": name, "image_path": str(input_dir / name),
                           "image_size_wh": [w, h], "target_clip_frame_index": offset})
            camera["source"]["state_json"] = str(sample_dir / (
                "reference_state.json" if offset is None else "target_state.json"))
            if copy_name:
                original = decoded[rgb_file_index(frame)]
                if original.shape != image.shape:
                    raise ValueError("Sample PNG and source RGB dimensions differ")
                delta = image.astype(float) - original.astype(float)
                if np.any(delta):
                    raise ValueError('Sample PNG differs from source RGB frame')
                camera["source"]["copied_rgb_png"] = str(sample_dir / copy_name)
                camera["source"]["png_vs_source_rgb_mse"] = float(np.mean(delta ** 2))
                camera["source"]["png_vs_source_rgb_max_abs"] = float(np.max(np.abs(delta)))
            images.append((name, image, copy_name))
            cameras.append(camera)
    finally:
        records.close()

    payload = {
        "schema_version": "1.0.0", "sample_id": sample["sample_id"],
        "usage": "Evaluation only. Never pass this file or whitebox images to VGGT.",
        "input_order": [row[0] for row in selected],
        "source": {"pi_source_json": str(source_path), "sample_json": str(sample_path),
                   "target_frame_states_json": str(sample_dir / "target_frame_states.json"),
                   "record_index_receipt": str(receipt_path),
                   "decode_method": "Sequential source MKV decode; zero-based file-local indices",
                   "training_status": sample["training_status"]},
        "estimated_alignment": True, "alignment_quality": "estimated",
        "alignment_note": "Observed game cameras interpolated at RGB times; not exact synchronized GT.",
        "coordinates": {
            "region_json": str(region_path), "origin_gta": origin.tolist(),
            "world_to_local": region["world_to_local"],
            "world": "Existing urban_region Godot Y-up local coordinates; source world units",
            "camera": "OpenCV: x right, y down, z forward; X_camera = R @ X_local + t",
            "K": "Original image pixels with integer pixel centers; normalized screen uv maps to uv*[W,H]-0.5",
            "K_pixel_edges": "Alternative screen coordinates uv*[W,H], matching project_box",
            "projection": "pixel_homogeneous = K @ (R @ X_local + t)",
            "godot_camera_to_world": "4x4 row-major array; Godot camera x right, y up, looks along -z",
            "derivation": "camera_geometry linear position / normalized ray interpolation, then RQ decomposition",
        },
        "whitebox": whitebox,
        "capture_canvas": {
            "recorded_render_resolution": cameras[1]['source']['client_brackets'][0]['camera']['render_resolution'],
            "exported_rgb_resolution": cameras[1]['image_size_wh'],
            "projection_policy": "Preserve recorded normalized rays and independent fx/fy; no crop, warp, or equal-focal correction",
            "rgb_capture_transform_verified": False,
            "note": "Ray self-consistency does not establish pixel alignment to recorded RGB or visual-mesh fidelity",
        },
        "cameras": cameras,
    }
    input_dir.mkdir(parents=True, exist_ok=True)
    evaluation.mkdir(parents=True, exist_ok=True)
    for name, image, copy_name in images:
        if copy_name:
            shutil.copyfile(sample_dir / copy_name, input_dir / name)
        else:
            save_image(input_dir / name, image)
    for side in ("reference", "target"):
        shutil.copyfile(sample_dir / f"{side}_whitebox.png", evaluation / f"{side}_whitebox.png")
    pair = np.hstack([images[1][1], read_image(evaluation / "target_whitebox.png")])
    comparison = np.full((pair.shape[0] + 40, pair.shape[1], 3), 255, dtype=np.uint8)
    comparison[40:] = pair
    for x, label in [(10, 'Source RGB'), (images[1][1].shape[1] + 10, 'Dataset proxy (NOT aligned GT)')]:
        cv2.putText(comparison, label, (x, 27), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 1)
    save_image(evaluation / "target_rgb_whitebox.jpg", comparison)
    (evaluation / "cameras.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"input": str(input_dir), "images": payload["input_order"],
                      "evaluation": str(evaluation / "cameras.json"),
                      "max_ray_reprojection_error_px": max(c["max_ray_reprojection_error_px"]
                                                           for c in cameras)}, indent=2))


if __name__ == "__main__":
    main()
