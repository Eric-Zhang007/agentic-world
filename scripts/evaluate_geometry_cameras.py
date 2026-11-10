"""Evaluate saved VGGT cameras without fitting a world alignment or running a model."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
URBAN = ROOT / "outputs/geometry_init/urban"


def basename(path):
    return str(path).replace("\\", "/").rsplit("/", 1)[-1]


def rotation(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("Invalid camera rotation")
    if not np.allclose(value @ value.T, np.eye(3), atol=1e-4) or not np.isclose(
            np.linalg.det(value), 1., atol=1e-4):
        raise ValueError("Camera rotation is not SO(3)")
    # Remove only floating-point orthogonality error before computing angles.
    u, _, vt = np.linalg.svd(value)
    return u @ vt


def direction_angle(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if min(na, nb) <= 1e-8:
        return None
    a, b = a / na, b / nb
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)),
                                      np.clip(np.dot(a, b), -1., 1.))))


def relative_metrics(first, second):
    values = {}
    for kind in ("predicted", "gt"):
        ri, ti = first[kind]
        rj, tj = second[kind]
        relative_r = rj @ ri.T
        relative_t = tj - relative_r @ ti
        ci, cj = -ri.T @ ti, -rj.T @ tj
        values[kind] = (relative_r, relative_t, ri @ (cj - ci))
    rp, tp, dp = values["predicted"]
    rg, tg, dg = values["gt"]
    delta = rp @ rg.T
    sine = np.linalg.norm([delta[2, 1] - delta[1, 2],
                           delta[0, 2] - delta[2, 0], delta[1, 0] - delta[0, 1]]) / 2
    angle = float(np.degrees(np.arctan2(sine, np.clip((np.trace(delta) - 1) / 2, -1, 1))))
    bp, bg = float(np.linalg.norm(dp)), float(np.linalg.norm(dg))
    valid = min(bp, bg) > 1e-8
    return {
        "from_image": first["image"], "to_image": second["image"],
        "relative_rotation_error_deg": angle,
        "relative_translation_direction_error_deg": direction_angle(tp, tg),
        "camera_displacement_direction_error_deg": direction_angle(dp, dg),
        "predicted_relative_R": rp.tolist(), "gt_relative_R": rg.tolist(),
        "predicted_relative_t": tp.tolist(), "gt_relative_t": tg.tolist(),
        "predicted_displacement_in_first_camera": dp.tolist(),
        "gt_displacement_in_first_camera": dg.tolist(),
        "predicted_baseline_model_units": bp, "gt_baseline_world_units": bg,
        "baseline_derived_scale_world_units_per_model_unit": bg / bp if valid else None,
        "baseline_status": "valid" if valid else "degenerate_baseline_direction_and_scale_undefined",
    }


def pixel_difference(a, b):
    if a.shape != b.shape:
        raise ValueError(f"Pixel comparison shape mismatch: {a.shape} vs {b.shape}")
    delta = a.astype(float) - b.astype(float)
    mse = float(np.mean(delta ** 2))
    return {"shape": list(a.shape), "mse_255": mse,
            "mae_255": float(np.mean(np.abs(delta))),
            "max_abs_255": float(np.max(np.abs(delta))),
            "psnr_db": float(10 * np.log10(255 ** 2 / mse)) if mse else None,
            "pixel_identical": mse == 0}


def verify_video_frame(path, gt, metadata, mapping, spec):
    import cv2
    from PIL import Image

    name, video, index_text, png = spec
    index = int(index_text)
    if index < 0 or name not in mapping:
        raise ValueError("Video verification requires an explicit image mapping and nonnegative frame index")
    matches = [i for i, p in enumerate(metadata["inputs"]) if basename(p) == name]
    if len(matches) != 1:
        raise ValueError("Video verification input must occur exactly once in this NPZ")
    input_index = matches[0]
    camera = next(c for c in gt["cameras"] if c["image"] == mapping[name])
    states_path = Path(gt["source"]["target_frame_states_json"])
    states = json.loads(states_path.read_text(encoding="utf-8"))
    row = states[index]
    source_index = row.get("rgb_segment_frame_index", row["rgb_decode_index"])
    if source_index != camera["source"]["source_video_frame_index"]:
        raise ValueError("Clip frame-to-source index does not match mapped GT camera")
    if Path(video).resolve() != (Path(gt["source"]["sample_json"]).parent / "target_rgb.mp4").resolve():
        raise ValueError("Verification video is not this sample's target_rgb.mp4")

    def decode(video_path, frame_index):
        capture = cv2.VideoCapture(str(video_path))
        try:
            for _ in range(frame_index + 1):
                ok, image = capture.read()
                if not ok:
                    raise ValueError(f"Cannot decode frame {frame_index} from {video_path}")
            return image
        finally:
            capture.release()

    clip_image = decode(video, index)
    source_image = decode(camera["source"]["rgb_video"], source_index)
    input_image = cv2.imread(str(png))
    evaluation_image = cv2.imread(camera["image_path"])
    if input_image is None or evaluation_image is None:
        raise ValueError("Missing local RGB for source verification")
    prep = metadata["preprocessing"]["images"][input_index]
    image = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    image = image.resize(tuple(prep["resized_wh"]), Image.Resampling.BICUBIC)
    top = prep["crop_top"]
    left_pad, top_pad, right_pad, bottom_pad = prep["padding_ltrb"]
    width, height = prep["output_wh"]
    crop_height = height - top_pad - bottom_pad
    image = image.crop((0, top, image.width, top + crop_height))
    processed = np.pad(np.asarray(image), ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
                       constant_values=255)
    with np.load(path, allow_pickle=False) as archive:
        saved = np.moveaxis(archive["images"][input_index], 0, -1) * 255.
    comparisons = {
        "local_input_png_vs_training_mp4_frame": pixel_difference(input_image, clip_image),
        "evaluation_png_vs_original_mkv_frame": pixel_difference(evaluation_image, source_image),
        "training_mp4_frame_vs_original_mkv_frame": pixel_difference(clip_image, source_image),
        "local_input_after_recorded_preprocessing_vs_npz_images": pixel_difference(processed, saved),
    }
    if not comparisons["evaluation_png_vs_original_mkv_frame"]["pixel_identical"]:
        raise ValueError("Evaluation RGB does not exactly match its declared original source frame")
    if comparisons["local_input_after_recorded_preprocessing_vs_npz_images"]["max_abs_255"] > .01:
        raise ValueError("Local input PNG does not reproduce the NPZ input pixels")
    return {"npz_input_name": name, "mapped_gt_image": mapping[name],
            "mapping_basis": "Explicit CLI mapping, NPZ provenance, target_frame_states index, and pixel comparisons",
            "npz_input_provenance": metadata.get("input_provenance"),
            "local_input_png": str(Path(png).resolve()), "training_video": str(Path(video).resolve()),
            "training_video_frame_index_zero_based": index,
            "target_frame_states_json": str(states_path), "source_frame_record": row,
            "original_video": camera["source"]["rgb_video"], "original_video_frame_index": source_index,
            "evaluation_png": camera["image_path"], "comparisons": comparisons,
            "interpretation": "Same frame identity established by provenance/index mapping; pixel metrics quantify decoding/re-encoding differences, not exact RGB equivalence."}


def evaluate(path, gt, mapping=None, verifications=()):
    mapping = mapping or {}
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].item())
        intrinsics = np.asarray(archive["intrinsic"], dtype=float)
        extrinsics = np.asarray(archive["extrinsic"], dtype=float)
    inputs = metadata["inputs"]
    preprocessing = metadata["preprocessing"]["images"]
    n = len(inputs)
    if n < 1 or intrinsics.shape != (n, 3, 3) or extrinsics.shape != (n, 3, 4):
        raise ValueError(f"Camera array/input shape mismatch: {path}")
    if len(preprocessing) != n or not all(np.isfinite(a).all() for a in (intrinsics, extrinsics)):
        raise ValueError("Invalid preprocessing count or nonfinite camera values")
    lookup = {c["image"]: c for c in gt["cameras"]}
    if len(lookup) != len(gt["cameras"]):
        raise ValueError("Duplicate GT image names")
    provenance = metadata.get("input_provenance", "")
    if provenance and gt["sample_id"] not in provenance:
        raise ValueError("NPZ provenance refers to a different sample")
    frames, poses = [], []
    for index, (input_path, prep, predicted_k, extrinsic) in enumerate(
            zip(inputs, preprocessing, intrinsics, extrinsics)):
        name = basename(input_path)
        match = "unique_input_basename"
        if name in mapping:
            name, match = mapping[name], "explicit_cli_image_map"
        elif name == "input.png" and gt["sample_id"] in provenance and "target_rgb" in provenance:
            name, match = "target_rgb.png", "legacy_input_png_explicit_target_rgb_provenance"
        if name not in lookup or basename(prep["path"]) != basename(input_path):
            raise ValueError(f"Cannot unambiguously match input/preprocessing to GT: {input_path}")
        camera = lookup[name]
        if list(prep["original_wh"]) != list(camera["image_size_wh"]):
            raise ValueError(f"Original image dimensions differ: {name}")
        transform = np.asarray(prep["original_to_processed_pixel_centers"], dtype=float)
        if transform.shape != (3, 3) or not np.isfinite(transform).all():
            raise ValueError("Invalid original-to-processed pixel transform")
        if not np.allclose(transform[2], [0, 0, 1]) or abs(np.linalg.det(transform)) < 1e-12:
            raise ValueError("Singular or non-affine preprocessing transform")
        gt_original = np.asarray(camera["K"], dtype=float)
        gt_processed = transform @ gt_original
        if not np.isfinite(gt_processed).all() or np.any(np.diag(gt_processed)[:2] <= 0):
            raise ValueError("Invalid GT focal lengths")
        predicted_original = np.linalg.solve(transform, predicted_k)
        focal_errors = {}
        for axis, k in (("fx", 0), ("fy", 1)):
            expected, actual = float(gt_processed[k, k]), float(predicted_k[k, k])
            if actual <= 0:
                raise ValueError("Nonpositive predicted focal length")
            signed = (actual - expected) / expected
            focal_errors[axis] = {"gt_processed_px": expected, "predicted_processed_px": actual,
                                  "signed_relative_error": signed, "absolute_relative_error": abs(signed),
                                  "absolute_relative_error_percent": abs(signed) * 100}
        frames.append({"image_index": index, "image": name, "npz_input_path": input_path,
                       "gt_match": match, "original_wh": prep["original_wh"],
                       "processed_wh": prep["output_wh"], "original_to_processed": transform.tolist(),
                       "gt_K_original": gt_original.tolist(), "gt_K_processed": gt_processed.tolist(),
                       "predicted_K_processed": predicted_k.tolist(),
                       "predicted_K_mapped_to_original": predicted_original.tolist(),
                       "focal_errors": focal_errors})
        poses.append({"image": name, "predicted": (rotation(extrinsic[:, :3]), extrinsic[:, 3]),
                      "gt": (rotation(camera["R"]), np.asarray(camera["t"], dtype=float))})
    if len({p["image"] for p in poses}) != n:
        raise ValueError("Duplicate matched input images")
    pairs = [relative_metrics(a, b) for a, b in combinations(poses, 2)]
    summary = {axis + "_mean_absolute_relative_error_percent": float(np.mean([
        f["focal_errors"][axis]["absolute_relative_error_percent"] for f in frames]))
               for axis in ("fx", "fy")}
    for key in ("relative_rotation_error_deg", "relative_translation_direction_error_deg",
                "camera_displacement_direction_error_deg"):
        valid = [p[key] for p in pairs if p[key] is not None]
        summary["mean_" + key] = float(np.mean(valid)) if valid else None
    checks = [verify_video_frame(path, gt, metadata, mapping, spec) for spec in verifications
              if spec[0] in {basename(p) for p in inputs}]
    return {"npz": str(path.resolve()), "image_count": n,
            "input_provenance": provenance, "model": metadata.get("model"),
            "vendor_commit": metadata.get("vendor_commit"),
            "explicit_image_mapping": mapping,
            "source_verification": checks,
            "gt_matching_note": "Exact names, recorded legacy provenance, or explicit CLI mapping; pixel verification only for entries in source_verification",
            "frames": frames, "pairs": pairs, "summary": summary,
            "relative_pose_status": "evaluated_all_pairs" if pairs else "undefined_for_single_camera",
            "scale_status": "pairwise_baseline_derived_evaluation_only" if pairs else "unobservable_from_single_camera"}


def self_test():
    def rz(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    q, shift, scale = rz(.7), np.array([4., -2., 3.]), 5.
    poses = []
    for i, center in enumerate((np.zeros(3), np.array([1., 2., .5]), np.array([-2., 1., 3.]))):
        r = rz(i * .2)
        rp, cp = r @ q.T, scale * q @ center + shift
        poses.append({"image": str(i), "gt": (r, -r @ center), "predicted": (rp, -rp @ cp)})
    for a, b in combinations(poses, 2):
        result = relative_metrics(a, b)
        for key in ("relative_rotation_error_deg", "relative_translation_direction_error_deg",
                    "camera_displacement_direction_error_deg"):
            assert result[key] < 1e-6, (key, result)
        np.testing.assert_allclose(result["baseline_derived_scale_world_units_per_model_unit"], 1 / scale)
    a = {"image": "a", "gt": (np.eye(3), np.zeros(3)), "predicted": (np.eye(3), np.zeros(3))}
    b = {"image": "b", "gt": (np.eye(3), np.array([-1., 0., 0.])),
         "predicted": (np.eye(3), np.array([1., 0., 0.]))}
    assert abs(relative_metrics(a, b)["relative_translation_direction_error_deg"] - 180) < 1e-8
    assert relative_metrics(a, a)["baseline_derived_scale_world_units_per_model_unit"] is None
    b["predicted"] = (rz(np.pi / 6), np.array([1., 0., 0.]))
    np.testing.assert_allclose(relative_metrics(a, b)["relative_rotation_error_deg"], 30., atol=1e-8)
    k = np.array([[800., 0., 639.5], [0., 900., 359.5], [0., 0., 1.]])
    transform = np.array([[.4, 0., -.3], [0., .5, 9.75], [0., 0., 1.]])
    mapped = transform @ k
    np.testing.assert_allclose([mapped[0, 0], mapped[1, 1]], [320., 450.])
    np.testing.assert_allclose(np.linalg.solve(transform, mapped), k)
    print("PASS: gauge/scale invariance, direction sign, rotation error, zero baseline, pixel transform")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", choices=["single", "joint", "joint3"], default=["single", "joint"])
    parser.add_argument("--reconstruction", type=Path, default=URBAN / "reconstruction")
    parser.add_argument("--cameras", type=Path, default=URBAN / "evaluation/cameras.json")
    parser.add_argument("--output", type=Path, help="Default: evaluation/camera_metrics_<runs>.json")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic metric tests without writing results")
    parser.add_argument("--image-map", action="append", default=[], metavar="INPUT_NAME=GT_NAME",
                        help="Explicit NPZ input basename to GT image mapping; repeatable")
    parser.add_argument("--verify-video-frame", action="append", nargs=4, default=[],
                        metavar=("INPUT_NAME", "TRAINING_MP4", "ZERO_BASED_FRAME", "LOCAL_INPUT_PNG"),
                        help="Record source frame and pixel verification for an explicitly mapped input")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    gt = json.loads(args.cameras.read_text(encoding="utf-8"))
    mapping = {}
    for entry in args.image_map:
        source, separator, target = entry.partition("=")
        if not separator or not source or not target or source in mapping:
            parser.error("--image-map requires unique INPUT_NAME=GT_NAME entries")
        if target not in {c["image"] for c in gt["cameras"]}:
            parser.error(f"Unknown GT image: {target}")
        mapping[source] = target
    requested_inputs = set()
    for name in args.runs:
        with np.load(args.reconstruction / (name + ".npz"), allow_pickle=False) as archive:
            requested_inputs.update(basename(p) for p in json.loads(archive["metadata"].item())["inputs"])
    if set(mapping) - requested_inputs or any(spec[0] not in requested_inputs for spec in args.verify_video_frame):
        parser.error("Mapping or verification contains an input absent from selected runs")
    results = {name: evaluate(args.reconstruction / (name + ".npz"), gt, mapping, args.verify_video_frame)
               for name in args.runs}
    output = args.output or args.cameras.parent / ("camera_metrics_" + "_".join(args.runs) + ".json")
    payload = {"schema_version": "1.0.0", "gt_cameras": str(args.cameras.resolve()),
               "estimated_alignment": gt.get("estimated_alignment"),
               "gt_alignment_quality": gt.get("alignment_quality"),
               "conventions": {
                   "intrinsics": "K_gt_processed = original_to_processed_pixel_centers @ K_gt_original",
                   "focal_error": "abs(predicted - GT) / GT; signed error also retained",
                   "relative_pose": "World-to-camera: R_ji = R_j R_i.T; t_ji = t_j - R_ji t_i",
                   "translation_direction": "Oriented angle 0..180 degrees between relative t vectors in camera j",
                   "camera_displacement": "Additional oriented angle between R_i(C_j-C_i) vectors in camera i",
                   "scale": "Each pair: GT baseline / predicted baseline, world units per model unit; evaluation only, never model input",
                   "position_error": "Not reported. No Sim3 fit and no two-camera scale-fitted position score.",
                   "single": "Intrinsics only; relative pose and metric scale undefined",
                   "rotation_numeric_handling": "SO(3) validated to 1e-4; SVD removes floating-point orthogonality error",
               }, "runs": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": {k: v["summary"] for k, v in results.items()}}, indent=2))


if __name__ == "__main__":
    main()
