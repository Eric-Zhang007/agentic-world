"""Run official VGGT on ordered RGB images without changing the host environment."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def preprocessing_records(paths, mode, output_hw):
    from PIL import Image

    records = []
    out_h, out_w = output_hw
    for path in paths:
        with Image.open(path) as image:
            width, height = image.size
        if mode == "pad" and height > width:
            rh, rw = 518, round(width * 518 / height / 14) * 14
        else:
            rw, rh = 518, round(height * 518 / width / 14) * 14
        crop_top = (rh - 518) // 2 if mode == "crop" and rh > 518 else 0
        cropped_h = min(rh, 518) if mode == "crop" else rh
        left, top = (out_w - rw) // 2, (out_h - cropped_h) // 2
        sx, sy = rw / width, rh / height
        records.append({
            "path": str(path), "original_wh": [width, height],
            "resized_wh": [rw, rh], "crop_top": crop_top,
            "padding_ltrb": [left, top, out_w - rw - left, out_h - cropped_h - top],
            "output_wh": [out_w, out_h],
            "original_to_processed_pixel_centers": [
                [sx, 0, (sx - 1) / 2 + left],
                [0, sy, (sy - 1) / 2 + top - crop_top], [0, 0, 1]],
            "original_to_processed_pixel_edges": [
                [sx, 0, left], [0, sy, top - crop_top], [0, 0, 1]],
        })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="RGB paths in sequence order")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", default="2", help="Physical GPU index")
    parser.add_argument("--vendor", type=Path,
                        default=Path(__file__).resolve().parents[1] / "vendor/vggt")
    parser.add_argument("--model", default="facebook/VGGT-1B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--mode", choices=["crop", "pad"], default="crop")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--input-provenance", help="Source episode/segment/frame description")
    args = parser.parse_args()
    paths = [path.resolve(strict=True) for path in args.images]
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    sys.path.insert(0, str(args.vendor.resolve()))

    import numpy as np
    import torch
    from PIL import Image
    from huggingface_hub import snapshot_download
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this inference run")
    started = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    print(f"Loading {len(paths)} RGB images on physical GPU {args.gpu}", flush=True)
    images = load_and_preprocess_images([str(p) for p in paths], mode=args.mode)
    transforms = preprocessing_records(paths, args.mode, list(images.shape[-2:]))
    rgb_dir = args.output.parent / (args.output.stem + "_rgb")
    rgb_dir.mkdir(parents=True, exist_ok=True)
    processed_rgb_paths = []
    for index, (path, tensor) in enumerate(zip(paths, images)):
        rgb_path = rgb_dir / f"{index:03d}_{path.stem}.png"
        pixels = np.rint(tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(pixels).save(rgb_path)
        processed_rgb_paths.append(str(rgb_path.resolve()))
    print("Resolving model cache; downloads use https://hf-mirror.com", flush=True)
    snapshot = snapshot_download(
        repo_id=args.model, revision=args.revision,
        allow_patterns=["config.json", "model.safetensors"],
        endpoint="https://hf-mirror.com", local_files_only=args.local_files_only,
    )
    if not (Path(snapshot) / "model.safetensors").is_file():
        raise RuntimeError(f"Missing model.safetensors in {snapshot}")
    model = VGGT.from_pretrained(snapshot, local_files_only=True).eval().to("cuda")
    loaded_seconds = time.perf_counter() - start
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    images = images.to("cuda")
    torch.cuda.synchronize()
    inference_start = time.perf_counter()
    print(f"Model loaded in {loaded_seconds:.2f}s; running {dtype}", flush=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        predictions = model(images)
    # Camera decoding and unprojection use float32 outside autocast.
    with torch.inference_mode():
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"].float(), images.shape[-2:])
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_start
    arrays = {
        "images": images.float().cpu().numpy(),
        "depth": predictions["depth"][0].float().cpu().numpy(),
        "depth_conf": predictions["depth_conf"][0].float().cpu().numpy(),
        "extrinsic": extrinsic[0].float().cpu().numpy(),
        "intrinsic": intrinsic[0].float().cpu().numpy(),
    }
    arrays["points"] = np.asarray(unproject_depth_map_to_point_map(
        arrays["depth"], arrays["extrinsic"], arrays["intrinsic"]), dtype=np.float32)
    for name, array in arrays.items():
        if not np.isfinite(array).all():
            raise RuntimeError(f"Nonfinite values in {name}")
    if not (arrays["depth"] > 0).all():
        raise RuntimeError("Nonpositive predicted depth")
    commit = subprocess.run(["git", "-C", str(args.vendor), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False).stdout.strip()
    metadata = {
        "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "requested_revision": args.revision,
        "model_snapshot": snapshot, "vendor_commit": commit,
        "download_endpoint": "https://hf-mirror.com",
        "inputs": [str(p) for p in paths], "input_provenance": args.input_provenance,
        "preprocessed_rgb_paths": processed_rgb_paths,
        "sam_alignment": "PNGs match images/depth/points pixel-for-pixel; no additional resize or crop",
        "preprocessing": {"function": "load_and_preprocess_images", "mode": args.mode,
                          "interpolation": "PIL BICUBIC", "images": transforms},
        "coordinates": {
            "intrinsic": "Processed image pixels; integer pixel centers, x right, y down",
            "extrinsic": "OpenCV world-to-camera [R|t]; camera x right, y down, z forward",
            "depth": "Camera z depth; predicted scale, not calibrated meters",
            "points": "Depth unprojected into common VGGT world frame; scale uncalibrated",
            "images": "Float32 RGB [0,1], NCHW after preprocessing",
        },
        "physical_gpu": args.gpu, "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "autocast_dtype": str(dtype),
        "timing_seconds": {"load_and_preprocess": loaded_seconds,
                           "inference_and_camera_decode": inference_seconds,
                           "before_save_total": time.perf_counter() - start},
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "arrays": {name: {"shape": list(a.shape), "dtype": str(a.dtype),
                          "min": float(a.min()), "max": float(a.max())}
                   for name, a in arrays.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(metadata, ensure_ascii=True, indent=2)
    np.savez_compressed(args.output, **arrays, metadata=np.asarray(serialized))
    args.output.with_suffix(".metadata.json").write_text(serialized + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "images": len(paths),
                      "inference_seconds": inference_seconds,
                      "shapes": {k: list(v.shape) for k, v in arrays.items()}}), flush=True)


if __name__ == "__main__":
    main()
