"""Real, sequential GroundingDINO -> SAM2 -> DINOv2 smoke test."""
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import time
import sys

os.environ.setdefault("HF_HOME", "/mnt/hdd/agentic-world-models")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="/home/star/minimax-h3/run/data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Ref2VA/0.png")
    parser.add_argument("--video", help="Extract the middle frame of an existing real video instead.")
    parser.add_argument("--output", default="/home/star/agentic_world/outputs/perception_smoke")
    parser.add_argument("--text", default="person. car. tree. building. road. dog.")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hud-profile", help="Recording-specific invalid screen-region profile.")
    parser.add_argument("--assert-detections", action="store_true",
                        help="Smoke-test assertion only; normal inference permits empty observations.")
    args = parser.parse_args()
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor

    torch.set_num_threads(4)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    models = json.loads(Path("/mnt/hdd/agentic-world-models/models.json").read_text())
    frame_id, timestamp = 0, 0.0
    if args.video:
        capture = cv2.VideoCapture(args.video)
        frame_id = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) // 2
        fps = capture.get(cv2.CAP_PROP_FPS)
        timestamp = frame_id / fps if fps > 0 else None
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(f"Cannot decode video: {args.video}")
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        image = Image.open(args.image).convert("RGB")
    hud_mask = None
    rejected_hud = []
    if args.hud_profile:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from perception.hud import load_profile, invalid_mask, clean_reference, overlap_fraction
        profile = load_profile(args.hud_profile)
        image.save(out / "input_raw.png")
        hud_mask = invalid_mask(np.asarray(image).shape, profile, margin=3)
        Image.fromarray(hud_mask).save(out / "invalid_pixels.png")
        image = Image.fromarray(clean_reference(np.asarray(image), hud_mask))
    image.save(out / "input.png")
    timings = {}
    start = time.monotonic()
    path = models["detector"]["path"]
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(path, local_files_only=True).to(args.device).eval()
    inputs = processor(images=image, text=args.text, return_tensors="pt").to(args.device)
    with torch.inference_mode():
        outputs = model(**inputs)
    detected = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=args.threshold,
        text_threshold=0.25, target_sizes=[(image.height, image.width)])[0]
    boxes = detected["boxes"].cpu().numpy()
    scores = detected["scores"].cpu().tolist()
    labels = detected["text_labels"]
    if hud_mask is not None:
        keep = []
        for i, box in enumerate(boxes):
            fraction = overlap_fraction(box, hud_mask)
            if fraction > 0:
                rejected_hud.append({"bbox_xyxy": box.tolist(), "category": labels[i],
                                     "invalid_fraction": fraction, "reason": "bbox_intersects_invalid_pixels"})
            else:
                keep.append(i)
        boxes = boxes[keep]
        scores = [scores[i] for i in keep]
        labels = [labels[i] for i in keep]
    timings["detector_sec"] = time.monotonic() - start
    del model, processor, inputs, outputs, detected
    gc.collect()
    torch.cuda.empty_cache()

    array = np.array(image)
    masks = np.empty((0, image.height, image.width), dtype=bool)
    mask_scores = []
    features = np.empty((0, 384), dtype=np.float32)
    timings.update(segmenter_sec=0.0, embedding_sec=0.0)
    if len(boxes):
        start = time.monotonic()
        path = models["segmenter"]["path"]
        processor = Sam2Processor.from_pretrained(path, local_files_only=True)
        model = Sam2Model.from_pretrained(path, local_files_only=True).to(args.device).eval()
        inputs = processor(images=image, input_boxes=[boxes.tolist()], return_tensors="pt").to(args.device)
        with torch.inference_mode():
            outputs = model(**inputs, multimask_output=False)
        masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
        masks = masks[:, 0].numpy().astype(bool)
        if hud_mask is not None:
            masks[:, hud_mask>0] = False
        mask_scores = outputs.iou_scores.cpu().flatten().tolist()
        assert len(masks) == len(boxes) and all(mask.any() for mask in masks)
        timings["segmenter_sec"] = time.monotonic() - start
        del model, processor, inputs, outputs
        gc.collect()
        torch.cuda.empty_cache()

        start = time.monotonic()
        path = models["embedding"]["path"]
        processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
        model = AutoModel.from_pretrained(path, local_files_only=True).to(args.device).eval()
        crops = []
        for i, (box, mask) in enumerate(zip(boxes, masks)):
            cv2.imwrite(str(out / f"mask_{i:03d}.png"), mask.astype(np.uint8) * 255)
            masked = array.copy()
            masked[~mask] = 127
            x1, y1, x2, y2 = box
            bounds = (max(0, int(x1)), max(0, int(y1)),
                      min(image.width, int(np.ceil(x2))), min(image.height, int(np.ceil(y2))))
            crop = Image.fromarray(masked).crop(bounds)
            crop.save(out / f"crop_{i:03d}.png")
            crops.append(crop)
        inputs = processor(images=crops, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            features = model(**inputs).last_hidden_state[:, 0]
            features = torch.nn.functional.normalize(features.float(), dim=-1).cpu().numpy()
        timings["embedding_sec"] = time.monotonic() - start
    assert features.shape == (len(boxes), 384) and np.isfinite(features).all()
    np.save(out / "embeddings.npy", features)
    observations = []
    overlay = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    for i, (box, mask) in enumerate(zip(boxes, masks)):
        color = ((67 + i * 53) % 255, (173 + i * 31) % 255, (101 + i * 97) % 255)
        overlay[mask] = (overlay[mask] * 0.6 + np.array(color) * 0.4).astype(np.uint8)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, f"{i}: {labels[i]} {scores[i]:.2f}", (x1, max(15, y1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        observations.append({
            "observation_id": f"smoke_{i:03d}", "track_id": None,
            "category": labels[i], "bbox_xyxy": box.tolist(),
            "confidence": scores[i], "mask_path": str(out / f"mask_{i:03d}.png"),
            "mask_predicted_iou": mask_scores[i], "mask_area": int(mask.sum()),
            "embedding_path": str(out / "embeddings.npy"), "embedding_index": i,
            "frame_id": frame_id, "timestamp": timestamp, "depth": None, "world_position": None,
            "source": "grounding-dino-tiny+sam2.1-hiera-tiny+dinov2-small",
        })
    cv2.imwrite(str(out / "overlay.png"), overlay)
    report = {
        "success": True, "image": args.video or args.image, "image_size": image.size,
        "device": args.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "versions": {name: importlib.metadata.version(name) for name in
                     ["torch", "torchvision", "transformers", "huggingface-hub", "numpy", "opencv-python-headless", "Pillow"]},
        "torch_path": torch.__file__, "models": models, "timings": timings,
        "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
        "embedding_shape": list(features.shape), "observations": observations,
        "hud_profile": args.hud_profile, "rejected_hud_proposals": rejected_hud,
        "inpainted_pixels_are_evidence": False if hud_mask is not None else None,
    }
    (out / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if args.assert_detections and not observations:
        raise AssertionError("Smoke test requires detections; inference result was saved successfully.")


if __name__ == "__main__":
    main()
