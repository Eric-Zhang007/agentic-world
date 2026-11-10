"""Compare prompt order and singleton queries without treating inferred geometry as truth."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("HF_HOME", "/mnt/hdd/agentic-world-models")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import save_json


def run(images, output):
    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    torch.set_num_threads(4)
    paths = json.loads(Path("/mnt/hdd/agentic-world-models/models.json").read_text())
    processor = AutoProcessor.from_pretrained(paths["detector"]["path"], local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        paths["detector"]["path"], local_files_only=True).to("cuda:0").eval()
    labels = ["car", "person", "tree", "dog", "bicycle", "truck", "bus", "motorcycle",
              "building", "traffic light", "bench"]
    prompts = {"joint": ". ".join(labels)+".", "reversed": ". ".join(reversed(labels))+".",
               **{f"single_{label}": label+"." for label in ("car", "person", "tree", "traffic light")}}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = []
    for index, image_path in enumerate(images):
        image = Image.open(image_path).convert("RGB")
        entry = {"source": image_path, "ground_truth_available": False, "runs": {}}
        for name, prompt in prompts.items():
            inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                result = model(**inputs)
            detected = processor.post_process_grounded_object_detection(
                result, inputs.input_ids, threshold=.25, text_threshold=.25,
                target_sizes=[(image.height, image.width)])[0]
            rows = [{"category": category, "bbox": box, "confidence": score}
                    for category, box, score in zip(detected["text_labels"],
                        detected["boxes"].cpu().tolist(), detected["scores"].cpu().tolist())]
            entry["runs"][name] = rows
            overlay = image.copy()
            draw = ImageDraw.Draw(overlay)
            for j, row in enumerate(rows):
                draw.rectangle(row["bbox"], outline="red", width=2)
                draw.text((5, 5+j*15), f"{row['category']} {row['confidence']:.3f}", fill="red")
            overlay.save(output / f"{index}_{name}.png")
        report.append(entry)
    save_json(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.images, args.output)
