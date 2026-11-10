"""Compile an accepted Godot action into a reproducible temporal-control trial."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from renderer.temporal_control import temporal_schedule


def prepare(attempt, output, reference, frame_count=73, width=1024, height=768, depth_dir=None):
    output.mkdir(parents=True, exist_ok=False)
    timeline = json.loads((attempt / "timeline.json").read_text())
    scene = json.loads((attempt / "scene.json").read_text())
    action = json.loads((attempt / "action.json").read_text())
    # Include the actual initialized state, rather than calling the first simulated
    # tick t=0. Endpoint-preserving resampling is explicit in the manifest.
    originals = [attempt / "preview/camera.png"] + sorted((attempt / "preview").glob("frame_*.png"))
    if len(originals) != len(timeline["frames"]) + 1:
        raise ValueError("Expected all rendered timeline frames plus the initialized camera")
    indices = np.rint(np.linspace(0, len(originals) - 1, frame_count)).astype(int)
    images = [np.asarray(Image.open(originals[i]).convert("RGB").resize((width, height))) for i in indices]
    if depth_dir is not None:
        depth_files=[depth_dir/"camera.png"]+sorted(depth_dir.glob("frame_*.png"))
        if len(depth_files)!=len(originals):raise ValueError("Depth and preview timeline lengths differ")
        controls=np.stack([np.asarray(Image.open(depth_files[i]).convert("RGB").resize((width,height))) for i in indices])
        control_kind="Godot relative camera depth, near white; no preview grid/shadows"
    else:
        controls = np.stack([np.repeat(cv2.Canny(image, 100, 200)[..., None], 3, axis=2) for image in images])
        control_kind="Canny from debug preview; potential grid/shadow contamination"
    np.savez_compressed(output / "control.npz", rgb=controls)
    sweep_phase = next(i for i, phase in enumerate(action["phases"]) if phase["type"] == "sweep")
    sweep_indices = [i + 1 for i, frame in enumerate(timeline["frames"]) if frame["phase"] == sweep_phase]
    key_source = [0, sweep_indices[len(sweep_indices)//2], sweep_indices[-1], len(originals)-1]
    keyframes = sorted(set(int(np.argmin(abs(indices - index))) for index in key_source))
    cases = [{"label": label, "seed": seed, "temporal_keyframes": keys,
              "controls": [{"path": str((output / "control.npz").resolve()), "strength": 0.85}],
              "cache_threshold": 0.0}
             for seed in (43, 71)
             for label, keys in ((f"dense_s{seed}", None), (f"events_s{seed}", keyframes),
                                 (f"endpoints_s{seed}", [0, frame_count - 1]))]
    (output / "cases.json").write_text(json.dumps(cases, indent=2))
    prompt = (
        "A continuous realistic fantasy action game shot matching the monkey warrior, tiger guardian, "
        "stone temple courtyard, lighting and costumes of <Picture 1>. Fixed camera with no cut or zoom. "
        "The monkey remains on the near left and the tiger on the far right. Over approximately three seconds, "
        "the monkey makes one short lateral step, performs exactly one broad horizontal two-handed staff sweep "
        "toward the tiger, then smoothly returns to a balanced ready stance while keeping the staff. "
        "The tiger actively tracks and guards with small independent balancing steps, remains standing, "
        "and does not collapse or take invented damage. Enrich this single action with convincing shoulder, "
        "hip and wrist coordination, weight transfer, natural follow-through, breathing and cloth motion. "
        "No additional strike, spin, jump, roll, teleportation, extra character, dropped weapon or scene change. "
        "The geometric control provides coarse motion and scene placement, not the final mannequin appearance."
    )
    (output / "prompt.txt").write_text(prompt)
    expected = []
    for j, source_index in enumerate(indices):
        frame = timeline["frames"][source_index-1] if source_index else scene
        expected.append({"frame": j, "video_time": j/24, "source_index": int(source_index),
                         "source_time": frame.get("time", 0),
                         "camera": frame.get("camera", scene["camera"]), "entities": frame["entities"]})
    manifest = {"attempt": str(attempt), "reference": str(reference), "action": action,
                "frames": frame_count, "width": width, "height": height, "fps": 24,
                "source_frame_indices": indices.tolist(), "keyframes": keyframes,
                "source_duration_sec": timeline["frames"][-1]["time"],
                "output_last_timestamp_sec": (frame_count-1)/24,
                "expected": expected,
                "method": "Experimental input/latent/residual temporal gating, not a trained sparse checkpoint",
                "control_kind":control_kind,
                "limits": ["VAE temporal compression limits anchor timing precision",
                           "Self-attention can propagate constraints beyond direct gated rows",
                           "Proxy body proportions differ from the appearance reference",
                           "Single-image inferred geometry is not metric ground truth",
                           "First RGB guide may disagree with inferred initial proxy pose"],
                "world_mutation": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    preview = output / "expected"
    preview.mkdir()
    sample_indices = sorted(set(np.rint(np.linspace(0, frame_count-1, 13)).astype(int).tolist() + keyframes))
    for j in sample_indices:
        Image.fromarray(images[j]).save(preview / f"{j:03d}.png")
    sheet = Image.new("RGB", (4*320, 3*268), "white")
    draw = ImageDraw.Draw(sheet)
    for row, keys in enumerate((None, keyframes, [0, frame_count-1])):
        schedule = temporal_schedule(frame_count, keys)
        draw.text((5, row*268+4), f"{('dense','events','endpoints')[row]} direct control: {schedule['active_fraction']:.1%}", fill="black")
        for col, j in enumerate(keyframes):
            thumbnail = Image.fromarray(controls[j] if j in schedule["active_frame_indices"] else np.zeros_like(controls[j]))
            thumbnail.thumbnail((320,240))
            sheet.paste(thumbnail, (col*320, row*268+25))
    sheet.save(output / "control_schedule.jpg")
    print(json.dumps({"output": str(output), "keyframes": keyframes, "cases": len(cases)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--attempt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--depth-dir",type=Path)
    a = p.parse_args()
    prepare(a.attempt, a.output, a.reference, depth_dir=a.depth_dir)
