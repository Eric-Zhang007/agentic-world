"""Create synchronized viewing artifacts without changing the world database."""
import argparse
import json
from pathlib import Path
import sys

import av
import numpy as np
from PIL import Image,ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_comfy_control import save_video


def main(root):
    root=root.resolve()
    manifest=json.loads((root/"manifest.json").read_text())
    metadata=json.loads((root/"render/metadata.json").read_text())
    rows=[]
    for run in metadata["runs"]:
        folder=root/"evaluation"/run["label"]
        review=json.loads((folder/"visual_review.json").read_text())
        metrics=json.loads((folder/"metrics.json").read_text())
        rows.append({"label":run["label"],"seed":run["seed"],"sample_seconds":run["sample_seconds"],
                     "control_fraction":run["temporal_schedule"]["active_fraction"],
                     "review":review,"endpoint_displacements":metrics["endpoint_displacements"],
                     "start_center_errors_px":{m["world_id"]:m["center_error_px"] for m in metrics["frames"][0]["matches"]},
                     "end_center_errors_px":{m["world_id"]:m["center_error_px"] for m in metrics["frames"][-1]["matches"]}})
        with av.open(str(root/"render"/f"{run['label']}.mp4")) as stream:
            all_frames=[frame.to_ndarray(format="rgb24") for frame in stream.decode(video=0)]
        contact=Image.new("RGB",(8*192,10*164),"white");draw=ImageDraw.Draw(contact)
        for index,pixels in enumerate(all_frames):
            x,y=(index%8)*192,(index//8)*164
            contact.paste(Image.fromarray(pixels).resize((192,144)),(x,y+20))
            draw.text((x+3,y+3),f"{index:02d} {index/24:.2f}s",fill="black")
        contact.save(folder/"all_frames.jpg")
    summary={"experiment":"fixed seed pairs, experimental temporal ControlNet gating",
             "runs":rows,"world_mutation":False,"limits":manifest["limits"]+[
                 "Two seeds are not a reliable success rate estimate",
                 "VLM sampled-frame scores are subjective, not human or 3D ground truth",
                 "Detector boxes sometimes merge body/weapon or miss the opponent; displacement errors are diagnostics, not calibrated endpoint measurements",
                 "No native RGB terminal guide; endpoint conditioning uses "+manifest["control_kind"]]}
    (root/"summary.json").write_text(json.dumps(summary,indent=2))
    attempt=Path(manifest["attempt"])
    if not attempt.is_absolute():attempt=Path(__file__).resolve().parents[1]/attempt
    originals=[attempt/"preview/camera.png"]+sorted((attempt/"preview").glob("frame_*.png"))
    proxy=[np.asarray(Image.open(originals[i]).convert("RGB")) for i in manifest["source_frame_indices"]]
    for seed in sorted(set(row["seed"] for row in rows)):
        labels=[f"{mode}_s{seed}" for mode in ("dense","events","endpoints")]
        clips=[proxy]
        for label in labels:
            with av.open(str(root/"render"/f"{label}.mp4")) as stream:
                clips.append([frame.to_ndarray(format="rgb24") for frame in stream.decode(video=0)])
        combined=[]
        for index in range(manifest["frames"]):
            sheet=Image.new("RGB",(1024,816),"black");draw=ImageDraw.Draw(sheet)
            for k,clip in enumerate(clips):
                x,y=(k%2)*512,(k//2)*408
                sheet.paste(Image.fromarray(clip[index]).resize((512,384)),(x,y+24))
                draw.text((x+8,y+6),f"{(['whitebox']+labels)[k]} | {index/24:.2f}s",fill="white")
            combined.append(np.asarray(sheet))
        save_video(np.stack(combined).astype(np.float32)/255,root/f"comparison_s{seed}.mp4")
        Image.fromarray(combined[manifest["keyframes"][1]]).save(root/f"comparison_s{seed}.jpg")
    lines=["# Temporal Motion Control Experiment", "", "Read-only trial; no rollout was committed to the persistent world.","",
           "| Case | Control coverage | Sampling seconds | Visual decision | Motion richness / 5 | Geometry / 5 |",
           "|---|---:|---:|---|---:|---:|"]
    for row in rows:
        r=row["review"]
        lines.append(f"| {row['label']} | {row['control_fraction']:.1%} | {row['sample_seconds']:.1f} | {r['decision']} | {r.get('motion_richness_1_to_5')} | {r.get('geometry_fidelity_1_to_5')} |")
    lines += ["","Scores are sampled-frame VLM judgments, not calibrated benchmark metrics.","", "## Per-case evidence",""]
    for row in rows:
        lines += [f"### {row['label']}","",json.dumps(row["review"],ensure_ascii=False,indent=2),""]
    lines += ["## Limitations",""]+["- "+limit for limit in summary["limits"]]
    (root/"report.md").write_text("\n".join(lines))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path)
    main(p.parse_args().root)
