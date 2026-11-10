"""Collect image-only detections, clock correspondences and world geometry separately."""
import argparse
import json
from pathlib import Path
import sys
import os

os.environ.setdefault("HF_HUB_OFFLINE","1")
import av
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from perception.hud import invalid_mask, load_profile, clean_reference, overlap_fraction
from scripts.audit_episode_alignment import rows, clock_time
from matching.costs import iou


def main(root, output, profile_path):
    audit=json.loads((output/"audit.json").read_text())
    world=[json.loads(x) for x in pq.read_table(root/"server/world_state.parquet")["payload_json"].to_pylist()]
    wt=np.array([r["server_monotonic_ns"] for r in world])/1e9
    manifest=json.loads(Path("/mnt/hdd/agentic-world-models/models.json").read_text())
    model_path=manifest["detector"]["path"]
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
    model=AutoModelForZeroShotObjectDetection.from_pretrained(model_path,local_files_only=True).to("cuda:0").eval()
    torch.set_num_threads(4)
    profile=load_profile(profile_path)
    results={"world":world[:500], "agents":{}, "detection_prompt":"person. car.", "threshold":0.3}
    for agent in ("agent_000","agent_001"):
        folder=root/"agents"/agent
        raw=rows(folder/"frame_timestamps.jsonl")
        pts=np.array([r["video_pts_ns"] for r in raw])/1e9
        rt=clock_time(audit["agents"][agent]["clock"],[r["sidecar_monotonic_ns"] for r in raw])
        # First 40s receipt jitter median eliminates showinfo read-batch jitter only.
        origin=float(np.median(rt[pts<40]-pts[pts<40]))
        clients=rows(root/".spool/agents"/agent/"client.jsonl")
        client_records=[]
        for r in clients:
            seq=r["sample_seq_estimate"]
            if seq>=500: break
            client_records.append({"time":float(wt[seq]), "camera":r["camera"], "seq":seq})
        frames=[]
        decoded_index=0
        segment_counts=[]
        for path in sorted(folder.glob("video_*.mkv")):
            segment=int(path.stem.split("_")[-1])
            with av.open(str(path)) as c:
                fps=float(c.streams.video[0].average_rate)
                if fps!=20:
                    raise ValueError("This prefix calibration expects verified CFR20 recordings")
                count=0
                for local_idx, frame in enumerate(c.decode(video=0)):
                    count+=1
                    if local_idx%10:
                        decoded_index+=1
                        continue
                    image=frame.to_ndarray(format="rgb24")
                    mask=invalid_mask(image.shape,profile,margin=3)
                    if agent=="agent_000":
                        mask[:int(image.shape[0]*.30),:]=255  # inspected developer-console overlay
                    clean=clean_reference(image,mask)
                    inputs=processor(images=Image.fromarray(clean),text="person. car.",return_tensors="pt").to("cuda:0")
                    with torch.inference_mode():
                        outputs=model(**inputs)
                    detection=processor.post_process_grounded_object_detection(
                        outputs, inputs.input_ids,threshold=.3,text_threshold=.25,
                        target_sizes=[image.shape[:2]])[0]
                    detections=[]
                    all_boxes=detection["boxes"].cpu().tolist()
                    labels=detection["text_labels"]
                    for box,confidence,label in zip(all_boxes,detection["scores"].cpu().tolist(),labels):
                        if label.strip().rstrip(".")!="person":continue
                        if any(other.strip().rstrip(".")!="person" and iou(box,b)>.8
                               for b,other in zip(all_boxes,labels)):continue
                        if overlap_fraction(box,mask)>.05: continue
                        detections.append({"bbox":box,"confidence":confidence})
                    # Decode every preceding segment; never guess offset from segment number.
                    time_global=decoded_index/fps
                    entry={"segment":path.name,"local_frame":local_idx,"decoded_global_index":decoded_index,
                           "decoded_pts":float(frame.time),"time":origin+time_global,
                           "global_video_sec":time_global,"width":frame.width,"height":frame.height,
                           "detections":detections}
                    frames.append(entry)
                    if detections:
                        name=f"{agent}_{segment:03d}_{local_idx:03d}.jpg"
                        Image.fromarray(image).save(output/name)
                        entry["image"]=name
                    decoded_index+=1
                segment_counts.append({"name":path.name,"frames":count})
        if len(pts)<decoded_index:
            raise ValueError("Not enough source timestamps for decoded prefix")
        delta=pts[:decoded_index]-np.arange(decoded_index)/20
        if np.max(abs(delta))>.05:
            raise ValueError("Decoded prefix and source timestamp timing disagree")
        results["agents"][agent]={"video_server_origin_sec":origin,"cameras":client_records,"frames":frames,
            "frame_mapping":{"method":"CFR cumulative decoded index checked against raw showinfo PTS",
                             "source_pts_vs_cfr_max_error_sec":float(np.max(abs(delta))),
                             "segments":segment_counts,"fps":20,
                             "scope":"verified first10 contiguous segments only; reject extrapolation"}}
        print(agent,len(frames),"frames",sum(bool(f["detections"]) for f in frames),"with person",flush=True)
    (output/"projection_evidence.json").write_text(json.dumps(results,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path);p.add_argument("output",type=Path);p.add_argument("profile",type=Path)
    a=p.parse_args();main(a.root,a.output,a.profile)
