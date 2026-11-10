"""Audit clocks and decoded frame correspondence before fitting any projection."""
import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw
import pyarrow.parquet as pq
from scipy.optimize import least_squares


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def robust_clock(sync):
    valid = [r for r in sync if r.get("round_trip_ms") is not None]
    threshold = float(np.percentile([r["round_trip_ms"] for r in valid], 25))
    selected = [r for r in valid if r["round_trip_ms"] <= threshold]
    x = np.array([r["sidecar_monotonic_ns"] for r in selected], float) / 1e9
    y = np.array([r["server_monotonic_ns"] for r in selected], float) / 1e9
    x0, y0 = x[0], y[0]
    train = np.arange(len(x)) % 4 != 0
    fit = least_squares(lambda p: (p[0]*(x[train]-x0)+p[1]-(y[train]-y0))*1000,
                        [1, 0], loss="soft_l1", f_scale=10)
    residual = (fit.x[0]*(x-x0)+fit.x[1]-(y-y0))*1000
    return {"local_origin_sec": x0, "server_origin_sec": y0, "alpha": float(fit.x[0]),
            "offset_sec": float(fit.x[1]), "rtt_cutoff_ms": threshold,
            "fit_pairs": int(train.sum()), "holdout_pairs": int((~train).sum()),
            "holdout_abs_error_p50_ms": float(np.median(abs(residual[~train]))),
            "holdout_abs_error_p95_ms": float(np.percentile(abs(residual[~train]), 95)),
            "drift_ppm": float((fit.x[0]-1)*1e6),
            "scope": "sidecar receipt clock to server, not actual exposure timestamp"}


def clock_time(clock, local_ns):
    return clock["server_origin_sec"] + clock["offset_sec"] + clock["alpha"] * (
        np.asarray(local_ns) / 1e9 - clock["local_origin_sec"])


def main(root, output):
    output.mkdir(parents=True, exist_ok=True)
    world = [json.loads(r) for r in pq.read_table(root/"server/world_state.parquet")["payload_json"].to_pylist()]
    wt = np.array([r["server_monotonic_ns"] for r in world])/1e9
    report = {"world_start_sec": float(wt[0]), "agents": {}}
    for agent in ("agent_000", "agent_001"):
        folder = root/"agents"/agent
        raw = rows(folder/"frame_timestamps.jsonl")
        clients = rows(root/".spool/agents"/agent/"client.jsonl")
        clock = robust_clock(rows(folder/"clock_sync.jsonl"))
        rt = clock_time(clock, [r["sidecar_monotonic_ns"] for r in raw])
        pts = np.array([r["video_pts_ns"] for r in raw])/1e9
        frame_ids = np.array([r["frame_index"] for r in raw])
        receipt = np.array([r["sidecar_monotonic_ns"] for r in raw])/1e9
        delays = receipt-receipt[0] - (pts-pts[0])
        camera = [{"seq": r["sample_seq_estimate"], **r["camera"]} for r in clients]
        decoded = []
        samples = []
        cumulative = 0
        for path in sorted(folder.glob("video_*.mkv")):
            with av.open(str(path)) as c:
                stream = c.streams.video[0]
                count = 0
                for local_index, frame in enumerate(c.decode(stream)):
                    if local_index % 10 == 0:
                        image = frame.to_ndarray(format="rgb24")
                        thumb = Image.fromarray(image).resize((384, 216))
                        # Index mapping is a CANDIDATE, subject to count/PTS validation.
                        global_index = cumulative+local_index
                        i = np.searchsorted(frame_ids, global_index)
                        record = None if i == len(frame_ids) or frame_ids[i] != global_index else raw[i]
                        if record:
                            t = float(rt[i])
                            widx = int(np.argmin(abs(wt-t)))
                            nearest = min(camera, key=lambda r: abs(r["seq"]-world[widx]["sample_seq"]))
                            sample = {"segment": path.name, "local_frame": local_index,
                                      "candidate_global_frame": global_index, "local_pts": float(frame.time),
                                      "raw_pts": float(pts[i]), "server_sec": t,
                                      "world_seq": world[widx]["sample_seq"], "camera": nearest}
                            samples.append(sample)
                            if local_index == 40:
                                draw = ImageDraw.Draw(thumb)
                                draw.rectangle((0,0,384,28), fill="black")
                                draw.text((3,3), f"{agent} {path.stem} world={sample['world_seq']}", fill="white")
                                thumb.save(output/f"{agent}_{path.stem}.jpg")
                        decoded.append({"segment":path.name, "local_index":local_index,
                                        "global_index_candidate":global_index,"pts":float(frame.time)})
                    count += 1
                cumulative += count
        # Detect reset/nonmonotonic raw timelines instead of fitting across restarts.
        report["agents"][agent] = {
            "clock": clock, "raw_rows": len(raw), "decoded_prefix_frames": cumulative,
            "raw_frame_id_gaps": int(np.sum(np.diff(frame_ids)!=1)),
            "raw_pts_nonmonotonic": int(np.sum(np.diff(pts)<=0)),
            "receipt_pts_drift_span_ms": float(np.ptp(delays)*1000),
            "receipt_pts_first40s_p5_p95_ms": np.percentile(delays[pts<40]*1000,[5,95]).tolist(),
            "first_raw_frame_server_sec": float(rt[0]), "samples":samples, "decoded_index_samples": decoded,
        }
    (output/"audit.json").write_text(json.dumps(report, indent=2))
    for agent, r in report["agents"].items():
        print(agent, json.dumps({k:v for k,v in r.items() if k not in ("samples","decoded_index_samples")},indent=2))
    images = sorted(output.glob("agent_*_video_*.jpg"))
    canvas = Image.new("RGB", (384*5,216*((len(images)+4)//5)))
    for i,p in enumerate(images):
        canvas.paste(Image.open(p),(i%5*384,i//5*216))
    canvas.save(output/"contact.jpg")


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    main(a.root,a.output)
