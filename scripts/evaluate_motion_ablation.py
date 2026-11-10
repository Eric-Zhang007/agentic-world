"""Read-only perception and blinded visual review of the temporal experiment."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import av
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.action_planner import json_call
from perception.detector import PerceptionPipeline
from matching.costs import iou


def project(points, camera, width, height):
    eye = np.asarray(camera["position"], dtype=float)
    forward = np.asarray(camera["target"], dtype=float) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0,1,0]); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    delta = np.asarray(points, dtype=float) - eye
    depth = delta @ forward
    if (depth <= 0).any():
        return None
    fy = .5 / math.tan(math.radians(camera.get("fov",50))/2)
    # Original Godot viewport is 960x640; experiment stretches those pixels.
    return np.c_[width * (.5 + (delta @ right)/depth * fy/1.5),
                 height * (.5 - (delta @ up)/depth * fy)]


def expected_actors(frame, width, height):
    result = []
    for entity in frame["entities"]:
        if entity["id"] not in ("player", "tiger_guard"):
            continue
        pose = entity["pose"]
        points = np.asarray(list(pose.values()))
        h = entity["size"][1]
        # Account for rendered head sphere and sole beyond joint centers.
        points = np.concatenate([points, [np.asarray(pose["head"])+[0,.085*h,0]],
                                 [np.asarray(pose["left_foot"])-[0,.044*h,0]],
                                 [np.asarray(pose["right_foot"])-[0,.044*h,0]]])
        pixels = project(points, frame["camera"], width, height)
        if pixels is None:
            continue
        bbox = [*pixels.min(axis=0), *pixels.max(axis=0)]
        result.append({"world_id": entity["id"], "bbox": list(map(float,bbox))})
    return result


def assign(expected, observations, width, height):
    actors = []
    for observation in sorted(observations, key=lambda o: o.confidence, reverse=True):
        if "staff" not in (observation.category or "") and not any(
            iou(observation.bbox_xyxy, kept.bbox_xyxy) > .8 for kept in actors
        ):
            actors.append(observation)
    if not actors:
        return [], [e["world_id"] for e in expected]
    costs = np.ones((len(expected),len(actors)))
    for i,e in enumerate(expected):
        a = np.asarray(e["bbox"])
        for j,o in enumerate(actors):
            b = np.asarray(o.bbox_xyxy)
            error = np.linalg.norm((a[:2]+a[2:])/2-(b[:2]+b[2:])/2)
            label = o.category or ""
            semantic = .5 if (e["world_id"]=="player" and "tiger" in label) or (e["world_id"]=="tiger_guard" and "monkey" in label) else 0
            costs[i,j] = error/math.hypot(width,height)+(1-iou(a,b))*.2+semantic
    rows,cols = linear_sum_assignment(np.c_[costs,np.full((len(expected),len(expected)),.55)])
    pairs,missing = [],[]
    for i,j in zip(rows,cols):
        if j>=len(actors) or costs[i,j]>=.55:
            missing.append(expected[i]["world_id"]); continue
        e,o = expected[i],actors[j]
        a,b=np.asarray(e["bbox"]),np.asarray(o.bbox_xyxy)
        error=float(np.linalg.norm((a[:2]+a[2:])/2-(b[:2]+b[2:])/2))
        pairs.append({"world_id":e["world_id"],"obs_id":o.obs_id,"track_id":o.track_id,
                      "category":o.category,"confidence":o.confidence,
                      "expected_bbox":e["bbox"],"observed_bbox":o.bbox_xyxy,
                      "center_error_px":error,"center_error_body_heights":error/max(1,a[3]-a[1]),
                      "bbox_iou":float(iou(a,b))})
    return pairs,missing


def evaluate(root,label,device):
    folder = root / "evaluation" / label
    folder.mkdir(parents=True,exist_ok=True)
    manifest = json.loads((root/"manifest.json").read_text())
    cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/default.yaml").read_text())
    video=root/"render"/f"{label}.mp4"
    with av.open(str(video)) as stream:
        frames=[frame.to_ndarray(format="rgb24") for frame in stream.decode(video=0)]
    if len(frames)!=manifest["frames"]:
        raise ValueError("Generated duration differs from expected control timeline")
    keys=manifest["keyframes"]
    review_indices=sorted(set(np.rint(np.linspace(0,len(frames)-1,25)).astype(int).tolist()+keys))
    for j in review_indices:
        Image.fromarray(frames[j]).save(folder/f"frame_{j:03d}.png")
    detector=PerceptionPipeline({**cfg["perception"],"device":device,
                                "categories":["monkey warrior","tiger warrior","staff"]})
    rows=[]
    for j in keys:
        path=folder/f"perception_{j:03d}.json"
        expected=expected_actors(manifest["expected"][j],manifest["width"],manifest["height"])
        if path.exists():
            row=json.loads(path.read_text())
        else:
            frame={"frame_id":j,"timestamp":j/24,"path":str(folder/f"frame_{j:03d}.png")}
            observations=detector.detect_frame(frame,folder/f"perception_{j:03d}",label)
            pairs,missing=assign(expected,observations,manifest["width"],manifest["height"])
            row={"frame":j,"matches":pairs,"missing":missing,
                 "observations":[o.to_dict() for o in observations]}
            path.write_text(json.dumps(row,indent=2))
        rows.append(row)
        image=Image.fromarray(frames[j]);draw=ImageDraw.Draw(image)
        for e in expected:
            draw.rectangle(e["bbox"],outline="cyan",width=2)
            draw.text(tuple(e["bbox"][:2]),"expected "+e["world_id"],fill="cyan")
        for pair in row["matches"]:
            draw.rectangle(pair["observed_bbox"],outline="yellow",width=2)
        image.save(folder/f"overlay_{j:03d}.png")
    metrics={"label":label,"frames":rows,"world_mutation":False,
             "limits":"GroundingDINO/SAM2/DINOv2 detections; deterministic semantic/projection assignment. Inferred proxy boxes are approximate. Not metric 3D, pose or contact ground truth."}
    drift=[]
    for first in rows[0]["matches"]:
        last=next((m for m in rows[-1]["matches"] if m["world_id"]==first["world_id"]),None)
        if last:
            center=lambda b:(np.asarray(b[:2])+np.asarray(b[2:]))/2
            observed_delta=center(last["observed_bbox"])-center(first["observed_bbox"])
            expected_delta=center(last["expected_bbox"])-center(first["expected_bbox"])
            drift.append({"world_id":first["world_id"],"expected_delta_px":expected_delta.tolist(),
                          "observed_delta_px":observed_delta.tolist(),
                          "displacement_error_px":float(np.linalg.norm(observed_delta-expected_delta)),
                          "identity_note":"role assignment; cross-frame identity not proven by projection"})
    metrics["endpoint_displacements"]=drift
    (folder/"metrics.json").write_text(json.dumps(metrics,indent=2))
    sheet=Image.new("RGB",(4*384,2*310),"white")
    draw=ImageDraw.Draw(sheet)
    for col,j in enumerate(keys):
        for row,path in enumerate((root/"expected"/f"{j:03d}.png",folder/f"overlay_{j:03d}.png")):
            im=Image.open(path);im.thumbnail((384,288))
            sheet.paste(im,(col*384,row*310+20))
            draw.text((col*384+4,row*310+3),f"{'proxy' if row==0 else 'generated'} t={j/24:.2f}s",fill="black")
    sheet.save(folder/"comparison.jpg")
    review_path=folder/"visual_review.json"
    if review_path.exists() and json.loads(review_path.read_text()).get("contract_version")!=2:
        archive=folder/"strict_proxy_review.json"
        if archive.exists():raise FileExistsError("Previous review archive already exists")
        review_path.rename(archive)
    if not review_path.exists():
        review=json_call(cfg["agent"],
            "Review a generated action against an explicit coarse world, not against the prompt alone. "
            "You are blinded to the control condition. Return JSON {decision:ACCEPT|REGENERATE|UNCERTAIN, "
            "single_horizontal_sweep:boolean|null, start_state_consistent:boolean|null, "
            "end_state_consistent:boolean|null, no_extra_causal_events:boolean|null, "
            "motion_richness_1_to_5:number, geometry_fidelity_1_to_5:number, "
            "appearance_fidelity_1_to_5:number, evidence:[string], violations:[string], uncertainties:[string]}. "
            "First image is appearance reference, second is paired expected/generated keyframes. "
            "Remaining images are chronological generated samples. Look for actual sideways step, "
            "ONE readable horizontal staff sweep and recovery; weapon twirling or raising alone is not a sweep. "
            "Allow natural secondary joint/cloth/weight motion. Reject extra strikes, teleportation, unplanned knockdown, "
            "scene replacement or changed endpoint placement. Coarse proxy shape/texture should NOT replace detailed identities. "
            "Do not infer damage from near contact. Separate unknown occluded events from observed errors. "
            "A matching first frame alone is not action success. All required state/event fields must be true to ACCEPT. "
            "IMPORTANT boundary-state contract: the proxy action criteria describe a coarse executable path, "
            "not a demand to copy every intermediate joint pose. Natural torso/hip turning, stance variation, "
            "wrist motion and temporary grip adjustment BETWEEN critical events are allowed; do not reject "
            "those merely because the rigid proxy lacks them. Judge final facing/placement/held weapon, "
            "one actual horizontal attack, and absence of extra causal events. A stationary horizontal hold "
            "or vertical flourish alone is not that attack. Both-hands pose is relevant to the attack only, "
            "not a blanket requirement throughout preparation and recovery. Do not require an offscreen patrol "
            "to appear in this video. Known input mismatch: original appearance-frame staff pose and inferred "
            "proxy initial grip differ. Separate this input ambiguity from observed generated-video violations. "
            "Numerical body-box metrics are approximate and cannot establish pose/contact correctness. "
            "You see sampled frames, not every frame; acknowledge this limit.",
            {"action":manifest["action"],"keyframe_times":[j/24 for j in keys],
             "sample_times":[j/24 for j in review_indices],"metrics":{
                 "endpoint_displacements":drift,"limits":metrics["limits"]}},
            [manifest["reference"],folder/"comparison.jpg"]+[folder/f"frame_{j:03d}.png" for j in review_indices])
        if review.get("decision")=="ACCEPT" and any(review.get(k) is not True for k in
            ("single_horizontal_sweep","start_state_consistent","end_state_consistent","no_extra_causal_events")):
            review["raw_decision"]="ACCEPT";review["decision"]="UNCERTAIN"
        review["contract_version"]=2
        review_path.write_text(json.dumps(review,indent=2))
    print(json.dumps({"label":label,"review":json.loads(review_path.read_text()),"drift":drift}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path)
    selection=p.add_mutually_exclusive_group(required=True)
    selection.add_argument("--label");selection.add_argument("--all",action="store_true")
    p.add_argument("--device",default="cuda:0")
    a=p.parse_args();root=a.root.resolve()
    labels=[case["label"] for case in json.loads((root/"cases.json").read_text())] if a.all else [a.label]
    for label in labels:
        deadline=time.monotonic()+1200
        while True:
            try:
                ready={run["label"] for run in json.loads((root/"render/metadata.json").read_text())["runs"]}
            except (FileNotFoundError,json.JSONDecodeError):
                ready=set()
            if label in ready:break
            if time.monotonic()>deadline:raise TimeoutError(f"No completed generation for {label}")
            time.sleep(5)
        evaluate(root,label,a.device)
