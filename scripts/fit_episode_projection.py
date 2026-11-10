"""Fit a small residual correction and validate on disjoint temporal blocks."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image,ImageDraw
from scipy.optimize import least_squares

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from compiler.calibration import camera_matrix


def fit_agent(data, agent, root):
    world=data["world"];wt=np.array([r["server_monotonic_ns"]/1e9 for r in world])
    stream=data["agents"][agent];cameras=stream["cameras"]
    ct=np.array([r["time"] for r in cameras])
    angles=np.degrees(np.unwrap(np.radians([r["camera"]["camera_rotation"] for r in cameras]),axis=0))
    positions=np.array([r["camera"]["camera_position"] for r in cameras])
    proposals=[r for r in stream["frames"] if len(r["detections"])==1 and wt[0]+2<r["time"]<min(wt[-1],ct[-1])-2]
    samples=[]
    excluded=[]
    for r in proposals:
        x0,y0,x1,y1=r["detections"][0]["bbox"]
        # Tiny or clipped boxes are poor calibration landmarks (including cursor false positives).
        if y1-y0<40 or x1-x0<12 or min(x0,y0)<2 or x1>r["width"]-2 or y1>r["height"]-2:
            excluded.append({"image":r.get("image"),"reason":"landmark_too_small_or_clipped"})
        else:
            samples.append(r)
    # Whole 4s recording segments held out, not neighboring frames interleaved.
    train=np.array([int(Path(r["segment"]).stem.split("_")[-1])%3!=2 for r in samples])

    def project(sample, params, axis):
        shift,scale,pitch,yaw,z=params
        t=sample["time"]+shift
        j=int(np.searchsorted(wt,t))
        if j<1 or j>=len(world):return None
        a,b=world[j-1],world[j]
        entities=[e for e in a["agents"] if e["agent_id"]!=agent]
        if len(entities)!=1:return None
        e=entities[0];other=next((v for v in b["agents"] if v["dataset_entity_id"]==e["dataset_entity_id"]),None)
        if other is None:return None
        alpha=(t-wt[j-1])/(wt[j]-wt[j-1])
        xyz=(1-alpha)*np.array(e["position"])+alpha*np.array(other["position"])
        xyz[2]+=z
        if t<ct[0] or t>ct[-1]:return None
        record=dict(cameras[int(np.argmin(abs(ct-t)))]["camera"])
        record["camera_position"]=[float(np.interp(t,ct,positions[:,i])) for i in range(3)]
        record["camera_rotation"]=[float(np.interp(t,ct,angles[:,i])) for i in range(3)]
        k,transform=camera_matrix(record,sample["width"],sample["height"],
                                 {"focal_scale":scale,"pitch_bias_deg":pitch,"yaw_bias_deg":yaw},axis)
        p=transform@np.r_[xyz,1]
        if p[2]<.15:return None
        uv=k@p[:3]
        return uv[:2]/uv[2],e["dataset_entity_id"],record

    usable=[i for i,s in enumerate(samples) if project(s,[0,1,0,0,0],"vertical") is not None]
    samples=[samples[i] for i in usable];train=train[usable]
    if len(samples)<12 or (~train).sum()<4:
        return {"status":"insufficient_evidence","usable_samples":len(samples),"heldout":int((~train).sum()),
                "excluded_landmarks":excluded,"parameters":None}
    centers=np.array([[(s["detections"][0]["bbox"][0]+s["detections"][0]["bbox"][2])/2,
                       (s["detections"][0]["bbox"][1]+s["detections"][0]["bbox"][3])/2] for s in samples])
    def residual(params,axis,subset):
        predicted=[]
        for s,truth,use in zip(samples,centers,subset):
            if not use:continue
            p=project(s,params,axis)
            predicted.extend([1000,1000] if p is None else p[0]-truth)
        return np.array(predicted)
    trials=[]
    for axis in ("vertical","horizontal"):
        for shift in (-1,0,1):
            fit=least_squares(residual,[shift,1,0,0,0],args=(axis,train),
                              bounds=([-2,.65,-5,-5,-.4],[2,1.5,5,5,.4]),
                              loss="soft_l1",f_scale=10,max_nfev=180)
            train_error=np.linalg.norm(residual(fit.x,axis,train).reshape(-1,2),axis=1)
            trials.append((float(np.median(train_error)),axis,fit))
    _,axis,fit=min(trials,key=lambda x:x[0])  # Never choose a model by held-out score.
    initial=np.linalg.norm(residual([0,1,0,0,0],axis,~train).reshape(-1,2),axis=1)
    error=np.linalg.norm(residual(fit.x,axis,~train).reshape(-1,2),axis=1)
    limited=bool(np.any(np.isclose(fit.x,[-2,.65,-5,-5,-.4],atol=1e-3)) or
                 np.any(np.isclose(fit.x,[2,1.5,5,5,.4],atol=1e-3)))
    passed=bool(np.median(error)<20 and np.percentile(error,95)<50 and not limited)
    report={"calibration_id":f"ep_20260812_090725_f8a5:{agent}:first40s",
            "status":"validated" if passed else "rejected",
            "scope":"other-player entity origin vs detected person bbox center; not calibrated bbox extent or metric depth",
            "parameters":dict(zip(["video_to_telemetry_offset_sec","focal_scale","pitch_bias_deg","yaw_bias_deg","entity_center_z_offset"],fit.x.tolist())),
            "fov_axis":axis,"fit_count":int(train.sum()),"holdout_count":int((~train).sum()),
            "holdout_raw_median_px":float(np.median(initial)),"holdout_median_px":float(np.median(error)),
            "holdout_p95_px":float(np.percentile(error,95)),"parameter_at_bound":limited,
            "valid_server_time_range":[min(s["time"] for s in samples)+fit.x[0],max(s["time"] for s in samples)+fit.x[0]],
            "candidate_fits":[{"train_median_px":score,"axis":ax,"parameters":f.x.tolist()} for score,ax,f in trials],
            "limitations":["detector bbox centers are noisy proxy landmarks, not manually surveyed 3D correspondences",
                           "FOV axis and free focal scale can be confounded; fitted effective K is the result",
                           "constant delay is only tested in first40s; no whole-episode extrapolation",
                           "entity identity assumed to be the single other player; no arbitrary world insertions"],
            "samples":[]}
    report["excluded_landmarks"]=excluded
    profile=[]
    for shift in np.linspace(fit.x[0]-.3,fit.x[0]+.3,25):
        trial=least_squares(lambda q:residual(np.r_[shift,q],axis,train),fit.x[1:],
                            bounds=([.65,-5,-5,-.4],[1.5,5,5,.4]),
                            loss="soft_l1",f_scale=10,max_nfev=70)
        e=np.linalg.norm(residual(np.r_[shift,trial.x],axis,train).reshape(-1,2),axis=1)
        profile.append({"offset_sec":float(shift),"train_median_px":float(np.median(e))})
    best=min(r["train_median_px"] for r in profile)
    plausible=[r["offset_sec"] for r in profile if r["train_median_px"]<=best+1]
    report["time_offset_profile"]=profile
    report["time_offset_near_optimal_range_sec"]=[min(plausible),max(plausible)]
    report["time_offset_range_definition"]="nuisance-refitted training median within1px of best; NOT a statistical confidence interval"
    for i,(s,truth) in enumerate(zip(samples,centers)):
        before=project(s,[0,1,0,0,0],axis);after=project(s,fit.x,axis)
        if after is None:continue
        detail={"image":s.get("image"),"time":s["time"],"holdout":bool(not train[i]),
                "world_id":str(after[1]),"observed":truth.tolist(),"projected":after[0].tolist(),
                "error_px":float(np.linalg.norm(after[0]-truth))}
        report["samples"].append(detail)
        if not train[i] and s.get("image"):
            image=Image.open(root/s["image"]);draw=ImageDraw.Draw(image)
            draw.rectangle(s["detections"][0]["bbox"],outline="lime",width=3)
            for point,color in ((before[0] if before else None,"red"),(after[0],"cyan")):
                if point is None:continue
                x,y=point
                if not (-10000<x<10000 and -10000<y<10000):continue
                draw.line((x-8,y,x+8,y),fill=color,width=3);draw.line((x,y-8,x,y+8),fill=color,width=3)
            draw.text((15,50),f"{agent} HELDOUT {report['status']} err={detail['error_px']:.1f}px",fill="white")
            image.save(root/("projection_"+s["image"]))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);a=p.parse_args()
    data=json.loads((a.root/"projection_evidence.json").read_text())
    result={agent:fit_agent(data,agent,a.root) for agent in data["agents"]}
    (a.root/"projection_calibration.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:{x:y for x,y in v.items() if x not in ("samples","candidate_fits")} for k,v in result.items()},indent=2))
