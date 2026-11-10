"""Fit a read-only experimental camera to appearance-image actor boxes."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_motion_ablation import expected_actors


def main(attempt, metrics_path, output):
    scene=json.loads((attempt/"scene.json").read_text())
    timeline=json.loads((attempt/"timeline.json").read_text())
    matches=json.loads(metrics_path.read_text())["frames"][0]["matches"]
    targets={m["world_id"]:np.asarray(m["observed_bbox"]) for m in matches}
    if set(targets)!={"player","tiger_guard"}:raise ValueError("Both source-image actors need observed boxes")
    original=copy.deepcopy(scene["camera"])
    def camera(x):return {"position":x[:3].tolist(),"target":x[3:].tolist(),"fov":50}
    def residual(x):
        candidate={**scene,"camera":camera(x)}
        expected=expected_actors(candidate,1024,768)
        if len(expected)!=2:return np.ones(6)*1e4
        result=[]
        for e in expected:
            box=np.asarray(e["bbox"]);target=targets[e["world_id"]]
            result.extend([((box[0]+box[2])-(target[0]+target[2]))/2,
                           box[1]-target[1],box[3]-target[3]])
        return np.asarray(result)
    candidates=[]
    for eye in ([-4,2,-1.5],[-3,1.8,-2.5],[-5,2,-.8]):
        fitted=least_squares(residual,np.asarray(eye+[-.4,.8,.5]),
                             bounds=([-15,.5,-15,-5,-1,-5],[15,8,15,5,5,5]),max_nfev=1000)
        candidates.append(fitted)
    fit=min(candidates,key=lambda r:np.linalg.norm(r.fun))
    if np.max(abs(fit.fun))>35:raise ValueError(f"Camera cannot align starting actor boxes: {fit.fun}")
    scene["camera"]=camera(fit.x)
    for frame in timeline["frames"]:frame["camera"]=scene["camera"]
    output.mkdir(parents=True,exist_ok=False)
    for name,data in (("scene.json",scene),("timeline.json",timeline),
                      ("action.json",json.loads((attempt/"action.json").read_text()))):
        (output/name).write_text(json.dumps(data,indent=2))
    report={"source_attempt":str(attempt),"target_detection_source":str(metrics_path),
            "original_camera":original,"experimental_camera":scene["camera"],
            "fit_residual_px":fit.fun.tolist(),"target_boxes":{k:v.tolist() for k,v in targets.items()},
            "world_mutation":False,"scope":"Inferred camera only; actor geometry, action, poses and event times unchanged. Targets from first guided pilot frame, not metric reconstruction. Only center-x and top/bottom fitted; widths excluded because costume changes silhouette."}
    (output/"calibration.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--attempt",type=Path,required=True)
    p.add_argument("--metrics",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();main(a.attempt,a.metrics,a.output)
