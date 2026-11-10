"""Real perception and projection metrics for world-control ablations."""
import argparse
import json
from pathlib import Path
import sys

import av
import numpy as np
import yaml
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from compiler.world_controls import coarse_object
from matching.matcher import match
from perception.detector import PerceptionPipeline
from perception.frame_sampler import sample_frames
from perception.visualize import draw_frame
from state.models import WorldObject,ObservationObject
from verifier.discrepancy import discrepancy


def main(a):
    root=a.root;render=a.render or root/"render"
    metadata=json.loads((render/"metadata.json").read_text())
    manifest=json.loads((root/"controls/manifest.json").read_text())
    config=yaml.safe_load(Path("configs/default.yaml").read_text())
    results=[];panels=[]
    for run in metadata["runs"]:
        label=run["label"];folder=render/"evaluation"/label;folder.mkdir(parents=True,exist_ok=True)
        frames=sample_frames(render/f"{label}.mp4",folder/"frames",5.0)
        detector=PerceptionPipeline({**config["perception"],"device":a.device,"categories":["person","car","dog","tree"]})
        rows=[];track={}
        for frame in frames:
            path=folder/f"observations_{frame['frame_id']}.json"
            if path.exists():observations=[ObservationObject(**o) for o in json.loads(path.read_text())]
            else:
                observations=detector.detect_frame(frame,folder/"perception"/str(frame["frame_id"]),label)
                path.write_text(json.dumps([o.to_dict() for o in observations]))
            expected=manifest["expected"][min(len(manifest["expected"])-1,round(frame["timestamp"]*24))]
            objects=[coarse_object(WorldObject(**o)) or WorldObject(**o) for o in expected["objects"]]
            result=match(objects,observations,expected["camera"],config["matching"],track).to_dict()
            rows.append({"frame_id":frame["frame_id"],"matching":result})
            overlay=folder/f"overlay_{frame['frame_id']}.png"
            draw_frame(frame,observations,result,objects,expected["camera"],overlay)
        report=discrepancy(label,rows,config["matching"])
        report["limitation"]="Partial recorded actors only; inferred sizes; no static scene geometry; three-frame sampling"
        (folder/"discrepancy.json").write_text(json.dumps(report,indent=2))
        result={"label":label,"timing":run,"summary":report["summary"]}
        results.append(result);print(json.dumps(result),flush=True)
        panel=Image.new("RGB",(960,218),"white");draw=ImageDraw.Draw(panel)
        draw.text((8,4),f"{label}: sample {run['sample_seconds']:.1f}s | missing {report['summary']['missing_ratio']:.2f} | center {report['summary']['mean_projection_error']}",fill="black")
        for i,frame in enumerate(frames[:3]):
            image=Image.open(folder/f"overlay_{frame['frame_id']}.png");image.thumbnail((320,190))
            panel.paste(image,(i*320,24))
        panels.append(panel)
    sheet=Image.new("RGB",(960,len(panels)*218),"white")
    for i,panel in enumerate(panels):sheet.paste(panel,(0,i*218))
    sheet.save(render/"comparison.jpg")
    (render/"evaluation.json").write_text(json.dumps(results,indent=2))
    cached=next((r for r in results if r["label"]=="layout_cache"),None)
    baseline=next((r for r in results if r["label"]=="layout"),None)
    if cached and baseline:
        (render/"cache_comparison.json").write_text(json.dumps({
            "sample_speedup":baseline["timing"]["sample_seconds"]/cached["timing"]["sample_seconds"],
            "sample_decode_save_speedup":baseline["timing"]["seconds"]/cached["timing"]["seconds"],
            "baseline":baseline["summary"],"cached":cached["summary"],
            "scope":"same seed, prompt, steps, weights, control; not a quality equivalence claim"},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("--render",type=Path)
    p.add_argument("--device",default="cuda:0");main(p.parse_args())
