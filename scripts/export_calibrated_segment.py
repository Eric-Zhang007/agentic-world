"""Verify a decoded segment against per-frame calibrated, unmodified world positions."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import av
import yaml
from PIL import ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from compiler.episode_alignment import EpisodeAlignment
from compiler.projection import project_object
from perception.hud import load_profile,invalid_mask
from matching.matcher import match
from state.models import ObservationObject


def main(root, calibration, output, agent, segment, hud):
    output.mkdir(parents=True,exist_ok=True)
    alignment=EpisodeAlignment(calibration/"projection_evidence.json",calibration/"projection_calibration.json",agent)
    evidence=json.loads((calibration/"projection_evidence.json").read_text())
    observed={(r["segment"],r["local_frame"]):r for r in evidence["agents"][agent]["frames"]}
    matching_config=yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/default.yaml").read_text())["matching"]
    profile=load_profile(hud)
    timeline=[]
    with av.open(str(root/"agents"/agent/segment)) as c:
        for index,frame in enumerate(c.decode(video=0)):
            objects,camera,meta=alignment.at_frame(segment,index,frame.width,frame.height)
            camera["invalid_regions_normalized"]=profile["rectangles_normalized"]
            row={**meta,"camera":camera,"objects":[asdict(o) for o in objects],
                 "projections":{o.object_id:project_object(o,camera) for o in objects if o.state["matchable"]}}
            if (segment,index) in observed:
                measurements=[ObservationObject(
                    obs_id=f"{agent}:{segment}:{index}:{n}",track_id=None,category="person",
                    bbox_xyxy=d["bbox"],confidence=d["confidence"],frame_id=index,timestamp=meta["server_time_sec"],
                    source="ORIGINAL:GroundingDINO") for n,d in enumerate(observed[segment,index]["detections"])]
                row["matching"]=match(objects,measurements,camera,matching_config).to_dict()
            timeline.append(row)
            if index in (0,40,79):
                image=frame.to_image()
                mask=invalid_mask((frame.height,frame.width),profile,margin=3)
                draw=ImageDraw.Draw(image)
                for obj in objects:
                    if not obj.state["matchable"]:continue
                    proj=project_object(obj,camera)
                    if proj and proj["visible"]:
                        x,y=proj["center"]
                        if mask[int(y),int(x)]:
                            continue
                        draw.ellipse((x-6,y-6,x+6,y+6),outline="cyan",width=3)
                        draw.text((x+8,y),obj.object_id.split(":")[-1],fill="cyan")
                draw.text((10,50),f"{agent} frame={index} server={meta['server_time_sec']:.3f}",fill="white")
                image.save(output/f"projection_{index:03d}.png")
    (output/"timeline.json").write_text(json.dumps(timeline,indent=2))
    print(json.dumps({"frames":len(timeline),"world_objects_per_frame":sorted(set(len(r["objects"]) for r in timeline)),
                      "first_server_time":timeline[0]["server_time_sec"],"last_server_time":timeline[-1]["server_time_sec"]}))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path);p.add_argument("calibration",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--agent",default="agent_001");p.add_argument("--segment",default="video_00002.mkv")
    p.add_argument("--hud",type=Path,required=True)
    a=p.parse_args();main(a.root,a.calibration,a.output,a.agent,a.segment,a.hud)
