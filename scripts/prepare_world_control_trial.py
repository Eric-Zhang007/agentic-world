"""Prepare same-segment calibrated world controls, never extract edges from source RGB."""
import argparse
import json
from pathlib import Path
import sys

import av
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from compiler.episode_alignment import EpisodeAlignment
from compiler.world_controls import compile_timeline
from perception.hud import load_profile,invalid_mask,clean_reference


def main(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    alignment=EpisodeAlignment(a.calibration/"projection_evidence.json",a.calibration/"projection_calibration.json","agent_001")
    timeline=[]
    for i in range(80):
        objects,camera,meta=alignment.at_frame("video_00002.mkv",i,a.width,a.height)
        timeline.append({**meta,"objects":[o.to_dict() for o in objects],"camera":camera})
    compile_timeline(timeline,out/"controls",a.frames)
    source=a.episode/"agents/agent_001/video_00002.mkv"
    with av.open(str(source)) as container:
        first=next(container.decode(video=0)).to_image().resize((a.width,a.height))
    pixels=np.array(first)
    mask=invalid_mask(pixels.shape,load_profile(a.hud),margin=3)
    Image.fromarray(clean_reference(pixels,mask)).save(out/"reference.png")
    cases=[{"label":"reference_only","controls":[],"cache_threshold":0},
           {"label":"layout","controls":[{"path":str(out/"controls/layout.npz"),"strength":.65,"end":.7}]},
           {"label":"depth","controls":[{"path":str(out/"controls/depth.npz"),"strength":.45,"end":.7}]},
           {"label":"proxy_canny","controls":[{"path":str(out/"controls/proxy_canny.npz"),"strength":.45,"end":.7}]},
           {"label":"gray","controls":[{"path":str(out/"controls/gray.npz"),"strength":.35,"end":.7}]},
           {"label":"layout_depth","controls":[{"path":str(out/"controls/layout.npz"),"strength":.4,"end":.7},
                        {"path":str(out/"controls/depth.npz"),"strength":.25,"end":.7}]},
           {"label":"layout_cache","controls":[{"path":str(out/"controls/layout.npz"),"strength":.65,"end":.7}],"cache_threshold":.2}]
    (out/"cases.json").write_text(json.dumps(cases,indent=2))
    print(json.dumps({"output":str(out),"cases":len(cases),"frames":a.frames,"source_video":str(source)}))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--episode",type=Path,required=True)
    p.add_argument("--calibration",type=Path,required=True);p.add_argument("--hud",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--frames",type=int,default=73)
    p.add_argument("--width",type=int,default=832);p.add_argument("--height",type=int,default=480)
    main(p.parse_args())
