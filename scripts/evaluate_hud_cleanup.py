"""Measure retained HUD edge evidence and compare valid-region structure."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_comfy_control import decode


def main(clean,previous):
    mask=np.asarray(Image.open(clean/"invalid_pixels.png"))>0
    source,_=decode(clean/"source_resampled.mp4")
    condition,_=decode(clean/"control_canny.mp4")
    output,_=decode(clean/"controlled.mp4")
    before,_=decode(previous/"controlled.mp4")
    reference=np.asarray(Image.open(clean/"reference.png"))
    valid=~(cv2.dilate(mask.astype("uint8"),np.ones((7,7),np.uint8))>0)
    result={"frames":len(output),"excluded_fraction":float(mask.mean()),
            "decoded_control_nonzero_pixels_inside_mask":int(np.count_nonzero(condition[:,mask]>16)),
            "raw_control_nonzero_inside_mask":json.loads((clean/"metadata.json").read_text())["hud"]["control_nonzero_inside_invalid"],
            "scope":"HUD-edge retention is not OCR proof of absence; manually inspect contact sheet",
            "reference_changed_outside_mask_max":int(np.max(np.abs(reference[~mask].astype(int)-np.asarray(Image.open(previous/"reference.png"))[~mask].astype(int))))}
    for label,clip in (("before",before),("after",output)):
        hud_retained=[];valid_distance=[]
        for src,img in zip(source,clip):
            truth=cv2.Canny(src,100,200)>0;edges=cv2.Canny(img,100,200)>0
            dt=cv2.distanceTransform((~edges).astype("uint8"),cv2.DIST_L2,3)
            region=truth&mask
            if region.any():hud_retained.append(float((dt[region]<=2).mean()))
            a=truth&valid;b=edges&valid
            if a.any() and b.any():
                ta=cv2.distanceTransform((~a).astype("uint8"),cv2.DIST_L2,3)
                tb=cv2.distanceTransform((~b).astype("uint8"),cv2.DIST_L2,3)
                valid_distance.append(float((ta[b].mean()+tb[a].mean())/2))
        result[label]={"source_HUD_edge_retention_within2px":float(np.mean(hud_retained)),
                       "valid_region_symmetric_edge_distance_px":float(np.mean(valid_distance))}
    canvas=Image.new("RGB",(416*3,264*3))
    draw=ImageDraw.Draw(canvas)
    for row,(label,clip) in enumerate((("source",source),("previous",before),("HUD cleaned",output))):
        for col,index in enumerate((0,len(output)//2,len(output)-1)):
            canvas.paste(Image.fromarray(clip[index]).resize((416,240)),(col*416,row*264+24))
            draw.text((col*416+4,row*264+5),f"{label} frame {index}",fill="white")
    canvas.save(clean/"hud_comparison.jpg")
    (clean/"hud_evaluation.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("clean",type=Path);p.add_argument("previous",type=Path)
    a=p.parse_args();main(a.clean,a.previous)
