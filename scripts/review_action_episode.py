"""Build an honest episode review: rejected attempts are not stitched as success."""
import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image,ImageDraw


def main(root):
    panels=[];records=[];accepted=[]
    key=lambda p:(int(p.name.split("_")[1]) if p.name.startswith("action_") else 0,int(p.name.rsplit("_",1)[-1]))
    for segment in sorted(root.glob("*segment_*"),key=key):
        video=segment/"render/controlled.mp4"
        if not video.exists():continue
        with av.open(str(video)) as c:
            frames=[f.to_image() for f in c.decode(video=0)]
        raw=json.loads((segment/"agent_decision.json").read_text()) if (segment/"agent_decision.json").exists() else {}
        panel=Image.new("RGB",(960,250),"white");draw=ImageDraw.Draw(panel)
        label=f"{segment.name}: {raw.get('action','PENDING')} | phase_complete={raw.get('phase_complete',False)} | action_complete={raw.get('action_complete',False)}"
        draw.text((8,5),label,fill="black")
        for j,index in enumerate((0,len(frames)//2,len(frames)-1)):
            image=frames[index].copy();image.thumbnail((320,220));panel.paste(image,(j*320,25))
        panels.append(panel)
        dense=Image.new("RGB",(960,750),"white");dd=ImageDraw.Draw(dense)
        for j,index in enumerate(np.linspace(0,len(frames)-1,9).round().astype(int)):
            image=frames[index].copy();image.thumbnail((320,220))
            x,y=(j%3)*320,(j//3)*250
            dd.text((x+5,y+4),f"frame {index} / {index/24:.2f}s",fill="black")
            dense.paste(image,(x,y+24))
        dense.save(segment/"timeline_review.jpg")
        before=json.loads((segment/"world_before.json").read_text())
        after=json.loads((segment/"world_after.json").read_text()) if (segment/"world_after.json").exists() else before
        b={o["object_id"]:o for o in before["objects"]};a={o["object_id"]:o for o in after["objects"]}
        record={"segment":segment.name,"video":str(video),"frames":len(frames),"decision":raw,
                "added_ids":sorted(set(a)-set(b)),"removed_ids":sorted(set(b)-set(a)),
                "changed_ids":[key for key in a if key in b and a[key]!=b[key]],
                "tick_before":before["tick"],"tick_after":after["tick"]}
        reference=Image.open(segment/"render/reference.png").convert("RGB")
        if reference.size==frames[0].size:
            record["first_frame_reference_mae"]=float(np.abs(np.asarray(reference,dtype=float)-np.asarray(frames[0],dtype=float)).mean())
        records.append(record)
        if raw.get("action")=="ACCEPT" and raw.get("phase_complete"):
            accepted.append(video)
    if panels:
        sheet=Image.new("RGB",(960,250*len(panels)),"white")
        for i,panel in enumerate(panels):sheet.paste(panel,(0,i*250))
        sheet.save(root/"review.jpg")
    state=json.loads((root/"action_state.json").read_text())
    result={"status":state["status"],"user_input_ready":state["user_input_ready"],"records":records,
            "generated_only_insertions":sum(len(r["added_ids"]) for r in records),
            "completed_phase_clips":sum(bool(r["decision"].get("phase_complete")) for r in records)}
    if accepted:
        target=root/("accepted_action.mp4" if state["status"]=="COMPLETE" else "accepted_prefix.mp4")
        with av.open(str(target),"w") as output:
            stream=None
            for path in accepted:
                with av.open(str(path)) as source:
                    for frame in source.decode(video=0):
                        if stream is None:
                            stream=output.add_stream("libx264",rate=24)
                            stream.width,stream.height=frame.width,frame.height;stream.pix_fmt="yuv420p"
                            stream.options={"crf":"18"}
                        fresh=av.VideoFrame.from_ndarray(frame.to_ndarray(format="rgb24"),format="rgb24")
                        for packet in stream.encode(fresh):output.mux(packet)
            for packet in stream.encode():output.mux(packet)
        result["accepted_sequence"]=str(target)
        result["sequence_scope"]="Concatenated accepted completed phases only; not guaranteed seamless or physical truth"
    (root/"review.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!="records"}))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);main(p.parse_args().root)
