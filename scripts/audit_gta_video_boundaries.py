"""Inspect actual decoded video boundaries; pixel differences are not 3D motion."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import av
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]/'outputs/gta_pi_loop_20261102'
REMOTE='/home/star/agentic_world/outputs/gta_pi_loop_20261102/'


def endpoints(path):
    opening=[];ending=[]
    with av.open(str(path)) as video:
        for frame in video.decode(video=0):
            image=frame.to_ndarray(format='rgb24')
            if len(opening)<2:opening.append(image)
            ending=(ending+[image])[-2:]
    return opening,ending


def mae(first,second):return float(np.abs(first.astype(float)-second.astype(float)).mean())


def audit(name):
    folder=ROOT/name
    with closing(sqlite3.connect((folder/'world.sqlite').as_uri()+'?mode=ro',uri=True)) as db:
        row=db.execute("SELECT value FROM world_meta WHERE key='whitebox_history'").fetchone()
    history=json.loads(row[0]) if row else []
    results=[];panels=[];previous=None
    for index,record in enumerate(history):
        path=ROOT/record['artifacts']['video'].removeprefix(REMOTE)
        opening,ending=endpoints(path)
        if previous is not None:
            results.append({'from_segment':index,'to_segment':index+1,
                'previous_last_step_mae_255':mae(*previous),
                'boundary_mae_255':mae(previous[-1],opening[0]),
                'next_first_step_mae_255':mae(*opening),
                'interpretation':'Whole-image RGB differences; camera, texture and all moving objects contribute. Not metric velocity.'})
            panels.append([*previous,*opening])
        previous=ending
    if panels:
        sheet=Image.new('RGB',(1280,len(panels)*208),'#202020');draw=ImageDraw.Draw(sheet)
        for row,images in enumerate(panels):
            for column,pixels in enumerate(images):
                image=Image.fromarray(pixels);image.thumbnail((320,180))
                sheet.paste(image,(column*320,row*208))
                label=['previous -1','previous last','next first','next +1'][column]
                draw.text((column*320+4,row*208+185),f'{row+1}->{row+2}: {label}',fill='white')
        sheet.save(folder/'boundary_frames.jpg',quality=95)
    (folder/'boundary_audit.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps({'case':name,'boundaries':results}))


if __name__=='__main__':
    for name in ['rural','urban','harbor','rural/continuation','rural/road_edge','urban/npc_walk']:
        if (ROOT/name/'world.sqlite').exists():audit(name)
