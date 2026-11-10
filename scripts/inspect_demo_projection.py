"""Overlay the planned player ground anchor on real endpoint RGB for visual review."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont

root=Path(__file__).resolve().parents[1]/'outputs/meeting_demo'
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
canvas=Image.new('RGB',(1536,1000),'white');draw=ImageDraw.Draw(canvas)
draw.text((24,15),'红色十字：白模计划中主角的地面根节点投影；不是检测到的脚底位置。',font=font,fill='#24382b')
results=[]
for i,(lane,segment,plan) in enumerate([
    ('keyboard','segment1/attempt1','segment1/plan'),
    ('keyboard','segment2/attempt1','segment2/plan'),
    ('action','segment_01/attempt_01','segment_01/attempt_01/grounded_plan'),
    ('action','segment_02/attempt_01','segment_02/attempt_01/plan')]):
    frame=json.loads((root/lane/plan/'timeline.json').read_text())['frames'][-1]
    player=next(e for e in frame['entities'] if e['id']=='player');camera=frame['camera'];k=camera['intrinsics']
    eye=np.asarray(camera['position']);forward=np.asarray(camera['target'])-eye;forward/=np.linalg.norm(forward)
    right=np.cross(forward,camera['up']);right/=np.linalg.norm(right);up=np.cross(right,forward)
    delta=np.asarray(player['position'])-eye;z=float(delta@forward)
    x=(k['cx']+k['fx']*float(delta@right)/z)*1024/k['width']
    y=(k['cy']-k['fy']*float(delta@up)/z)*576/k['height']
    image=Image.open(root/lane/segment/'evidence/frames/000072.png').convert('RGB')
    paint=ImageDraw.Draw(image);paint.line((x-15,y,x+15,y),fill='#ef263e',width=3);paint.line((x,y-15,x,y+15),fill='#ef263e',width=3)
    ox=(i%2)*768;oy=62+(i//2)*465
    draw.text((ox+16,oy),lane+' / '+segment,font=font,fill='#24382b')
    canvas.paste(image.resize((768,432)),(ox,oy+28))
    results.append({'lane':lane,'segment':segment,'planned_ground_anchor_px':[x,y],
                    'scope':'Projection only; observed actor anchor not automatically measured'})
canvas.save(root/'projection_review.png')
(root/'projection_review.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
