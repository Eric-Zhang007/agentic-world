"""Make a synchronized new-scene whitebox/depth/H3 review artifact."""
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np

root=Path(__file__).resolve().parents[1]/'outputs/aligned_rollout_20261014'
paths=[root/'whitebox.mp4',root/'control_input.mp4',root/'generated.mp4']
labels=['Whitebox RGB / Video 1','Depth control 0.5','H3 / agent-written prompt']
caps=[cv2.VideoCapture(str(p)) for p in paths]
assert all(c.isOpened() and int(c.get(cv2.CAP_PROP_FRAME_COUNT))==124 for c in caps)
work=root/'comparison_work.mp4';target=root/'comparison.mp4'
writer=cv2.VideoWriter(str(work),cv2.VideoWriter_fourcc(*'mp4v'),24,(1536,424))
assert writer.isOpened()
sheets=[]
for i in range(124):
    canvas=np.zeros((424,1536,3),np.uint8)
    for j,c in enumerate(caps):
        ok,f=c.read();assert ok
        canvas[40:,j*512:(j+1)*512]=cv2.resize(f,(512,384))
        cv2.putText(canvas,labels[j]+f' | {i/24:.2f}s',(j*512+8,25),
                    cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    writer.write(canvas)
    if i in [0,12,31,62,93,123]:
        cv2.imwrite(str(root/f'comparison_{i:03d}.jpg'),canvas);sheets.append(canvas)
writer.release()
for c in caps:c.release()
subprocess.run(['ffmpeg','-v','error','-n','-i',str(work),'-c:v','libx264','-crf','18',
                '-pix_fmt','yuv420p','-movflags','+faststart',str(target)],check=True)
work.unlink();cv2.imwrite(str(root/'contact_sheet.jpg'),np.concatenate(sheets,axis=0))
c=cv2.VideoCapture(str(target));count=0
while c.read()[0]:count+=1
c.release();assert count==124
(root/'comparison.json').write_text(json.dumps({'frames':count,'fps':24,'sources':[str(p) for p in paths]}))
print(target)
