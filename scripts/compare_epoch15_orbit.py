"""Frame-aligned whitebox / old baseline / epoch-15 comparison."""
from pathlib import Path
import json
import cv2
import numpy as np
import av

root=Path(__file__).resolve().parents[1]
out=root/'outputs/epoch15_orbit_20261103'
old=root/'outputs/native_whitebox_reference_20261013'
def old_video(directory):
    flat=directory/'native_video.mp4'
    return flat if flat.exists() else directory/'render/native_video.mp4'
paths=[old/'whitebox_reference.mp4',old_video(old),old_video(old/'depth_half'),out/'depth_off/render/depth_off.mp4',out/'depth_half/render/depth_half.mp4']
labels=['Whitebox reference','Previous base: depth OFF','Previous base: depth 0.5','Epoch 15: depth OFF','Epoch 15: depth 0.5']
caps=[cv2.VideoCapture(str(p)) for p in paths]
assert all(c.isOpened() and int(c.get(cv2.CAP_PROP_FRAME_COUNT))==124 for c in caps)
with av.open(str(out/'comparison.mp4'),'w') as container:
    stream=container.add_stream('libx264',rate=24)
    stream.width=1920;stream.height=576;stream.pix_fmt='yuv420p';stream.options={'crf':'18'}
    sheets=[]
    for i in range(124):
        canvas=np.zeros((576,1920,3),dtype=np.uint8)
        for j,cap in enumerate(caps):
            ok,frame=cap.read();assert ok
            x=j*384
            canvas[48:336,x:x+384]=cv2.resize(frame,(384,288))
            cv2.putText(canvas,labels[j],(x+5,27),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1)
            cv2.putText(canvas,f'Frame {i} / {i/24:.3f}s',(x+5,365),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        if i in (0,31,62,93,123):
            cv2.imwrite(str(out/f'comparison_{i:03d}.jpg'),canvas[:380])
            sheets.append(canvas[:380])
        for packet in stream.encode(av.VideoFrame.from_ndarray(canvas,format='bgr24')):container.mux(packet)
    for packet in stream.encode():container.mux(packet)
for c in caps:c.release()
cv2.imwrite(str(out/'contact.jpg'),np.concatenate(sheets,axis=0))
(out/'comparison_sources.json').write_text(json.dumps([{'label':l,'path':str(p)} for l,p in zip(labels,paths)],indent=2))
print(out/'comparison.mp4')
