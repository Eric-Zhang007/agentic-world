"""Export synchronized planned proxy and actual H3 frames for visual inspection."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import cv2
import numpy as np


def panel(image,title,width=640,height=480):
    result=np.full((height+32,width,3),245,np.uint8)
    h,w=image.shape[:2];scale=min(width/w,height/h)
    resized=cv2.resize(image,(round(w*scale),round(h*scale)))
    y=32+(height-resized.shape[0])//2;x=(width-resized.shape[1])//2
    result[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
    cv2.putText(result,title,(10,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,20,20),1,cv2.LINE_AA)
    return result


def export(preview,video,output):
    if output.exists():raise FileExistsError(output)
    proxies=[preview/'camera.png']+sorted(preview.glob('frame_*.png'))
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS)
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count!=len(proxies) or abs(fps-24)>1e-5:
        cap.release();raise ValueError('Comparison requires boundary-inclusive aligned 24 fps video')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        intermediate=Path(tmp)/'comparison.mp4'
        writer=cv2.VideoWriter(str(intermediate),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1280,512))
        if not writer.isOpened():raise RuntimeError('Video writer unavailable')
        try:
            for i,path in enumerate(proxies):
                ok,image=cap.read();proxy=cv2.imread(str(path))
                if not ok or proxy is None:raise ValueError('Missing actual or proxy frame')
                writer.write(np.concatenate([panel(proxy,f'PLAN - camera control | {i/fps:.2f}s'),
                    panel(image,f'ACTUAL - generated video | {i/fps:.2f}s')],axis=1))
        finally:writer.release();cap.release()
        subprocess.run(['ffmpeg','-v','error','-n','-i',str(intermediate),'-c:v','libx264',
                        '-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],check=True)
    print(json.dumps({'output':str(output),'frames':count,'fps':fps}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['preview','video','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();export(a.preview,a.video,a.output)
