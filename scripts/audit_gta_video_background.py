"""Independent image-space background motion evidence, not metric camera recovery."""
import json
from pathlib import Path
import cv2
import numpy as np
import av

ROOT=Path(__file__).resolve().parents[1]/'outputs/gta_pi_loop_20261102'


def inspect(path):
    with av.open(str(path)) as container:
        first=last=None
        for frame in container.decode(video=0):
            if first is None:first=frame.to_ndarray(format='rgb24')
            last=frame.to_ndarray(format='rgb24')
    h,w=first.shape[:2]
    mask=np.full((h,w),255,np.uint8)
    mask[int(h*.38):,int(w*.25):int(w*.72)]=0
    mask[:int(h*.035),:]=0
    sift=cv2.SIFT_create(nfeatures=3000)
    k1,d1=sift.detectAndCompute(cv2.cvtColor(first,cv2.COLOR_RGB2GRAY),mask)
    k2,d2=sift.detectAndCompute(cv2.cvtColor(last,cv2.COLOR_RGB2GRAY),mask)
    if d1 is None or d2 is None:return {'status':'insufficient_features'}
    matches=[m for pair in cv2.BFMatcher().knnMatch(d1,d2,k=2) if len(pair)==2 for m,n in [pair] if m.distance<.7*n.distance]
    if len(matches)<8:return {'status':'insufficient_matches','matches':len(matches)}
    p=np.float32([k1[m.queryIdx].pt for m in matches]);q=np.float32([k2[m.trainIdx].pt for m in matches])
    transform,inliers=cv2.findHomography(p,q,cv2.RANSAC,3.)
    if transform is None:return {'status':'no_consistent_transform'}
    selected=inliers.ravel()>0;delta=q[selected]-p[selected]
    corners=np.float32([[[0,0],[w-1,0],[w-1,h-1],[0,h-1]]])
    moved=cv2.perspectiveTransform(corners,transform)[0]
    return {'status':'measured','resolution':[w,h],'matches':len(matches),'inliers':int(selected.sum()),
        'median_background_shift_px':np.median(delta,axis=0).tolist(),
        'median_background_displacement_px':float(np.median(np.linalg.norm(delta,axis=1))),
        'homography':transform.tolist(),'corner_displacement_px':(moved-corners[0]).tolist(),
        'mask_scope':'Exclude central lower foreground and top HUD strip; no semantic mask claimed',
        'interpretation':'Image-space change only; cannot distinguish camera motion, generated scene deformation, or parallax.'}


def main():
    result={}
    for path in ROOT.rglob('controlled.mp4'):
        if path.parent.name=='render':result[str(path.relative_to(ROOT))]=inspect(path)
    (ROOT/'independent_background_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
