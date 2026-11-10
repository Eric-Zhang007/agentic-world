"""Compare independent background optical-flow time series; diagnostic only."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def motion(path, limit):
    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
    values = []
    previous = None
    while len(values) < limit:
        ok, image = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(cv2.resize(image, (640, 360)), cv2.COLOR_BGR2GRAY)
        current = [None, None, 0]
        if previous is not None:
            mask = np.zeros_like(gray)
            mask[15:140, 20:620] = 255
            points = cv2.goodFeaturesToTrack(previous, 300, .015, 8, mask=mask)
            if points is not None:
                tracked, status, error = cv2.calcOpticalFlowPyrLK(previous, gray, points, None,
                    winSize=(25,25), maxLevel=3)
                back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, previous, tracked, None,
                    winSize=(25,25), maxLevel=3)
                good = (status[:,0]>0)&(back_status[:,0]>0)&(np.linalg.norm(back-points,axis=2)[:,0]<1.5)
                displacement = (tracked-points)[good,0]
                if len(displacement) >= 15:
                    dx,dy = np.median(displacement,axis=0)
                    current = [float(dx), float(dy),len(displacement)]
        values.append(current)
        previous = gray
    capture.release()
    return values


def fits(a,b,start,end):
    results = []
    for lag in range(-48,49):
        x = np.arange(start,min(end,len(a)))
        y = x+lag
        keep = (y>=0)&(y<len(b))
        x,y=x[keep],y[keep]
        av,bv=a[x,0],b[y,0]
        valid=np.isfinite(av)&np.isfinite(bv)
        av,bv=av[valid],bv[valid]
        if len(av)<30 or min(np.std(av),np.std(bv))<.05:
            continue
        results.append({'proxy_index_minus_rgb_index':lag,'pairs':len(av),
            'horizontal_flow_correlation':float(np.corrcoef(av,bv)[0,1])})
    return sorted(results,key=lambda r:r['horizontal_flow_correlation'],reverse=True)


def affine_fit(a,b):
    x=np.arange(len(a),dtype=float)
    def score(scale,offset,start,end):
        xx=x[start:end]
        yy=scale*xx+offset
        av=a[start:end,0]
        bv=np.interp(yy,np.arange(len(b)),b[:,0],left=np.nan,right=np.nan)
        valid=np.isfinite(av)&np.isfinite(bv)
        if valid.sum()<30 or min(np.std(av[valid]),np.std(bv[valid]))<.05:
            return None
        return float(np.corrcoef(av[valid],bv[valid])[0,1])
    boundary=len(a)//2
    best=(-2,None,None)
    for scale in np.arange(.85,1.201,.0025):
        for offset in range(-100,101):
            corr=score(scale,offset,0,boundary)
            if corr is not None and corr>best[0]:
                best=(corr,float(scale),offset)
    corr,scale,offset=best
    if scale is None:
        return {'status':'INSUFFICIENT_BACKGROUND_MOTION','scale':None,'offset':None}
    return {'mapping':'proxy_index = scale * rgb_index + offset', 'scale':scale,'offset':offset,
        'train_interval':[0,boundary], 'train_correlation':corr,
        'holdout_interval':[boundary,len(a)],'holdout_correlation':score(scale,offset,boundary,len(a)),
        'holdout_zero_lag_correlation':score(1,0,boundary,len(a)),
        'warning':'Image-derived diagnostic only; must validate actors and geometry independently before repair.'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--rgb',required=True)
    parser.add_argument('--proxy',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=1442)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    signals={}
    for key in ('rgb','proxy'):
        signals[key]=motion(getattr(args,key),args.limit)
        print(key,len(signals[key]),flush=True)
    a,b=[np.array(signals[k],dtype=float) for k in ('rgb','proxy')]
    intervals=[(0,len(a)),(0,len(a)//2),(len(a)//2,len(a)),(360,600)]
    report={'inputs':{'rgb':args.rgb,'proxy':args.proxy},'signals':signals,
        'affine_timing':affine_fit(a,b),
        'fits':[{'rgb_interval':[start,end],'best':fits(a,b,start,end)[:5],
                 'zero_lag':next((r for r in fits(a,b,start,end) if r['proxy_index_minus_rgb_index']==0),None)}
                for start,end in intervals],
        'scope':'Background flow timing diagnostic, not proof of geometry or entity alignment; no media modified.'}
    (args.output/'motion_timing.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report['fits'],indent=2))


if __name__=='__main__':
    main()
