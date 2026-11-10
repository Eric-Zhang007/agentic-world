"""Reusable background features, historical fallback, and quantity accounting."""
import argparse
from collections import Counter
from functools import lru_cache
from pathlib import Path
import shutil
import zipfile
import cv2
import numpy as np
import orjson


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('D:/GTA_passage_pairs_20261020'))
    args=p.parse_args();root=args.root;cache=root/'features';cache.mkdir(exist_ok=True)
    points={r['point_id']:r for r in map(orjson.loads,(root/'points.jsonl').read_bytes().splitlines())}
    cv2.setNumThreads(2);sift=cv2.SIFT_create(nfeatures=1800)
    @lru_cache(maxsize=128)
    def features(key):
        file=cache/(key+'.npz')
        if file.exists():
            try:
                with file.open('rb') as stream:
                    with np.load(stream) as data:
                        return data['xy'],data['desc'],float(data['fraction'])
            except (zipfile.BadZipFile, EOFError, ValueError, KeyError):
                print('REBUILD_INTERRUPTED_FEATURE',key,flush=True)
                file.unlink()
        if not file.exists():
            point=points[key];image=cv2.imread(str(Path(point['cache_root'])/f'rgb_video_{point["frame_index"]:06d}.png'))
            if image is None:raise ValueError('Missing cached frame')
            image=cv2.resize(image,(640,360));mask=np.full((360,640),255,np.uint8)
            mask[:22]=0;mask[324:]=0;mask[:,:13]=0;mask[:,627:]=0
            w,h=point['resolution']
            for box in point['dynamic_boxes']:
                x0,y0,x1,y1=[round(v*s) for v,s in zip(box,[640/w,360/h,640/w,360/h])]
                mask[max(0,y0):min(360,y1+1),max(0,x0):min(640,x1+1)]=0
            k,d=sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),mask)
            temporary=file.with_suffix('.partial.npz')
            np.savez_compressed(temporary,xy=np.array([v.pt for v in k],np.float32).reshape(-1,2),
                                desc=d if d is not None else np.empty((0,128),np.float32),fraction=(mask>0).mean())
            temporary.replace(file)
        with np.load(file) as data:return data['xy'],data['desc'],float(data['fraction'])
    def evidence(a,b):
        ka,da,fa=features(a);kb,db,fb=features(b)
        if min(len(da),len(db))<8:return None
        raw=cv2.BFMatcher().knnMatch(da,db,k=2)
        matches=[m[0] for m in raw if len(m)==2 and m[0].distance<.75*m[1].distance]
        unique={}
        for m in sorted(matches,key=lambda x:x.distance):unique.setdefault(m.trainIdx,m)
        matches=list(unique.values())
        if len(matches)<12:return None
        axy=np.float32([ka[m.queryIdx] for m in matches]);bxy=np.float32([kb[m.trainIdx] for m in matches])
        _,mask=cv2.findHomography(axy,bxy,cv2.RANSAC,3.)
        if mask is None:return None
        keep=mask.ravel().astype(bool)
        cells=[len({(min(3,int(x/160)),min(2,int(y/120))) for x,y in xy[keep]}) for xy in (axy,bxy)]
        result={'inliers':int(keep.sum()),'ratio':float(keep.mean()),'grid_cells':cells,
                'background_fraction':[fa,fb],'good_matches':len(matches)}
        if keep.sum()>=16 and keep.mean()>=.35 and min(cells)>=3 and min(fa,fb)>=.15:return result
        return None
    accepted={};attempted=set();attempt_count=0
    ledger=root/'background_matches.jsonl'
    if ledger.exists():
        for line in ledger.read_bytes().splitlines():
            row=orjson.loads(line);accepted[row['target_point']]=row
    for name in ('candidates_r20_t5_a90.jsonl','candidates_r30_t2_a110.jsonl'):
        rows=[orjson.loads(line) for line in (root/name).read_bytes().splitlines()]
        with ledger.open('ab') as destination:
            for i,row in enumerate(rows):
                target=row['target_point']
                if target in accepted:continue
                for rank,option in enumerate(row['options']):
                    if option['kind']=='gap_unknown':continue
                    pair=(option['reference_point'],target)
                    if pair in attempted:continue
                    attempted.add(pair);attempt_count+=1
                    proof=evidence(*pair)
                    if proof:
                        value={**option,'background_evidence':proof,'reference_rank':rank,
                               'search_stage':name,'review_status':'AUTOMATIC_BACKGROUND_EVIDENCE_ONLY'}
                        destination.write(orjson.dumps(value)+b'\n');destination.flush();accepted[target]=value;break
                if i%100==0:
                    status={'stage':name,'processed_targets':i,'accepted_targets':len(accepted),'attempted_edges_this_run':attempt_count}
                    (root/'progress.json').write_bytes(orjson.dumps(status));print(status,flush=True)
    out=root/'dataset';out.mkdir(exist_ok=True);manifest=[]
    for i,row in enumerate(sorted(accepted.values(),key=lambda r:r['target_point'])):
        key=f'passage_{i:05d}';folder=out/'samples'/key;folder.mkdir(parents=True,exist_ok=True)
        refs=[]
        for side,pid in [('reference',row['reference_point']),('target',row['target_point'])]:
            point=points[pid]
            for kind,label in [('rgb_video','rgb'),('proxy_video','whitebox')]:
                source=Path(point['cache_root'])/f'{kind}_{point["frame_index"]:06d}.png'
                shutil.copy2(source,folder/f'{side}_{label}.png')
            (folder/f'{side}_state.json').write_bytes(orjson.dumps(point))
        sample={**row,'sample_id':key,'conditions':{'reference_pair':{'rgb':'reference_rgb.png','whitebox':'reference_whitebox.png'},'target_whitebox':'target_whitebox.png'},
                'supervision':{'target_rgb':'target_rgb.png'},'training_status':'PAIR_ONLY_VIDEO_AND_VISUAL_QA_PENDING'}
        (folder/'sample.json').write_bytes(orjson.dumps(sample,option=orjson.OPT_INDENT_2))
        manifest.append({'sample_id':key,'sample_root':'samples/'+key,**row})
    (out/'samples.jsonl').write_bytes(b''.join(orjson.dumps(v)+b'\n' for v in manifest))
    report={'exported_pairs':len(manifest),'kinds':dict(Counter(v['kind'] for v in manifest)),
            'episodes':dict(Counter(v['episode_id'] for v in manifest)),
            'ages':dict(Counter('under10s' if v['age_sec']<10 else '10to30s' if v['age_sec']<30 else '30to120s' if v['age_sec']<120 else 'over120s' for v in manifest)),
            'paid_api_calls':0,'status':'PAIR_EXTRACTION_DONE_TRAINING_QA_PENDING'}
    (root/'report.json').write_bytes(orjson.dumps(report,option=orjson.OPT_INDENT_2));print(report,flush=True)


if __name__=='__main__':main()
