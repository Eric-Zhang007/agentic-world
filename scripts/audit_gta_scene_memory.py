"""Batch visual audit of scene memory pairs; never silently promote candidates."""
import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import time

import cv2
import numpy as np
import requests
import yaml


INSTRUCTION = '''Audit coarse whitebox-to-RGB SCENE BACKGROUND MEMORY training pairs.
Each labelled image is one independent sample: leftmost column is the strictly
historical RGB (top)/whitebox (bottom) reference. Five remaining columns are
target RGB(top)/whitebox(bottom), at frames 0,30,60,90,119 of a 120-frame clip.
Target RGB is supervision for auditing, NEVER a conditioning reference.
Judge each sample independently. Require identifiable shared static background
between reference and target (specific building, road layout, terrain or structure),
not just the same person, similar textures, or a generic flat floor. A close
camera alone is not evidence. Mark uncertain if overlap cannot be established.
Coarse geometry is expected: do not demand texture, foliage details or exact
limb animation. Wireframe vehicles are acceptable only when they do not create a
MAJOR near-field background visibility disagreement. Reject a large truck/body
blocking most of RGB while proxy shows the hidden background, opaque proxy walls
where RGB shows transparent fences, wrong camera/layout, major timing/state
disagreements, menu/console overlays, broken decode or cuts. A small ordinary HUD
may be masked; a console overlay over scene content is not ordinary HUD.
Distinguish shared background from evidence of leave-and-return: do not claim
long-term memory merely because a pair is acceptable. Do not infer unseen detail.
For every sample return exactly one JSON entry:
{"reviews":[{"sample_id":"...","alignment_ok":true,"background_memory_useful":true,
"no_major_occlusion_conflict":true,"no_ui_contamination":true,"uncertain":false,
"background_evidence":"specific shared visible background with columns",
"reason":"specific evidence including defects if any"}]}.
All five flags are booleans; uncertainty is NOT acceptance. This is sampled
visual audit, not full temporal inspection or demonstrated training performance.'''


def validate_results(body, ids):
    rows = body['reviews']
    if len(rows) != len(ids) or {r['sample_id'] for r in rows} != set(ids):
        raise ValueError('Review sample IDs mismatch')
    for row in rows:
        for key in ('alignment_ok', 'background_memory_useful', 'no_major_occlusion_conflict', 'no_ui_contamination', 'uncertain'):
            if type(row.get(key)) is not bool:
                raise ValueError('Invalid boolean: ' + key)
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in ('reason', 'background_evidence')):
            raise ValueError('Missing specific review evidence')
        row['accepted'] = all(row[k] for k in ('alignment_ok','background_memory_useful','no_major_occlusion_conflict','no_ui_contamination')) and not row['uncertain']
    return rows


def card(folder):
    refs = [cv2.imread(str(folder / name)) for name in ('reference_rgb.png','reference_whitebox.png')]
    target = cv2.imread(str(folder / 'video_review.jpg'))
    if target is None or any(r is None for r in refs):
        raise ValueError('Missing review evidence: ' + folder.name)
    ref = np.vstack([cv2.resize(r, (320,180)) for r in refs])
    canvas = np.hstack([ref, cv2.resize(target, (1600,360))])
    ok, encoded = cv2.imencode('.jpg',canvas,[cv2.IMWRITE_JPEG_QUALITY,90])
    if not ok: raise ValueError('Image encode failed')
    return encoded.tobytes()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pilot-indices',type=int,nargs='*')
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load(Path('configs/default.yaml').read_text())['agent']
    chosen=[]
    for root in a.roots:
        if a.pilot_indices is not None:
            keys=json.loads((root/'pending_audit_full_source/index.json').read_bytes())
            paths=[root/'training/samples'/keys[i]/'sample.json' for i in a.pilot_indices]
        else:
            paths=sorted((root/'training/samples').glob('*/sample.json'))
        for path in paths:
            sample=json.loads(path.read_bytes())
            if a.pilot_indices is None and sample['training_status'].startswith('EXCLUDED'):continue
            out=a.output/(sample['sample_id']+'.json')
            if out.exists():continue
            chosen.append((path.parent,out))
    batches=[chosen[i:i+4] for i in range(0,len(chosen),4)]
    (a.output/'run.json').write_text(json.dumps({'samples_pending':len(chosen),'batches':len(batches),'model':cfg['model'],'roots':[str(r) for r in a.roots],'scope':'Sampled scene-background audit, not final training acceptance'},indent=2))

    def audit(batch):
        content=[{'type':'text','text':INSTRUCTION}]
        ids=[]
        for folder,out in batch:
            ids.append(folder.name)
            content.extend([{'type':'text','text':'sample_id: '+folder.name},
                {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(card(folder)).decode(),'detail':'high'}}])
        response=requests.post(cfg['base_url'].rstrip('/')+'/chat/completions',
            headers={'Authorization':'Bearer '+cfg['api_key']},
            json={'model':cfg['model'],'messages':[{'role':'user','content':content}]},timeout=240)
        response.raise_for_status()
        body=response.json()
        text=body['choices'][0]['message']['content'].strip()
        if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
        results=validate_results(json.loads(text),ids)
        by_id={r['sample_id']:r for r in results}
        for folder,out in batch:
            row=by_id[folder.name]
            row.update(sample_path=str(folder/'sample.json'),model=cfg['model'],
                sampled_target_frames=[0,30,60,90,119],reviewed_at=time.time(),batch_ids=ids,
                batch_usage=body.get('usage'),scope='Historical pair plus five target pairs; not exhaustive temporal review')
            out.write_text(json.dumps(row,indent=2),encoding='utf-8')
        return Counter('accepted' if r['accepted'] else 'rejected_or_uncertain' for r in results)

    total=Counter()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        pending=iter(batches)
        active={}
        for _ in range(a.workers):
            batch=next(pending,None)
            if batch is not None:active[pool.submit(audit,batch)]=batch
        i=0
        failed=False
        while active:
            done,_=wait(active,return_when=FIRST_COMPLETED)
            for future in done:
                batch=active.pop(future)
                try:
                    total.update(future.result())
                except Exception as error:
                    failed=True
                    (a.output/'error.json').write_text(json.dumps({'error_type':type(error).__name__,
                        'sample_ids':[f.name for f,o in batch],'status':'STOPPED_ON_ERROR'},indent=2))
                    print('STOPPED_ON_ERROR',type(error).__name__,flush=True)
                i+=1
                if not failed:
                    batch=next(pending,None)
                    if batch is not None:active[pool.submit(audit,batch)]=batch
            status={'completed_batches':i,'total_batches':len(batches),'counts_this_run':dict(total),'updated_at':time.time()}
            (a.output/'status.json').write_text(json.dumps(status,indent=2))
            print(json.dumps(status),flush=True)
        if failed:raise SystemExit(1)
    print('AUDIT_REQUESTS_COMPLETE',flush=True)


if __name__=='__main__':main()
