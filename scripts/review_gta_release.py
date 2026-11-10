"""Review every mechanically qualified clip with historical RGB/proxy evidence."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import time
import threading

import cv2
import numpy as np
import requests
import yaml
from memory.gta_release_media import cache_frames,cached_frame

_api_slots=threading.BoundedSemaphore(6)


REVIEW_INSTRUCTION = '''You audit paired GTA gameplay and coarse whitebox training data for object-centric historical appearance memory.
Two synchronized contact sheets show RGB and whitebox at the same source frames. Tile 0 is the strictly historical reference; all other tiles are target supervision shown ONLY for quality review. Numbered yellow boxes are projected candidate objects, not true masks. Their labels O1 etc map to dataset entity IDs in metadata.
Decide whether this entire clip is suitable for coarse whitebox-to-RGB training and which candidate objects can use historical exterior appearance. Whitebox is intentionally coarse: upright proxies during walking/running and wireframe vehicles are acceptable. Do NOT require exact articulation, textures, lighting or windows. Reject wrong camera/world alignment, major unexplained proxy foreground obstruction, scene cuts, console/menu overlays, bad decode, or unreliable identity assignment. Small HUD text can be excluded by pixel loss mask.
For each object independently: verify a clearly visible RGB appearance in tile 0; verify the SAME instance corresponds to that projection in target tiles, not a similar nearby object; reject invisible/ambiguous references, occluded vehicle occupants, tiny unidentifiable objects, or changed visible exterior. A historical pose may differ: never freeze the historical pose. A changed door/damage/carry state must be described as target state, not silently copied from reference. If uncertain, reject the object's reference rather than inventing evidence. Background objects without stable object evidence receive no identity-memory label.
Return JSON ONLY with these exact fields:
{"pair_acceptable":true,"reason":"specific evidence","target_action":"short literal target action/camera description","hud_regions_normalized":[[x0,y0,x1,y1]],"objects":[{"label":"O1","reference_acceptable":true,"reason":"specific observed evidence","appearance":"description of only stable visible exterior","state_change":"none observed or concrete change","reference_bbox_normalized":[x0,y0,x1,y1]}]}.
BBox and HUD coordinates refer to ONE original full frame normalized 0..1, not contact-sheet coordinates. Object bbox must bound the actual RGB object in tile 0, not blindly copy the projected box. If a box cannot be assigned confidently, reference_acceptable=false. Evaluate all supplied objects. pair_acceptable may be true while no object reference is acceptable. Do not infer unseen object properties. Do not include target RGB as a proposed memory input. This is sampled visual QA, not a claim that every intermediate frame was visually inspected.'''


def sheets(window, out):
    manifest = json.loads(Path(window['source_manifest']).read_text(encoding='utf-8'))
    root = Path(window['episode_root'])
    cached=cache_frames(window,out.parents[1])
    start, end = window['target_start_frame'], window['target_end_frame_exclusive']
    offsets = np.linspace(0,end-start-1,8,dtype=int).tolist()
    objects = sorted(window['objects'],key=lambda o:(o['reference_bbox'][2]-o['reference_bbox'][0])*(o['reference_bbox'][3]-o['reference_bbox'][1]),reverse=True)[:4]
    images = []
    for kind in ('rgb_video','proxy_video'):
        cards = []
        if True:
            for tile, offset in enumerate([-1,*offsets]):
                frame_index=window['reference_frame'] if tile==0 else start+offset
                image=cached_frame(cached,kind,frame_index)
                image = cv2.resize(image,(640,360))
                width,height = manifest['resolution']
                for number,obj in enumerate(objects):
                    box = obj['reference_bbox'] if tile==0 else obj['target_bboxes'][offset]
                    x0,y0,x1,y1 = [round(v*s) for v,s in zip(box,[640/width,360/height,640/width,360/height])]
                    cv2.rectangle(image,(x0,y0),(x1,y1),(0,255,255),1)
                    cv2.putText(image,f'O{number+1}',(x0,max(35,y0)),0,.5,(0,255,255),1)
                cv2.rectangle(image,(0,0),(640,24),(0,0,0),-1)
                cv2.putText(image,f'Tile {tile} '+('HISTORICAL REFERENCE' if tile==0 else f'TARGET +{offset} frames'),(5,17),0,.45,(255,255,255),1)
                cards.append(image)
        canvas = np.concatenate([np.concatenate(cards[i:i+3],axis=1) for i in (0,3,6)],axis=0)
        path = out/(kind+'.jpg')
        if not cv2.imwrite(str(path),canvas,[cv2.IMWRITE_JPEG_QUALITY,88]):
            raise OSError('Could not write review evidence')
        images.append(path)
    selected = [{'label':f'O{i+1}',**o} for i,o in enumerate(objects)]
    (out/'selected_objects.json').write_text(json.dumps(selected),encoding='utf-8')
    return images, selected


def parse_json(content):
    text = content.strip()
    if text.startswith('```'):
        text = text.split('\n',1)[1].rsplit('```',1)[0]
    return json.loads(text)


def check_verdict(result,objects):
    if not isinstance(result.get('pair_acceptable'),bool) or not isinstance(result.get('objects'),list):
        raise ValueError('Invalid review schema')
    allowed={o['label'] for o in objects}
    if {o['label'] for o in result['objects']}!=allowed or len(result['objects'])!=len(allowed):
        raise ValueError('Missing or duplicated object review')
    for obj in result['objects']:
        if not isinstance(obj.get('reference_acceptable'),bool):raise ValueError('Invalid object verdict')
        if obj['reference_acceptable']:
            box=obj.get('reference_bbox_normalized')
            if not isinstance(box,list) or len(box)!=4 or not all(isinstance(v,(float,int)) and np.isfinite(v) and 0<=v<=1 for v in box) or box[0]>=box[2] or box[1]>=box[3]:
                raise ValueError('Invalid reference bbox')


def review_batch(windows,release,cfg):
    started=time.monotonic()
    pending=[w for w in windows if not (release/'visual_review'/w['sample_id']/'decision.json').exists()]
    if not pending:return
    if len(pending)==1:return review(pending[0],release,cfg)
    content=[{'type':'text','text':REVIEW_INSTRUCTION+'\nYou will receive independent windows. Evaluate each separately; never transfer identities or evidence between windows. Instead of one verdict, return {"reviews":[{"sample_id":"exact supplied ID", ...all fields of the verdict schema above...}]}, one entry for each supplied sample ID.'}]
    prepared={}
    for window in pending:
        out=release/'visual_review'/window['sample_id'];out.mkdir(parents=True,exist_ok=True)
        try:images,objects=sheets(window,out)
        except (OSError,ValueError,KeyError):
            review(window,release,cfg);continue
        prepared[window['sample_id']]=(out,objects)
        content.append({'type':'text','text':'BEGIN INDEPENDENT WINDOW '+json.dumps({'sample_id':window['sample_id'],
            'objects':[{'label':o['label'],'entity_id':o['entity_id'],'category':o['category']} for o in objects]})})
        for path in images:
            content.append({'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(path.read_bytes()).decode(),'detail':'high'}})
    if not prepared:return
    prepare_sec=time.monotonic()-started
    error=None
    for attempt in range(3):
        try:
            request_started=time.monotonic()
            with _api_slots:
                response=requests.post(cfg['base_url'].rstrip('/')+'/chat/completions',headers={'Authorization':'Bearer '+cfg['api_key']},
                    json={'model':cfg['model'],'messages':[{'role':'user','content':content}]},timeout=240)
                if response.status_code==429 or response.status_code>=500:time.sleep(min(60,float(response.headers.get('Retry-After',15*(attempt+1)))))
            response.raise_for_status();body=response.json()
            for out,_ in prepared.values():
                (out/f'response_batch_attempt_{attempt}.json').write_text(json.dumps(body),encoding='utf-8')
            results=parse_json(body['choices'][0]['message']['content'])['reviews']
            if len(results)!=len(prepared) or {r['sample_id'] for r in results}!=set(prepared):raise ValueError('Batch sample IDs mismatch')
            for result in results:check_verdict(result,prepared[result['sample_id']][1])
            for result in results:
                key=result.pop('sample_id');out,_=prepared[key]
                result.update(status='REVIEWED',model=cfg['model'],sampled_target_frames=8,review_batch_size=len(prepared),
                    batch_prepare_sec=round(prepare_sec,3),batch_request_sec=round(time.monotonic()-request_started,3),
                    frame_extraction='SEQUENTIAL_EXACT_DECODER_INDEX',review_scope='sampled visual QA plus full media/record checks',batch_usage=body.get('usage'))
                (out/'decision.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
                print(key,'REVIEWED',result['pair_acceptable'],flush=True)
            return
        except (requests.RequestException,ValueError,KeyError,IndexError,TypeError) as caught:
            error=caught
            if attempt<2:time.sleep(3*(attempt+1))
    for key,(out,_) in prepared.items():
        (out/'decision.json').write_text(json.dumps({'status':'REVIEW_FAILED','error_type':type(error).__name__,
            'error':str(error)[:1000],'pair_acceptable':False,'objects':[]}),encoding='utf-8')
        print(key,'REVIEW_FAILED',flush=True)


def review(window, release, cfg):
    out = release/'visual_review'/window['sample_id']
    out.mkdir(parents=True,exist_ok=True)
    if (out/'decision.json').exists():
        return
    try:
        images, objects = sheets(window,out)
        content = [{'type':'text','text':REVIEW_INSTRUCTION+'\nMetadata: '+json.dumps({
            'sample_id':window['sample_id'],'objects':[{'label':o['label'],'entity_id':o['entity_id'],'category':o['category']} for o in objects]})}]
        for path in images:
            content.append({'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(path.read_bytes()).decode(),'detail':'high'}})
        result = None
        for attempt in range(3):
            try:
                with _api_slots:
                    response = requests.post(cfg['base_url'].rstrip('/')+'/chat/completions',
                        headers={'Authorization':'Bearer '+cfg['api_key']},
                        json={'model':cfg['model'],'messages':[{'role':'user','content':content}]},timeout=180)
                    if response.status_code==429 or response.status_code>=500:
                        time.sleep(min(60,float(response.headers.get('Retry-After',15*(attempt+1)))))
                response.raise_for_status()
                body = response.json()
                (out/f'response_attempt_{attempt}.json').write_text(json.dumps(body),encoding='utf-8')
                result = parse_json(body['choices'][0]['message']['content'])
                if not isinstance(result.get('pair_acceptable'),bool) or not isinstance(result.get('objects'),list):
                    raise ValueError('Invalid review schema')
                allowed = {o['label'] for o in objects}
                if {o['label'] for o in result['objects']} != allowed or len(result['objects']) != len(allowed):
                    raise ValueError('Missing or duplicated object review')
                for obj in result['objects']:
                    if not isinstance(obj.get('reference_acceptable'),bool):
                        raise ValueError('Invalid object verdict')
                    if obj['reference_acceptable']:
                        box = obj.get('reference_bbox_normalized')
                        if not isinstance(box,list) or len(box)!=4 or not all(isinstance(v,(float,int)) and np.isfinite(v) and 0<=v<=1 for v in box) or box[0]>=box[2] or box[1]>=box[3]:
                            raise ValueError('Invalid reference bbox')
                result.update(status='REVIEWED',model=cfg['model'],sampled_target_frames=8,
                              frame_extraction='SEQUENTIAL_EXACT_DECODER_INDEX',
                              review_scope='sampled visual QA plus full media/record checks',usage=body.get('usage'))
                break
            except (requests.RequestException,ValueError,KeyError,IndexError,TypeError) as error:
                result = {'status':'REVIEW_FAILED','error_type':type(error).__name__,'error':str(error)[:1000],'pair_acceptable':False,'objects':[]}
                if attempt<2:
                    time.sleep(2*(attempt+1))
        (out/'decision.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(window['sample_id'],result['status'],result['pair_acceptable'],flush=True)
    except (OSError,ValueError,KeyError) as error:
        (out/'decision.json').write_text(json.dumps({'status':'REVIEW_FAILED','error_type':type(error).__name__,'pair_acceptable':False,'objects':[]}),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--batch-size',type=int,default=3)
    parser.add_argument('--wait-qualification',action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path('configs/default.yaml').read_text())['agent']
    processed = set()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while True:
            consumed=[0]
            def batches():
                batch=[]
                for path in sorted((args.release/'qualification').glob('*/windows.jsonl')):
                    if path in processed or not (path.parent/'summary.json').exists():continue
                    processed.add(path);consumed[0]+=1
                    with path.open() as source:
                        for line in source:
                            w=json.loads(line)
                            if w['status']!='NEEDS_VISUAL_REVIEW' or (args.release/'visual_review'/w['sample_id']/'decision.json').exists():continue
                            batch.append(w)
                            if len(batch)==args.batch_size:yield batch;batch=[]
                if batch:yield batch
            groups=iter(batches())
            active=set()
            for _ in range(args.workers):
                group=next(groups,None)
                if group is not None:active.add(pool.submit(review_batch,group,args.release,cfg))
            while active:
                finished,active=wait(active,return_when=FIRST_COMPLETED)
                for future in finished:
                    future.result()
                    group=next(groups,None)
                    if group is not None:active.add(pool.submit(review_batch,group,args.release,cfg))
            if not args.wait_qualification or ((args.release/'qualification_summary.json').exists() and not consumed[0]):
                break
            time.sleep(15)
    (args.release/'visual_review_complete.json').write_text(json.dumps({'segments_consumed':len(processed)}),encoding='utf-8')


if __name__ == '__main__':
    main()
