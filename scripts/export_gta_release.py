"""Export accepted object-memory samples with model masks and immutable evidence."""
import argparse
import base64
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import cv2
import numpy as np

from memory.gta_release import Records
from memory.gta_conditioning import reference_prompt
from memory.gta_release_media import cache_frames,cached_frame,open_video,paired_clips

_thread_state=threading.local()


def records_for(path):
    if getattr(_thread_state,'record_path',None)!=path:
        if hasattr(_thread_state,'records'):_thread_state.records.close()
        _thread_state.records=Records(path)
        _thread_state.record_path=path
    return _thread_state.records


@lru_cache(maxsize=1)
def frame_index(path):
    return {r['proxy_frame_index']:r for r in map(json.loads,path.read_text().splitlines())}


def box_overlap(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.


class MaskWorker:
    def __init__(self):
        self.lock=threading.Lock()
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
        from server_exec import connect
        self.client = connect()
        remote = '/tmp/gta_mask_worker_release_20261018.py'
        with self.client.open_sftp() as sftp:
            sftp.put(str(Path(__file__).with_name('gta_mask_worker.py')),remote)
        command = 'PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 /home/star/agentic_world/.venv/bin/python -u '+shlex.quote(remote)
        self.stdin,self.stdout,self.stderr = self.client.exec_command(command,timeout=300)
        self.stdout.channel.set_combine_stderr(True)
        while True:
            line = self.stdout.readline()
            if not line:
                raise RuntimeError('Mask worker failed to start')
            if line.strip() == 'GTA_READY':
                break

    def infer(self,image,boxes):
        with self.lock:
            return self._infer(image,boxes)

    def _infer(self,image,boxes):
        ok, data = cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,95])
        if not ok:
            raise ValueError('Failed to encode image')
        self.stdin.write(json.dumps({'image':base64.b64encode(data.tobytes()).decode(),'boxes':boxes})+'\n')
        self.stdin.flush()
        while True:
            line = self.stdout.readline()
            if not line:
                raise RuntimeError('Mask worker disconnected')
            if line.startswith('GTA_RESULT '):
                result = json.loads(line[len('GTA_RESULT '):])
                if 'error' in result:
                    raise RuntimeError('Mask inference failed: '+result['error'])
                return result['objects']

    def close(self):
        self.stdin.close()
        self.client.close()


def get_frame(path,index):
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES,index)
        ok,image = cap.read()
        if not ok:
            raise ValueError('Cannot decode reference')
        return image
    finally:
        cap.release()


def checked_image(path,image):
    if not cv2.imwrite(str(path),image):
        raise OSError('Cannot write image')


def export(window,release,worker):
    started=time.monotonic()
    timings={}
    review_dir = release/'visual_review'/window['sample_id']
    decision = json.loads((review_dir/'decision.json').read_text(encoding='utf-8'))
    verdict = release/'export_verdicts'/(window['sample_id']+'.json')
    if verdict.exists():
        return
    verdict.parent.mkdir(exist_ok=True)
    valid = [o for o in decision.get('objects',[]) if o.get('reference_acceptable') is True]
    if not decision.get('pair_acceptable') or not valid:
        verdict.write_text(json.dumps({'status':'EXCLUDED_REVIEW_UNAVAILABLE' if decision.get('status')=='REVIEW_FAILED' else 'REJECTED_VISUAL_QA','reason':decision.get('reason',decision.get('error_type'))}),encoding='utf-8')
        return
    selected = {o['label']:o for o in json.loads((review_dir/'selected_objects.json').read_text())}
    manifest = json.loads(Path(window['source_manifest']).read_text())
    source = Path(window['episode_root'])
    cached=cache_frames(window,release)
    rgb=cached_frame(cached,'rgb_video',window['reference_frame'])
    whitebox=cached_frame(cached,'proxy_video',window['reference_frame'])
    h,w = rgb.shape[:2]
    boxes = [[max(0,int(b[0]*w)),max(0,int(b[1]*h)),min(w,int(np.ceil(b[2]*w))),min(h,int(np.ceil(b[3]*h)))] for b in (o['reference_bbox_normalized'] for o in valid)]
    masks = worker.infer(rgb,boxes)
    timings['reference_and_mask_sec']=round(time.monotonic()-started,3)
    approved = []
    for obj,box,mask in zip(valid,boxes,masks):
        if mask['area'] < 32 or mask['predicted_iou'] < .6:
            continue
        if box_overlap(box,selected[obj['label']]['reference_bbox'])<.1:
            continue
        approved.append((obj,box,mask))
    if not approved:
        verdict.write_text(json.dumps({'status':'REJECTED_REFERENCE_MASK','reason':'No sufficiently supported reference mask'}),encoding='utf-8')
        return
    out = release/'dataset'/'samples'/window['sample_id']
    out.mkdir(parents=True,exist_ok=True)
    (out/'references').mkdir(exist_ok=True)
    checked_image(out/'references/rgb.png',rgb)
    checked_image(out/'references/whitebox.png',whitebox)
    loss_mask = np.ones((h,w),dtype=np.uint8)*255
    # Recorded HUD corner zones plus reviewer-specific overlays; never inpaint evidence.
    hud = [[0,0,.12,.065],[.88,.93,1,1]]+decision.get('hud_regions_normalized',[])
    for box in hud:
        if not isinstance(box,list) or len(box)!=4 or not all(isinstance(v,(int,float)) and np.isfinite(v) and 0<=v<=1 for v in box):
            raise ValueError('Invalid HUD region')
        x0,y0,x1,y1 = box
        loss_mask[int(y0*h):int(np.ceil(y1*h)),int(x0*w):int(np.ceil(x1*w))] = 0
    checked_image(out/'valid_pixels.png',loss_mask)
    edges = []
    for n,(review,box,prediction) in enumerate(approved):
        obj = selected[review['label']]
        mask = cv2.imdecode(np.frombuffer(base64.b64decode(prediction['mask_png']),np.uint8),cv2.IMREAD_GRAYSCALE)
        mask[loss_mask==0] = 0
        if int((mask>0).sum()) <32:
            continue
        name = f'object_{n:02d}'
        checked_image(out/'references'/(name+'_mask.png'),mask)
        x0,y0,x1,y1 = box
        checked_image(out/'references'/(name+'_crop.png'),rgb[y0:y1,x0:x1])
        np.save(out/'references'/(name+'_embedding.npy'),np.asarray(prediction['embedding'],dtype=np.float32))
        edge = {'object_id':window['episode_id']+':entity:'+obj['entity_id'],
                'dataset_entity_id':obj['entity_id'],'category':obj['category'],
                'reference_id':window['sample_id']+':ref0',
                'reference_rgb':'references/rgb.png','reference_whitebox':'references/whitebox.png',
                'reference_crop':'references/'+name+'_crop.png','reference_mask':'references/'+name+'_mask.png',
                'embedding':'references/'+name+'_embedding.npy','reference_bbox_xyxy':box,
                'target_projected_bboxes':obj['target_bboxes'],
                'observed_at_ns':window['observed_at_ns'],'used_at_ns':window['used_at_ns'],
                'used_until_ns':window['target_end_ns'],'used_by':window['sample_id'],
                'appearance_revision':window['episode_id']+':entity:'+obj['entity_id']+':observed:'+str(window['observed_at_ns']),
                'revision_scope':'visible exterior observation, not a claim of unchanged hidden state',
                'state_change_review':review.get('state_change'),
                'mask_provenance':'SAM2.1 box-prompted model estimate; not ground truth',
                'mask_predicted_iou':prediction['predicted_iou'],
                'lifecycle':obj['lifecycle']}
        edges.append(edge)
    if not edges:
        verdict.write_text(json.dumps({'status':'REJECTED_HUD_REFERENCE'}),encoding='utf-8');return
    (out/'prompt.txt').write_text(reference_prompt(edges),encoding='utf-8')
    clips=paired_clips(window,release)
    for media_index,(name,kind,start) in enumerate([('target_rgb.mp4','rgb_video',window['target_rgb_start']),('target_whitebox.mp4','proxy_video',window['target_start_frame'])]):
        command = ['ffmpeg','-v','error','-y','-hwaccel','cuda','-threads','1','-i',str(source/manifest[kind]),'-vf',
                   f'trim=start_frame={start}:end_frame={start+120},settb=expr=1/{manifest["video_fps"]},setpts=N',
                   '-frames:v','120','-r',str(manifest['video_fps']),'-fps_mode','cfr','-an','-c:v','libx264','-preset','fast','-crf','18','-threads','2',str(out/name)]
        if clips is not None:shutil.copyfile(clips[media_index],out/name)
        else:subprocess.run(command,check=True,timeout=180)
        cap = open_video(out/name);count=0
        while cap.read()[0]: count+=1
        cap.release()
        if count!=120: raise ValueError('Exported clip count mismatch')
    timings['through_media_sec']=round(time.monotonic()-started,3)
    frame_rows = frame_index(Path(window['source_manifest']).parent/'frames.jsonl')
    indices = [window['reference_frame'],*range(window['target_start_frame'],window['target_end_frame_exclusive'])]
    records = records_for(release/window['record_index'])
    cameras = {};worlds={}
    for i in indices:
        f = frame_rows[i]
        for field in ('lower_client_sample_seq','upper_client_sample_seq'):
            seq = f['camera_interpolation'][field]
            cameras[str(seq)] = records.camera(window['agent_id'],f['client_clock_domain'],seq)
        tick = f['world_capture_tick']
        worlds[str(tick)] = records.world(f['server_clock_domain'],tick)
    for name,data in [('frame_state_index.json',[frame_rows[i] for i in indices]),('camera_records.json',cameras),('world_records.json',worlds),('visual_review.json',decision)]:
        (out/name).write_text(json.dumps(data),encoding='utf-8')
    timings['total_sec']=round(time.monotonic()-started,3)
    (out/'processing_times.json').write_text(json.dumps(timings),encoding='utf-8')
    for name in ('rgb_video.jpg','proxy_video.jpg'):
        shutil.copyfile(review_dir/name,out/('qa_'+name))
    sample = {'sample_id':window['sample_id'],'episode_id':window['episode_id'],'agent_id':window['agent_id'],
              'source_segment':window['segment'],'frame_count':120,'fps':manifest['video_fps'],
              'media_timeline':'constant frame rate by paired decoder index; original capture times retained in frame_state_index',
              'conditions':{'prompt':'prompt.txt','prompt_provenance':'deterministic object-reference mapping; no target RGB caption',
                            'reference_pairs':[{'reference_id':window['sample_id']+':ref0','rgb':'references/rgb.png','whitebox':'references/whitebox.png'}],'target_whitebox':'target_whitebox.mp4'},
              'supervision':{'target_rgb':'target_rgb.mp4','valid_pixels':'valid_pixels.png'},
              'object_references':edges,'frame_state_index':'frame_state_index.json',
              'camera_records':'camera_records.json','world_records':'world_records.json','review':'visual_review.json',
              'quality_scope':'full-frame decode and record checks; 8 target frames visually reviewed; estimated source time alignment; coarse motion conditioning',
              'training_ready':True}
    (out/'sample.json').write_text(json.dumps(sample,indent=2),encoding='utf-8')
    verdict.write_text(json.dumps({'status':'EXPORTED','sample':str(out/'sample.json'),'object_references':len(edges)}),encoding='utf-8')
    print('EXPORTED',window['sample_id'],len(edges),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--wait-review',action='store_true')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    worker=MaskWorker()
    try:
      with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while True:
            pending_count=[0]
            def pending():
                for path in sorted((args.release/'qualification').glob('*/windows.jsonl')):
                    if not (path.parent/'summary.json').exists():continue
                    with path.open() as source:
                        for line in source:
                            window=json.loads(line)
                            if window['status']!='NEEDS_VISUAL_REVIEW':continue
                            if (args.release/'visual_review'/window['sample_id']/'decision.json').exists() and not (args.release/'export_verdicts'/(window['sample_id']+'.json')).exists():
                                pending_count[0]+=1;yield window
            for _ in pool.map(lambda window:export(window,args.release,worker),pending(),buffersize=args.workers):pass
            if not args.wait_review or ((args.release/'visual_review_complete.json').exists() and not pending_count[0]):break
            time.sleep(15)
    finally:
        worker.close()
    (args.release/'export_complete.json').write_text(json.dumps({'complete':True}))


if __name__=='__main__':main()
