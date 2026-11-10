"""Export new historical scene references with previously decoded target clips."""
import argparse
from collections import Counter
from functools import lru_cache
from pathlib import Path
import shutil
import time
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import orjson

from memory.gta_validation import camera_exclusion
from memory.gta_release_media import open_video,paired_clips,source_media_path,rgb_file_index
from scripts.prepare_gta_scene_training import clip


def save(path,value):path.write_bytes(orjson.dumps(value,option=orjson.OPT_INDENT_2))


@lru_cache(maxsize=2)
def frames(path):
    return {r['proxy_frame_index']:r for r in map(orjson.loads,Path(path).read_bytes().splitlines())}


def verify_video(path,first_image,last_image):
    cap=open_video(path);count=0;first=None;last=None;cards=[]
    try:
        while True:
            ok,image=cap.read()
            if not ok:break
            if count==0:first=image
            if count in (0,30,60,90,119):cards.append(cv2.resize(image,(320,180)))
            last=image;count+=1
    finally:cap.release()
    if count!=120 or first.shape!=first_image.shape:raise ValueError('Video length/shape mismatch')
    scores=[cv2.PSNR(first,first_image)]
    if last_image is not None:scores.append(cv2.PSNR(last,last_image))
    if min(scores)<26:raise ValueError('Video/frame alignment mismatch')
    return cards,scores


def export(row,points,root,release):
    target=points[row['target_point']];ref=points[row['reference_point']]
    key=target['point_id'];out=root/'training'/'samples'/key;out.mkdir(parents=True,exist_ok=True)
    result=out/'sample.json'
    if result.exists():return orjson.loads(result.read_bytes())['training_status']
    mpath=Path(target['source_manifest']);manifest=orjson.loads(mpath.read_bytes())
    segment=target['episode_id']+'_'+target['agent_id']+'_'+target['segment']
    idx=target['frame_index'];all_rows=frames(str(mpath.parent/'frames.jsonl'))
    selected=[all_rows.get(i) for i in range(idx,idx+120)]
    errors=orjson.loads((release/'qualification'/segment/'frame_errors.json').read_bytes())
    if any(r is None or camera_exclusion(r) or str(r['proxy_frame_index']) in errors for r in selected):
        save(result,{'sample_id':key,'training_status':'EXCLUDED_TARGET_CAMERA','reference':row});return 'EXCLUDED_TARGET_CAMERA'
    if any(rgb_file_index(b)!=rgb_file_index(a)+1 or b['server_time_estimate_ns']<=a['server_time_estimate_ns']
           or a['server_clock_domain']!=b['server_clock_domain'] for a,b in zip(selected,selected[1:])):
        raise ValueError('Invalid target timeline')
    old_id=segment+f'_w{idx//120-1:03d}'
    decision_path=release/'visual_review'/old_id/'decision.json'
    decision=orjson.loads(decision_path.read_bytes()) if decision_path.exists() else {}
    if decision.get('status')=='REVIEWED' and decision.get('pair_acceptable') is False:
        save(result,{'sample_id':key,'training_status':'EXCLUDED_PREVIOUS_TARGET_QA','reference':row,'reason':decision.get('reason')})
        return 'EXCLUDED_PREVIOUS_TARGET_QA'
    for side,p in [('reference',ref),('target',target)]:
        for kind,label in [('rgb_video','rgb'),('proxy_video','whitebox')]:
            shutil.copy2(Path(p['cache_root'])/f'{kind}_{p["frame_index"]:06d}.png',out/f'{side}_{label}.png')
        save(out/f'{side}_state.json',p)
    videos=[];scores={}
    if rgb_file_index(selected[0])%120==0:
        paired_clips({**target,'episode_root':target['source_root'],
                      'target_rgb_start':rgb_file_index(selected[0]),'target_start_frame':idx},release)
    for kind,name,start in [('rgb_video','target_rgb.mp4',rgb_file_index(selected[0])),('proxy_video','target_whitebox.mp4',idx)]:
        existing=release/'dataset'/'samples'/old_id/name
        cached=release/'paired_clips'/segment/f'{kind}_{start//120:04d}.mp4'
        dest=out/name
        if existing.exists():shutil.copy2(existing,dest)
        elif start%120==0 and cached.exists():shutil.copy2(cached,dest)
        else:
            if dest.exists():dest.unlink()
            clip(source_media_path(target,manifest,kind),start,dest,
                 'E:/gta-data-tools/ffmpeg-compat/imageio_ffmpeg/binaries/ffmpeg-win64-v4.2.2.exe',manifest['video_fps'])
        first=cv2.imread(str(out/('target_rgb.png' if kind=='rgb_video' else 'target_whitebox.png')))
        last_path=Path(target['cache_root'])/f'{kind}_{idx+119:06d}.jpg'
        last=cv2.imread(str(last_path)) if last_path.exists() else None
        cards,psnr=verify_video(dest,first,last);videos.append(cards);scores[kind]=psnr
    valid=np.full(first.shape[:2],255,np.uint8)
    for box in decision.get('hud_regions_normalized',[]):
        x0,y0,x1,y1=[round(v*s) for v,s in zip(box,[valid.shape[1],valid.shape[0]]*2)]
        valid[y0:y1,x0:x1]=0
    cv2.imwrite(str(out/'valid_pixels.png'),valid)
    cards=[]
    for i in range(5):
        tile=np.vstack([videos[0][i],videos[1][i]])
        cv2.putText(tile,f'+{[0,30,60,90,119][i]}',(5,20),0,.5,(0,255,255),1);cards.append(tile)
    cv2.imwrite(str(out/'video_review.jpg'),np.hstack(cards),[cv2.IMWRITE_JPEG_QUALITY,90])
    four=[cv2.resize(cv2.imread(str(out/name)),(640,360)) for name in ('reference_rgb.png','reference_whitebox.png','target_rgb.png','target_whitebox.png')]
    comparison=np.vstack([np.hstack(four[:2]),np.hstack(four[2:])]);cv2.putText(comparison,f'{key} age={row["age_sec"]:.1f}s',(8,25),0,.55,(0,255,255),1)
    cv2.imwrite(str(out/'comparison.jpg'),comparison)
    prompt='Render the target whitebox video of this location. Use the earlier paired RGB and whitebox frame as a reference for overlapping background surfaces. Follow target camera and geometry. Do not copy historical people, vehicles, or their poses. Unseen background is unconstrained by the reference.'
    (out/'prompt.txt').write_text(prompt,encoding='utf-8')
    save(out/'target_frame_states.json',selected)
    sample={'sample_id':key,'episode_id':target['episode_id'],'frame_count':120,'fps':manifest['video_fps'],
        'training_status':'VIDEO_QA_PENDING',
        'reference':row,'reference_time_ns':ref['time_ns'],'target_time_ns':target['time_ns'],
        'conditions':{'reference_pair':{'rgb':'reference_rgb.png','whitebox':'reference_whitebox.png'},'target_whitebox':'target_whitebox.mp4','prompt':'prompt.txt'},
        'supervision':{'target_rgb':'target_rgb.mp4','valid_pixels':'valid_pixels.png'},
        'alignment_psnr':scores,'inherited_target_review':decision,
        'visual_reference_review':'Not individually visually reviewed; background features are automatic evidence only'}
    save(result,sample);return sample['training_status']


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('D:/GTA_passage_pairs_20261020'))
    p.add_argument('--release',type=Path,default=Path('D:/GTA_memory_release_20261018'));p.add_argument('--follow',action='store_true')
    p.add_argument('--workers',type=int,default=3)
    os.environ.setdefault('FFMPEG_BINARY','E:/gta-data-tools/ffmpeg-compat/imageio_ffmpeg/binaries/ffmpeg-win64-v4.2.2.exe')
    args=p.parse_args();points={r['point_id']:r for r in map(orjson.loads,(args.root/'points.jsonl').read_bytes().splitlines())}
    done=set();counts=Counter()
    while True:
        pending=[row for row in map(orjson.loads,(args.root/'background_matches.jsonl').read_bytes().splitlines()) if row['target_point'] not in done]
        def work(row):return row['target_point'],export(row,points,args.root,args.release)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for key,status in pool.map(work,pending,buffersize=args.workers):
                done.add(key);counts[status]+=1
                save(args.root/'training_progress.json',dict(counts))
                if len(done)%25==0:print(len(done),dict(counts),flush=True)
        if not args.follow:break
        if (args.root/'report.json').exists():
            latest=[orjson.loads(line) for line in (args.root/'background_matches.jsonl').read_bytes().splitlines()]
            if all(row['target_point'] in done for row in latest):break
        time.sleep(5)
    print('EXPORT_DONE',dict(counts),flush=True)


if __name__=='__main__':main()
