"""Attach aligned target clips to reviewed scene references, without paid APIs."""
import argparse
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import orjson

from memory.gta_release_media import open_video,source_media_path
from memory.gta_validation import camera_exclusion


def save(path,data):
    path.write_bytes(orjson.dumps(data,option=orjson.OPT_INDENT_2))


def clip(source, start, destination, ffmpeg, fps):
    if destination.exists():return
    subprocess.run([ffmpeg,'-v','error','-y','-threads','1','-i',str(source),'-an',
        '-vf',f'trim=start_frame={start}:end_frame={start+120},settb=expr=1/{fps},setpts=N',
        '-r',str(fps),'-vsync','1','-c:v','h264_nvenc','-preset','slow','-cq','18','-b:v','0',str(destination)],
        check=True,timeout=180,capture_output=True)


def decode(path):
    cap=open_video(path);frames=[]
    try:
        while True:
            ok,image=cap.read()
            if not ok:break
            frames.append(image)
    finally:cap.release()
    if len(frames)!=120:raise ValueError('Target video is not exactly 120 frames')
    return frames


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('D:/GTA_scene_pairs_complete_20261019'))
    parser.add_argument('--release',type=Path,default=Path('D:/GTA_memory_release_20261018'))
    parser.add_argument('--ffmpeg',default='E:/gta-data-tools/ffmpeg-compat/imageio_ffmpeg/binaries/ffmpeg-win64-v4.2.2.exe')
    args=parser.parse_args();dataset=args.root/'dataset'
    review=orjson.loads(Path('configs/gta_scene_review_20261019.json').read_bytes())
    labels={f'scene_pair_{i:05d}':{k:v for k,v in group.items() if k!='ids'} for group in review['groups'] for i in group['ids']}
    rows=[orjson.loads(line) for line in (dataset/'samples.jsonl').read_bytes().splitlines()]
    if set(labels)!={r['sample_id'] for r in rows}:raise ValueError('Visual review does not cover exact candidate set')
    ledger=[]
    for row in rows:
        key=row['sample_id'];folder=dataset/row['sample_root'];sample=orjson.loads((folder/'sample.json').read_bytes())
        sample['visual_review']={'scope':review['scope'],'reviewer':review['reviewer'],**labels[key]}
        reason=review['exclusions'].get(key)
        state=orjson.loads((folder/'target_state.json').read_bytes())
        source_path=Path(state['source_manifest']);manifest=orjson.loads(source_path.read_bytes())
        segment_key=state['episode_id']+'_'+state['agent_id']+'_'+state['segment']
        index=state['frame_index']
        all_frames={r['proxy_frame_index']:r for r in map(orjson.loads,(source_path.parent/'frames.jsonl').read_bytes().splitlines())}
        frames=[all_frames.get(i) for i in range(index,index+120)]
        errors=orjson.loads((args.release/'qualification'/segment_key/'frame_errors.json').read_bytes())
        if not reason and any(r is None or camera_exclusion(r) or str(r['proxy_frame_index']) in errors for r in frames):
            reason='Missing or invalid camera/world frame in target video interval'
        if not reason:
            if any(b['rgb_decode_index']!=a['rgb_decode_index']+1 or b['server_time_estimate_ns']<=a['server_time_estimate_ns']
                   or b['server_clock_domain']!=a['server_clock_domain'] for a,b in zip(frames,frames[1:])):
                reason='Noncontiguous decoder indices or target clock timeline'
        if reason:
            sample['training_status']='DIAGNOSTIC_ONLY';sample['exclusion_reason']=reason
            save(folder/'sample.json',sample);ledger.append({'sample_id':key,'status':'DIAGNOSTIC_ONLY','reason':reason})
            continue
        videos=[]
        for kind,name,start in [('rgb_video','target_rgb.mp4',frames[0]['rgb_decode_index']),('proxy_video','target_whitebox.mp4',index)]:
            clip(source_media_path(state,manifest,kind),start,folder/name,args.ffmpeg,manifest['video_fps'])
            video=decode(folder/name)
            if list(video[0].shape[1::-1])!=state['resolution']:raise ValueError('Unexpected video dimensions')
            reference=cv2.imread(str(folder/('target_rgb.png' if kind=='rgb_video' else 'target_whitebox.png')))
            psnr=cv2.PSNR(reference,video[0])
            if psnr<26:raise ValueError(f'First-frame alignment failed: {key} {psnr}')
            videos.append(video)
        save(folder/'target_frame_states.json',frames)
        cards=[]
        for i in (0,30,60,90,119):
            rgb=cv2.resize(videos[0][i],(384,216));proxy=cv2.resize(videos[1][i],(384,216))
            cv2.putText(rgb,f'{key} +{i}',(5,20),0,.45,(0,255,255),1)
            cards.append(np.vstack([rgb,proxy]))
        cv2.imwrite(str(folder/'target_video_review.jpg'),np.hstack(cards),[cv2.IMWRITE_JPEG_QUALITY,92])
        sample.update(training_status='TARGET_VIDEO_AWAITING_VISUAL_REVIEW',frame_count=120,fps=manifest['video_fps'],
            target_frame_states='target_frame_states.json')
        sample['conditions']['target_whitebox']='target_whitebox.mp4'
        sample['supervision']['target_rgb']='target_rgb.mp4'
        # Frame-pair conditions remain accessible without mislabeling video supervision.
        sample['frame_pair']={'target_whitebox':'target_whitebox.png','target_rgb':'target_rgb.png'}
        save(folder/'sample.json',sample)
        ledger.append({'sample_id':key,'status':sample['training_status']})
        print('VIDEO_VERIFIED',key,flush=True)
    save(args.root/'training_preparation.json',ledger)
    shutil.copy2('configs/gta_scene_review_20261019.json',dataset/'frame_visual_review.json')
    print('PREPARATION_COMPLETE',len(ledger),flush=True)


if __name__=='__main__':main()
