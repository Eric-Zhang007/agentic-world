"""Decode each segment once for exact-index review frames, without random seeks."""
import json
from pathlib import Path
import threading
import subprocess
import os

import cv2
import numpy as np


_locks={}
_guard=threading.Lock()


def rgb_file_index(frame):
    """Segmented bundles retain episode-global decode indices for provenance."""
    value = frame.get('rgb_segment_frame_index', frame.get('rgb_decode_index'))
    if type(value) is not int or value < 0:
        raise ValueError('Missing or invalid RGB file-local frame index')
    return value


def source_media_path(record, manifest, kind):
    explicit = record.get('media_paths', {}).get(kind)
    if explicit:
        return Path(explicit)
    root = record.get('episode_root') or record['source_root']
    return Path(root) / manifest[kind]


def open_video(path):
    return cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])


def decode_count(path):
    command=['ffmpeg','-v','error','-xerror','-hwaccel','cuda','-i',str(path),'-map','0:v:0',
             '-an','-vf','setpts=N/(24*TB)','-fps_mode','passthrough','-progress','pipe:1','-nostats','-f','null','NUL']
    result=subprocess.run(command,capture_output=True,text=True,timeout=180)
    if result.returncode:raise ValueError('Full decode failed: '+result.stderr[-1500:])
    counts=[int(line.split('=',1)[1]) for line in result.stdout.splitlines() if line.startswith('frame=')]
    if not counts or 'progress=end' not in result.stdout:raise ValueError('No completed decode report')
    return counts[-1]


def cache_frames(window,release):
    key=window['episode_id']+'_'+window['agent_id']+'_'+window['segment']
    out=release/'review_frames'/key
    with _guard:lock=_locks.setdefault(key,threading.Lock())
    with lock:
        if (out/'complete.json').exists():return out
        out.mkdir(parents=True,exist_ok=True)
        manifest_path=Path(window['source_manifest'])
        manifest=json.loads(manifest_path.read_text())
        frames={r['proxy_frame_index']:r for r in map(json.loads,(manifest_path.parent/'frames.jsonl').read_text().splitlines())}
        wanted=set()
        for start in range(120,max(frames,default=-1)+1,120):
            wanted.add(start-120)
            wanted.update(start+int(i) for i in np.linspace(0,119,8,dtype=int))
        wanted.intersection_update(frames)
        for kind in ('rgb_video','proxy_video'):
            mapping={}
            for i in wanted:
                index=rgb_file_index(frames[i]) if kind=='rgb_video' else i
                if index is not None:mapping.setdefault(index,[]).append(i)
            cap=open_video(source_media_path(window,manifest,kind));count=0
            try:
                while True:
                    ok,image=cap.read()
                    if not ok:break
                    for i in mapping.get(count,[]):
                        ext='.png' if i%120==0 else '.jpg'
                        if not cv2.imwrite(str(out/f'{kind}_{i:06d}{ext}'),image):raise OSError('Frame cache write failed')
                    count+=1
            finally:cap.release()
            if mapping and max(mapping)>=count:raise ValueError('Review frame beyond decoded video')
        (out/'complete.json').write_text(json.dumps({'source':str(manifest_path),'decode_method':'sequential exact decoder indices'}))
    return out


def cached_frame(root,kind,index):
    ext='.png' if index%120==0 else '.jpg'
    image=cv2.imread(str(root/f'{kind}_{index:06d}{ext}'))
    if image is None:raise ValueError('Missing exact-index frame')
    return image


def paired_clips(window,release):
    """Encode each source stream once, with exact 120-frame segment boundaries."""
    if window['target_rgb_start']%120:
        return None
    key=window['episode_id']+'_'+window['agent_id']+'_'+window['segment']
    out=release/'paired_clips'/key
    with _guard:lock=_locks.setdefault('clips:'+key,threading.Lock())
    with lock:
        manifest=json.loads(Path(window['source_manifest']).read_text())
        if not (out/'complete.json').exists():
            out.mkdir(parents=True,exist_ok=True)
            summary=json.loads((release/'qualification'/key/'summary.json').read_text())
            for kind in ('rgb_video','proxy_video'):
                boundaries=','.join(str(i) for i in range(120,summary['decoded_frames'][kind],120))
                command=[os.environ.get('FFMPEG_BINARY','ffmpeg'),'-v','error','-y','-hwaccel','cuda','-threads','1',
                         '-i',str(source_media_path(window,manifest,kind)),
                         '-map','0:v:0','-an','-vf',f'settb=expr=1/{manifest["video_fps"]},setpts=N',
                         '-r',str(manifest['video_fps']),'-vsync','1','-c:v','h264_nvenc','-preset','slow','-rc','vbr','-cq','18','-b:v','0',
                         '-g','120','-force_key_frames','expr:gte(n,n_forced*120)','-bf','0',
                         '-f','segment','-segment_frames',boundaries,'-reset_timestamps','1',str(out/(kind+'_%04d.mp4'))]
                subprocess.run(command,check=True,timeout=300)
            (out/'complete.json').write_text(json.dumps({'frame_count_per_clip':120,'fps':manifest['video_fps'],
                'timeline':'constant decoder-index cadence; source timestamps remain in state records'}))
    return (out/f'rgb_video_{window["target_rgb_start"]//120:04d}.mp4',
            out/f'proxy_video_{window["target_start_frame"]//120:04d}.mp4')
