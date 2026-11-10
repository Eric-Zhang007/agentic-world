"""Pair archived planned previews, actual controls and video without rerendering."""
import json
from pathlib import Path
import subprocess
import tempfile
import cv2
import numpy as np
from export_camera_comparison import panel

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/feishu_20261013/whitebox_pairs'
BASE=ROOT/'outputs'


def preview(directory):
    return [directory/'camera.png']+sorted(directory.glob('frame_*.png'))


def render(name, segments):
    target=OUT/(name+'.mp4')
    if target.exists():raise FileExistsError(target)
    records=[];total=0
    with tempfile.TemporaryDirectory(dir=OUT) as temporary:
        raw=Path(temporary)/'raw.mp4'
        writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),24,(1920,512))
        if not writer.isOpened():raise RuntimeError('Writer unavailable')
        try:
            for number,spec in enumerate(segments):
                segment=BASE/spec['segment']
                movie=BASE/spec['video']
                planned=spec.get('planned','planned_preview')
                cap=cv2.VideoCapture(str(movie))
                fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if not cap.isOpened() or fps!=24:raise ValueError(movie)
                offset=spec.get('offset',0)
                if offset:
                    alignment=json.loads((segment/'continuation.alignment.json').read_text())
                    if offset!=alignment['trim_start'] or count!=alignment['continuation_frames']:
                        raise ValueError('Continuation alignment mismatch')
                arrays=spec.get('arrays',False)
                if arrays:
                    with np.load(segment/'controls/gray.npz') as f:gray=f['rgb']
                    with np.load(segment/'controls/layout.npz') as f:control_array=f['rgb']
                    if len(gray)!=count or len(control_array)!=count:raise ValueError('Early control count')
                    paths=None;controls=None
                else:
                    paths=preview(segment/planned)
                    if len(paths)!=count or not all(p.is_file() for p in paths):
                        raise ValueError(f'Preview mismatch {segment}: {len(paths)} vs {count}')
                    controls=cv2.VideoCapture(str(segment/'render/control_input.mp4'))
                    if not controls.isOpened() or int(controls.get(cv2.CAP_PROP_FRAME_COUNT))!=count+offset:
                        raise ValueError('Control count mismatch')
                    controls.set(cv2.CAP_PROP_POS_FRAMES,offset)
                record={**spec,'frames':count,'fps':fps,'start_output_frame':total,
                        'proxy_source':str(segment/('controls/gray.npz' if arrays else planned)),
                        'control_source':str(segment/('controls/layout.npz' if arrays else 'render/control_input.mp4')),
                        'video_source':str(movie),'control_trim_start':offset}
                records.append(record)
                try:
                    for index in range(count):
                        ok,actual=cap.read()
                        if not ok:raise ValueError('Video decode failed')
                        if arrays:
                            proxy=cv2.cvtColor(gray[index],cv2.COLOR_RGB2BGR)
                            control=cv2.cvtColor(control_array[index],cv2.COLOR_RGB2BGR)
                        else:
                            proxy=cv2.imread(str(paths[index]));ok,control=controls.read()
                            if proxy is None or not ok:raise ValueError('Preview/control decode failed')
                        time=index/24
                        label='CORRECTED interpretation' if planned=='corrected_preview' else 'PLAN whitebox'
                        state=spec.get('state','DIAGNOSTIC')
                        frame=np.concatenate([
                            panel(proxy,f'{label} | part {number+1} {time:.2f}s'),
                            panel(control,('ACTUAL Layout input' if arrays else 'ACTUAL depth input')+f' | {time:.2f}s'),
                            panel(actual,f'H3 | {state} | {time:.2f}s')],axis=1)
                        writer.write(frame)
                        if index in {0,count//2,count-1}:
                            cv2.imwrite(str(OUT/f'{name}_part{number+1}_{index:03d}.jpg'),frame)
                    total+=count
                finally:
                    cap.release()
                    if controls is not None:controls.release()
        finally:writer.release()
        subprocess.run(['ffmpeg','-v','error','-n','-i',str(raw),'-c:v','libx264','-crf','19',
                        '-pix_fmt','yuv420p','-movflags','+faststart',str(target)],check=True)
    verify=cv2.VideoCapture(str(target));decoded=0
    while True:
        ok,_=verify.read()
        if not ok:break
        decoded+=1
    verify.release()
    if decoded!=total:raise ValueError('Output truncated')
    result={'name':name,'output':str(target),'frames':total,'fps':24,'seconds':total/24,
            'segments':records,'verification':'All output frames decoded; no resampling or padding',
            'scope':'Archived preview frames, not newly reconstructed geometry. Corrected preview is an accepted interpretation, not measurement.'}
    (OUT/(name+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(name,total,'frames',flush=True)
    return result


def segment(root,index=0,**kwargs):
    path=f'{root}/segment_{index:02d}'
    return {'segment':path,'video':path+'/render/controlled.mp4',**kwargs}


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    cases={
        'V01_early_11s':[
            {'segment':f'action_wukong_outline_20261008/segment_{i:03d}',
             'video':f'action_wukong_outline_20261008/segment_{i:03d}/render/controlled.mp4',
             'arrays':True,'state':'historical ACCEPT'} for i in range(3)],
        'V02_memory_hold':[segment('appearance_memory_20261012',state='historical ACCEPT')],
        'V03_small_step':[segment('refined_rollouts_20261012',state='historical ACCEPT')],
        'V04_posture_plan':[segment('posture_live_20261010',i,state='historical ACCEPT') for i in range(2)],
        'V04_posture_corrected':[segment('posture_live_20261010',i,planned='corrected_preview',state='historical ACCEPT') for i in range(2)],
        'V05_continuous_attempt':[segment('continuous_motion_20261012',state='partial ACCEPT'),
            {**segment('continuous_motion_20261012',1,state='REJECTED'),'video':'continuous_motion_20261012/segment_01/continuation.mp4','offset':21}],
        'V06_follow_camera':[segment('natural_walk_dolly_20261012',state='REJECTED')],
    }
    for name,case in [('V07_orbit_baseline','baseline'),('V08_remove_player','remove_player_detail'),
                      ('V08_remove_staff','remove_staff_detail'),('V08_remove_tiger','remove_tiger_detail'),
                      ('V09_append_architecture','append_architecture')]:
        cases[name]=[{'segment':'identity_repair_20261013/live/segment_00',
                      'video':f'identity_repair_20261013/prompt_regression_run1/{case}/render/controlled.mp4',
                      'state':'NOT appearance-accepted'}]
    results=[render(name,specs) for name,specs in cases.items()]
    (OUT/'manifest.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':main()
