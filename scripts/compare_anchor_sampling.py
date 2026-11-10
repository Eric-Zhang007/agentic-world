"""Compare endpoint and sequential tracking on the exact same local video."""
import argparse
import json
from pathlib import Path
import sys
import time
import av
import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from perception.motion_evidence import track_points


def run(video, annotations, output, scene=None, translation_bounds=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=annotations['landmarks']
    initial=np.asarray([r['pixel'] for r in rows],float)
    points=initial.copy();alive=np.ones(len(points),bool)
    first=previous=last=None
    trace=[];started=time.perf_counter()
    with av.open(str(video)) as container:
        stream=container.streams.video[0]
        for index,frame in enumerate(container.decode(stream)):
            current=frame.to_ndarray(format='gray')
            if current.shape!=(annotations['height'],annotations['width']):
                raise ValueError('Video and annotation dimensions differ')
            if previous is not None:
                indices=np.flatnonzero(alive)
                tracked=track_points(previous,current,points[indices])
                keep=indices[tracked['valid']]
                alive[:]=False;alive[keep]=True
                points[keep]=tracked['points'][tracked['valid']]
            else:
                first=current.copy()
            trace.append(dict(frame=index,time_sec=frame.time,
                              points=[p.tolist() if ok else None for p,ok in zip(points,alive)]))
            previous=last=current
    if first is None:raise ValueError('Empty video')
    dense_seconds=time.perf_counter()-started
    started=time.perf_counter()
    sparse=track_points(first,last,initial)
    sparse_seconds=time.perf_counter()-started
    dense=list(zip(points,alive))
    endpoint=list(zip(sparse['points'],sparse['valid']))
    report=dict(video=str(video),frame_count=len(trace),dense_seconds=dense_seconds,
        endpoint_tracking_seconds=sparse_seconds,world_committed=False,
        limitation='Track survival is not correct identity. No ground truth or automatic world update.',tracks=[])
    for row,(a,ok_a),(b,ok_b) in zip(rows,dense,endpoint):
        report['tracks'].append(dict(id=row['id'],dense=a.tolist() if ok_a else None,
            endpoint=b.tolist() if ok_b else None,
            disagreement_px=float(np.linalg.norm(a-b)) if ok_a and ok_b else None))
    for name,image,pairs in [('first',first,zip(initial,np.ones(len(initial),bool))),
                              ('dense_last',last,dense),('endpoint_last',last,endpoint)]:
        display=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        for row,(point,ok) in zip(rows,pairs):
            if not ok:continue
            x,y=np.rint(point).astype(int)
            cv2.circle(display,(x,y),5,(0,255,0),2)
            cv2.putText(display,row['id'],(x+8,y),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,0),1)
        cv2.imwrite(str(output/(name+'.jpg')),display)
    (output/'report.json').write_text(json.dumps(report,indent=2))
    (output/'tracks.json').write_text(json.dumps(trace,indent=2))
    if scene is not None:
        from whitebox.observation_alignment import align_observation
        selected=[dict(row,pixel=point.tolist()) for row,(point,ok) in zip(rows,dense) if ok]
        if len(selected)<3:
            report['camera_update']={'status':'INSUFFICIENT_ANCHORS'}
        else:
            observation=dict(width=annotations['width'],height=annotations['height'],
                timestamp_sec=trace[-1]['time_sec'],static_landmarks=selected,
                camera_translation_bounds=translation_bounds,actor_roots=[])
            candidate,alignment=align_observation(scene,observation)
            alignment['scope']='Camera-only candidate from tracked static anchors; actors not updated.'
            (output/'camera_candidate.json').write_text(json.dumps(candidate,indent=2))
            report['camera_update']=alignment
        (output/'report.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('video','annotations','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--scene',type=Path)
    p.add_argument('--translation-limit',type=float,default=2)
    a=p.parse_args()
    if not np.isfinite(a.translation_limit) or a.translation_limit<=0:
        p.error('Translation limit must be positive')
    scene=json.loads(a.scene.read_text()) if a.scene else None
    print(json.dumps(run(a.video,json.loads(a.annotations.read_text()),a.output,scene,
        [[-a.translation_limit,a.translation_limit]]*3),indent=2))
