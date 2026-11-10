"""Track annotated static anchors through images without editing world state."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from perception.motion_evidence import track_points


def track_sequence(initial_image, annotations, frames, output):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    previous = cv2.imread(str(initial_image),cv2.IMREAD_GRAYSCALE)
    if previous is None:raise ValueError('Cannot read initial image')
    if previous.shape != (annotations['height'],annotations['width']):
        raise ValueError('Initial annotation dimensions differ')
    rows = deepcopy(annotations['landmarks'])
    points = np.asarray([r['pixel'] for r in rows],float)
    alive = np.ones(len(rows),dtype=bool)
    reports=[]
    for index,frame in enumerate(frames):
        current=cv2.imread(str(frame['image']),cv2.IMREAD_GRAYSCALE)
        if current is None:raise ValueError('Cannot read target image')
        indices=np.flatnonzero(alive)
        started=time.perf_counter()
        tracked=track_points(previous,current,points[indices])
        elapsed=time.perf_counter()-started
        surviving=indices[tracked['valid']]
        alive[:]=False
        alive[surviving]=True
        points[surviving]=tracked['points'][tracked['valid']]
        accepted=[]
        for i in surviving:
            row=deepcopy(rows[i]);row['pixel']=points[i].tolist();accepted.append(row)
        report=dict(image=str(frame['image']),timestamp_sec=frame['timestamp_sec'],
            static_landmarks=accepted,lost_ids=[r['id'] for i,r in enumerate(rows) if not alive[i]],
            width=annotations['width'],height=annotations['height'],seconds=elapsed,
            status='TRACKED_2D_CANDIDATES' if len(accepted)>=3 else 'INSUFFICIENT_ANCHORS',
            world_committed=False,scope='Roundtrip-checked image tracks; not occlusion or semantic identity proof.')
        errors=[]
        reference={r['id']:r['pixel'] for r in frame.get('static_landmarks',[])}
        for row in accepted:
            if row['id'] in reference:
                errors.append(dict(id=row['id'],manual_error_px=float(np.linalg.norm(np.asarray(row['pixel'])-reference[row['id']]))))
        report['manual_comparison']=errors
        reports.append(report)
        display=cv2.cvtColor(current,cv2.COLOR_GRAY2BGR)
        for row in accepted:
            p=tuple(np.rint(row['pixel']).astype(int))
            cv2.circle(display,p,5,(0,255,0),2)
            cv2.putText(display,row['id'],(p[0]+8,p[1]),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,0),1)
        cv2.imwrite(str(output/f'{index+1:02d}.jpg'),display)
        previous=current
    (output/'report.json').write_text(json.dumps(reports,indent=2))
    return reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('image','annotations','frames','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    report=track_sequence(a.image,json.loads(a.annotations.read_text()),json.loads(a.frames.read_text())['frames'],a.output)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
