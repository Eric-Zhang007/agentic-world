"""Inspect declared background ROIs; image motion is not camera reconstruction."""
import argparse
import json
from pathlib import Path
import sys

import av
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception.motion_evidence import pair_motion


def inspect(video, output, regions, width=640):
    if not regions or width < 32:
        raise ValueError('Background ROIs and a usable analysis width are required')
    for box in regions:
        if (len(box) != 4 or not np.isfinite(box).all() or
                not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1):
            raise ValueError('ROIs must be normalized xyxy boxes')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, samples = [], []
    previous = None
    accumulated = np.eye(3)
    complete = True
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 24)
        origin = None
        next_sample = 0.
        for index, frame in enumerate(container.decode(video=0)):
            timestamp = float(frame.time) if frame.time is not None else index/fps
            if origin is None:
                origin = timestamp
            timestamp -= origin
            rgb = frame.to_ndarray(format='bgr24')
            image = cv2.resize(rgb, (width, round(rgb.shape[0]*width/rgb.shape[1])))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mask = np.zeros(gray.shape, np.uint8)
            for box in regions:
                x0,y0,x1,y1 = np.rint(np.array(box)*[width,len(gray),width,len(gray)]).astype(int)
                mask[y0:y1,x0:x1] = 255
                cv2.rectangle(image, (x0,y0), (x1,y1), (0,255,255), 1)
            if previous is not None:
                measured = pair_motion(previous, gray, mask)
                if measured['status'] == 'MEASURED':
                    matrix = np.eye(3)
                    matrix[:2] = measured['dominant_affine']
                    accumulated = matrix @ accumulated
                else:
                    complete = False
                rows.append({**measured, 'frame_id': index, 'time_sec': timestamp,
                             'cumulative_affine': accumulated[:2].tolist() if complete else None})
            previous = gray
            if timestamp >= next_sample:
                path = output/f'roi_{index:04d}.jpg'
                cv2.imwrite(str(path), image)
                samples.append({'time_sec':timestamp,'path':str(path.resolve())})
                next_sample = timestamp+.75
    if previous is None:
        raise ValueError('Video is empty')
    measured = [r for r in rows if r['status']=='MEASURED']
    summary = {'video':str(Path(video).resolve()), 'regions':regions, 'analysis_width':width,
               'pair_count':len(rows), 'measured_pairs':len(measured),
               'cumulative_affine':accumulated[:2].tolist() if complete and rows else None,
               'median_inlier_fraction':float(np.median([r['dominant_fit_fraction'] for r in measured])) if measured else None,
               'median_residual_p90_px':float(np.median([r['residual_p90_px'] for r in measured])) if measured else None,
               'provenance':'MEASURED_IMAGE_FLOW_IN_DECLARED_BACKGROUND_REGIONS',
               'limitations':['ROIs chosen by caller must remain background throughout the clip.',
                              'Accumulated pairwise 2D fits drift and do not recover 3D extrinsics.',
                              'Coherent image flow does not prove rigid geometry or actor stationarity.'],
               'samples':samples}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (output/'pairs.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--regions', required=True, help='JSON normalized background xyxy boxes')
    args = parser.parse_args()
    print(json.dumps(inspect(args.video,args.output,json.loads(args.regions)),indent=2))
