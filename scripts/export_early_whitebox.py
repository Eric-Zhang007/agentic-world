"""Export the saved three-phase 11.25s coarse whitebox, without rebuilding it."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    records = []
    for index in range(3):
        segment = a.episode / f'segment_{index:03d}'
        controls = json.loads((segment/'render/controls_request.json').read_text())
        manifest = json.loads((segment/'controls/manifest.json').read_text())
        records.append({'segment': str(segment.resolve()), 'frames': manifest['frames'],
                        'fps': manifest['fps'], 'actual_h3_controls': controls})
    assert [r['frames'] for r in records] == [73, 73, 124]
    assert all(r['fps'] == 24 for r in records)
    for kind in ['layout', 'gray', 'depth']:
        with tempfile.TemporaryDirectory(dir=a.output) as temp:
            intermediate = Path(temp)/'video.mp4'
            writer = None
            try:
                for record in records:
                    with np.load(Path(record['segment'])/f'controls/{kind}.npz') as archive:
                        frames = archive['rgb']
                    assert frames.shape == (record['frames'], 512, 640, 3)
                    if writer is None:
                        writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*'mp4v'),24,(640,512))
                        if not writer.isOpened():
                            raise RuntimeError('Cannot open video writer')
                    for frame in frames:
                        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            finally:
                if writer is not None:
                    writer.release()
            target = a.output/f'first_action_11s_{kind}.mp4'
            subprocess.run(['ffmpeg','-v','error','-n','-i',str(intermediate),'-c:v','libx264',
                            '-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],check=True)
            cap = cv2.VideoCapture(str(target))
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 270
            assert cap.get(cv2.CAP_PROP_FPS) == 24
            cap.set(cv2.CAP_PROP_POS_FRAMES,135)
            ok, image = cap.read()
            cap.release()
            assert ok
            cv2.imwrite(str(a.output/f'{kind}_middle.png'),image)
    (a.output/'source.json').write_text(json.dumps({'segments':records,'frames':270,'fps':24,
        'duration_seconds':11.25,'resolution':[640,512],
        'scope':'Archived planned coarse boxes, not video-reconciled geometry. Layout was actual H3 control; gray and depth were saved diagnostics. No articulated skeleton.'},indent=2))
    print(a.output)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
