"""Compare actual native video input, generated output, and no-video control."""
from pathlib import Path
import argparse
import json
import subprocess

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth-variants', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / 'outputs/native_whitebox_reference_20261013'
    paths = [out / 'whitebox_reference.mp4', out / 'native_video.mp4',
             root / 'outputs/depth_strength_ablation_20261013/no_depth.mp4']
    labels = ['Native whitebox reference', 'Video + image; depth OFF', 'Earlier image only; depth OFF']
    name = 'native_video_comparison'
    if args.depth_variants:
        paths = [out/'whitebox_reference.mp4', out/'native_video.mp4',
                 out/'depth_half/native_video.mp4', out/'depth_full/native_video.mp4']
        labels = ['Native whitebox reference', 'Native video; depth OFF',
                  'Native video; depth 0.5', 'Native video; depth 1.0']
        name = 'native_depth_comparison'
    columns = 2 if args.depth_variants else 3
    rows = 2 if args.depth_variants else 1
    caps = [cv2.VideoCapture(str(p)) for p in paths]
    assert all(c.isOpened() and int(c.get(cv2.CAP_PROP_FRAME_COUNT)) == 124 for c in caps)
    work = out / (name+'_work.mp4')
    video = out / (name+'.mp4')
    writer = cv2.VideoWriter(str(work), cv2.VideoWriter_fourcc(*'mp4v'), 24, (columns*512, rows*424))
    assert writer.isOpened()
    sheets = []
    for i in range(124):
        canvas = np.zeros((rows*424, columns*512, 3), np.uint8)
        for j, cap in enumerate(caps):
            ok, frame = cap.read()
            assert ok
            x, y = (j % columns)*512, (j // columns)*424
            canvas[y+40:y+424, x:x+512] = cv2.resize(frame, (512, 384))
            cv2.putText(canvas, labels[j]+f' | {i/24:.2f}s', (x+6, y+24),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(canvas)
        if i in [0, 12, 31, 62, 93, 123]:
            cv2.imwrite(str(out/f'{name}_{i:03d}.jpg'), canvas)
            sheets.append(canvas)
    writer.release()
    for cap in caps:
        cap.release()
    subprocess.run(['ffmpeg', '-v', 'error', '-n', '-i', str(work), '-c:v', 'libx264',
                    '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)], check=True)
    work.unlink()
    cv2.imwrite(str(out/(name+'_sheet.jpg')), np.concatenate(sheets, axis=0))
    cap = cv2.VideoCapture(str(video))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    assert n == 124
    (out/(name+'.json')).write_text(json.dumps({'sources': [str(p) for p in paths],
        'decoded_frames': n, 'note': ('Same prompt and native reference; depth strength differs, GPUs 0/1/2.'
        if args.depth_variants else 'Native video case also appends modality-role prompt; not single-variable comparison.')}, indent=2))
    print(video)


if __name__ == '__main__':
    main()
