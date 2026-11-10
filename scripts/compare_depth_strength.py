"""Build synchronized evidence for the frozen depth strength experiment."""
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'outputs/depth_strength_ablation_20261013'
    paths = [out / 'no_depth.mp4', out / 'depth_half.mp4',
             root / 'outputs/identity_repair_20261013/prompt_regression_run1/baseline/generated.mp4']
    labels = ['Depth OFF', 'Depth 0.5', 'Depth 1.0 (historical baseline)']
    caps = [cv2.VideoCapture(str(p)) for p in paths]
    assert all(c.isOpened() and int(c.get(cv2.CAP_PROP_FRAME_COUNT)) == 124
               and abs(c.get(cv2.CAP_PROP_FPS) - 24) < 1e-6 for c in caps)
    target = out / 'depth_comparison.mp4'
    temporary = out / 'depth_comparison_work.mp4'
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*'mp4v'), 24, (1536, 424))
    assert writer.isOpened()
    sheets = []
    for i in range(124):
        canvas = np.zeros((424, 1536, 3), np.uint8)
        for j, cap in enumerate(caps):
            ok, frame = cap.read()
            assert ok
            canvas[40:424, j*512:(j+1)*512] = cv2.resize(frame, (512, 384))
            cv2.putText(canvas, labels[j] + f' | {i/24:.2f}s', (j*512+8, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(canvas)
        if i in [0, 12, 31, 62, 93, 123]:
            cv2.imwrite(str(out / f'comparison_{i:03d}.jpg'), canvas)
            sheets.append(canvas)
    writer.release()
    subprocess.run(['ffmpeg', '-v', 'error', '-n', '-i', str(temporary), '-c:v', 'libx264',
                    '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)], check=True)
    temporary.unlink()
    for cap in caps:
        cap.release()
    cv2.imwrite(str(out / 'contact_sheet.jpg'), np.concatenate(sheets, axis=0))
    check = cv2.VideoCapture(str(target))
    decoded = 0
    while check.read()[0]:
        decoded += 1
    check.release()
    assert decoded == 124
    (out / 'comparison.json').write_text(json.dumps({'sources': [str(p) for p in paths],
        'labels': labels, 'decoded_frames': decoded, 'fps': 24, 'world_commit': False}, indent=2))
    print(target)


if __name__ == '__main__':
    main()
