"""Export synchronized diagnostic videos without changing their timing."""
import argparse
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np

from export_camera_comparison import panel


def export(videos, labels, output):
    if len(videos) != len(labels) or not 2 <= len(videos) <= 4:
        raise ValueError('Provide two to four videos and one label per video')
    if output.exists():
        raise FileExistsError(output)
    captures = [cv2.VideoCapture(str(path)) for path in videos]
    try:
        counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in captures]
        rates = [c.get(cv2.CAP_PROP_FPS) for c in captures]
        if not all(c.isOpened() for c in captures) or len(set(counts)) != 1 or counts[0] < 1:
            raise ValueError('Readable videos with identical frame counts required')
        if any(abs(rate - 24) > 1e-5 for rate in rates):
            raise ValueError('Expected aligned 24 fps videos; no temporal resampling')
        rows = (len(videos) + 1) // 2
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
            intermediate = Path(temporary) / 'grid.mp4'
            writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*'mp4v'),
                                     24, (1280, rows * 512))
            if not writer.isOpened():
                raise RuntimeError('Video writer unavailable')
            try:
                for frame in range(counts[0]):
                    tiles = []
                    for capture, label in zip(captures, labels):
                        ok, image = capture.read()
                        if not ok:
                            raise ValueError('Unexpected end of video')
                        tiles.append(panel(image, f'{label} | {frame / 24:.2f}s'))
                    if len(tiles) % 2:
                        tiles.append(np.full_like(tiles[0], 245))
                    image = np.concatenate([np.concatenate(tiles[i:i+2], axis=1)
                                            for i in range(0, len(tiles), 2)], axis=0)
                    writer.write(image)
                    if frame in {0, 12, counts[0] // 2, counts[0] - 1}:
                        cv2.imwrite(str(output.with_name(output.stem + f'_{frame:03d}.jpg')), image)
            finally:
                writer.release()
            subprocess.run(['ffmpeg', '-v', 'error', '-n', '-i', str(intermediate),
                            '-c:v', 'libx264', '-crf', '19', '-pix_fmt', 'yuv420p',
                            '-movflags', '+faststart', str(output)], check=True)
    finally:
        for capture in captures:
            capture.release()
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, action='append', required=True)
    parser.add_argument('--label', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.video, args.label, args.output)
