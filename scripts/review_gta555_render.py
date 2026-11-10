"""Create RGB/proxy comparisons using the renderer's explicit decode mapping."""
import argparse
import json
from pathlib import Path
import av
import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt', required=True, type=Path)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    receipt = json.loads(a.receipt.read_text())
    rows = [json.loads(line) for line in Path(receipt['frames']).open() if line.strip()]
    valid = [i for i, r in enumerate(rows) if r['status'] == 'estimated_alignment']
    if not valid:
        raise ValueError('No camera-valid frames to inspect')
    chosen = {valid[round(i * (len(valid) - 1) / 3)] for i in range(4)}
    cards, mappings = [], []
    rgb_path = a.source / receipt['rgb_relative']
    with av.open(str(rgb_path)) as rgb, av.open(receipt['proxy']) as proxy, av.open(str(a.output / 'rgb_whitebox.mp4'), 'w') as output:
        rgb_frames = iter(rgb.decode(video=0))
        rgb_index = -1
        rgb_frame = None
        fps = proxy.streams.video[0].average_rate
        stream = output.add_stream('libx264', rate=fps)
        stream.width, stream.height = 1280, 384
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '20', 'preset': 'fast'}
        count = 0
        for i, frame in enumerate(proxy.decode(video=0)):
            row = rows[i]
            target = row['rgb_decode_index']
            if target < rgb_index:
                raise ValueError('Non-monotonic RGB mapping')
            while rgb_index < target:
                rgb_frame = next(rgb_frames)
                rgb_index += 1
            left = cv2.resize(rgb_frame.to_ndarray(format='bgr24'), (640, 360))
            right = cv2.resize(frame.to_ndarray(format='bgr24'), (640, 360))
            panel = np.zeros((384, 1280, 3), dtype=np.uint8)
            panel[24:, :640] = left
            panel[24:, 640:] = right
            cv2.putText(panel, f'RGB decode {target} | {row["status"]}', (8, 17), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
            cv2.putText(panel, f'Rebuilt proxy frame {i}', (648, 17), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
            if i in chosen:
                cards.append(panel)
                mappings.append({'proxy_frame': i, 'rgb_decode_index': target, 'status': row['status']})
            for packet in stream.encode(av.VideoFrame.from_ndarray(panel, format='bgr24')):
                output.mux(packet)
            count += 1
        for packet in stream.encode():
            output.mux(packet)
    if count != len(rows):
        raise ValueError('Comparison frame count mismatch')
    cv2.imwrite(str(a.output / 'contact_sheet.jpg'), np.concatenate(cards, axis=0))
    (a.output / 'comparison.json').write_text(json.dumps({
        'frames': count, 'selected_frames': mappings, 'source_rgb': str(rgb_path),
        'source_proxy': receipt['proxy'], 'mapping': 'rgb_decode_index, sequential decode',
        'visual_review': 'PENDING'}, indent=2))


if __name__ == '__main__':
    main()
