"""Build timestamp-aligned source/proxy/output inspection artifacts."""
import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw


def read_video(path):
    with av.open(str(path)) as video:
        return [frame.to_image() for frame in video.decode(video=0)]


def main(args):
    generated = read_video(args.output/'generated.mp4')
    proxy = read_video(args.inputs/'condition.mp4')
    manifest = json.loads((args.data/'manifest.json').read_text())
    reference = [Image.open(args.data/f['file']).convert('RGB') for f in manifest['frames']]
    times = np.array([f['time'] for f in manifest['frames']])
    selected = set(np.linspace(0, len(generated)-1, 6).astype(int).tolist())
    rows = []
    with av.open(str(args.output/'comparison.mp4'), 'w') as writer:
        stream = writer.add_stream('libx264', rate=24)
        stream.width, stream.height, stream.pix_fmt = 1152, 240, 'yuv420p'
        stream.options = {'crf': '18'}
        for index, frame in enumerate(generated):
            time = index/24
            original = reference[max(0, int(np.searchsorted(times, time, side='right')-1))]
            row = Image.new('RGB', (1152, 240), 'white')
            draw = ImageDraw.Draw(row)
            for col, (image, label) in enumerate(zip(
                    [original, proxy[index], frame],
                    ['Original GTA (5fps samples)', 'Existing approximate proxy', 'StreamRender zero-shot'])):
                draw.text((col*384+6, 6), f'{label} / {time:.2f}s', fill='black')
                row.paste(image.resize((384, 216)), (col*384, 24))
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(row), format='rgb24')):
                writer.mux(packet)
            if index in selected:
                rows.append(row)
                frame.save(args.output/f'eval_{index:03d}.png')
        for packet in stream.encode():
            writer.mux(packet)
    sheet = Image.new('RGB', (1152, 240*len(rows)), 'white')
    for i, row in enumerate(rows):
        sheet.paste(row, (0, i*240))
    sheet.save(args.output/'comparison_sheet.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'inputs', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
