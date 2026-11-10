"""Decode a completed replay, verify frame accounting, and export a contact sheet."""
import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw


def inspect(root):
    run = json.loads((root / 'run.json').read_text())
    timings = [json.loads(line) for line in (root / 'timings.jsonl').read_text().splitlines()]
    completion = json.loads((root / 'completion.json').read_text())
    expected = sum(row['output_frames'] for row in timings) + completion['flush_frames']
    reference_frames = sum(row['reference_frames'] for row in timings)
    samples = set(np.linspace(0, max(0, expected - 1), 8).astype(int).tolist())
    tiles, means, deviations, motion = [], [], [], []
    previous = None
    with av.open(str(root / 'generated.mp4')) as video:
        stream = video.streams.video[0]
        fps, width, height = float(stream.average_rate), stream.width, stream.height
        for index, frame in enumerate(video.decode(video=0)):
            rgb = frame.to_ndarray(format='rgb24')
            means.append(float(rgb.mean()))
            deviations.append(float(rgb.std()))
            small = np.asarray(frame.to_image().resize((168, 96))).astype(np.float32)
            if previous is not None:
                motion.append(float(np.abs(small - previous).mean()))
            previous = small
            if index in samples:
                tile = Image.new('RGB', (336, 216), 'white')
                tile.paste(frame.to_image().resize((336, 192)), (0, 24))
                ImageDraw.Draw(tile).text((8, 6), f'frame {index} / {index/fps:.3f}s', fill='black')
                tiles.append(tile)
    report = dict(rounds=len(timings), requested_rounds=run['rounds'],
                  frames=len(means), expected_frames=expected, reference_frames=reference_frames, fps=fps,
                  width=width, height=height, duration_sec=len(means)/fps,
                  mean_pixel=float(np.mean(means)), minimum_frame_std=min(deviations, default=0),
                  mean_frame_change=float(np.mean(motion)) if motion else 0)
    report['complete'] = (len(timings) == run['rounds'] == completion['rounds']
                          and len(means) == expected == reference_frames and expected > 0)
    (root / 'inspection.json').write_text(json.dumps(report, indent=2))
    if tiles:
        sheet = Image.new('RGB', (336 * 4, 216 * ((len(tiles) + 3) // 4)), 'white')
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % 4) * 336, (i // 4) * 216))
        sheet.save(root / 'contact_sheet.jpg')
    print(json.dumps(report, indent=2))
    if not report['complete']:
        raise SystemExit('Incomplete replay: inspect inference logs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    inspect(parser.parse_args().output)
