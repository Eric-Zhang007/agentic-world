"""Compare real videos at matching presentation times; no padding or synthesis.

Explicit: --video history=path.mp4 --video nohistory=path.mp4 --output new_dir
Discovery: --root experiment_dir --output new_dir
Discovery groups videos by outbound/return and history/nohistory path tokens.
Ambiguous groups require explicit --video arguments. Exit 2 means no comparison.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import re
import sys

import av
import numpy as np
from PIL import Image, ImageDraw


def discover(root):
    groups = {}
    for path in sorted(root.rglob('*.mp4')):
        relative = path.relative_to(root).as_posix().lower()
        tokens = set(re.findall(r'[a-z0-9]+', relative))
        phase = next((p for p in ('outbound', 'return') if p in tokens), None)
        branch = 'nohistory' if ('nohistory' in tokens or 'no_history' in relative) else 'history' if 'history' in tokens else None
        if phase is None and branch is None:
            continue
        label = '/'.join(x for x in (phase, branch) if x)
        groups.setdefault(label, []).append(path)
    if any(len(paths) != 1 for paths in groups.values()):
        raise ValueError('Ambiguous discovered videos; specify --video label=path: '+str(groups))
    return [(label, paths[0]) for label, paths in groups.items()]


def read_video(path, width):
    images, timestamps, durations = [], [], []
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError(f'{path}: expected exactly one video stream')
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        rate = stream.average_rate
        dimensions = set()
        for frame in container.decode(stream):
            if frame.is_corrupt or frame.pts is None or frame.time_base is None:
                raise ValueError(f'{path}: corrupt frame or missing presentation timestamp')
            timestamps.append(float(frame.pts*frame.time_base))
            duration = getattr(frame, 'duration', None)
            durations.append(float(duration*frame.time_base) if duration else None)
            dimensions.add((frame.width, frame.height))
            height = max(2, round(width*frame.height/frame.width))
            images.append(frame.to_image().convert('RGB').resize((width, height), Image.Resampling.LANCZOS))
    if len(images) < 2 or len(dimensions) != 1:
        raise ValueError(f'{path}: need at least two frames with constant dimensions')
    pts = np.asarray(timestamps)
    if np.any(np.diff(pts) <= 0):
        raise ValueError(f'{path}: timestamps are not strictly increasing')
    relative = pts-pts[0]
    last_duration = durations[-1] or float(np.median(np.diff(relative)))
    return dict(path=str(path.resolve()), images=images, times=relative,
                first_pts=float(pts[0]), last_pts=float(pts[-1]),
                duration=float(relative[-1]+last_duration), last_duration=last_duration,
                duration_source='decoded_frame_duration' if durations[-1] else 'median_pts_interval',
                dimensions=list(next(iter(dimensions))), rate=rate)


def compare(videos, output, width=480, tolerance=.001):
    if len(videos) < 2:
        raise ValueError('Need at least two existing video branches; no comparison generated')
    if len({label for label, _ in videos}) != len(videos):
        raise ValueError('Video labels must be unique')
    if len({path.resolve() for _, path in videos}) != len(videos):
        raise ValueError('Cannot compare a video with itself as another branch')
    if output.exists():
        raise ValueError('Output already exists; choose a new directory')
    loaded = [read_video(path, width) for _, path in videos]
    reference = loaded[0]
    times = reference['times']
    aligned = []
    for value in loaded:
        if abs(value['duration']-reference['duration']) > tolerance or abs(value['times'][-1]-times[-1]) > tolerance:
            raise ValueError('Video durations/endpoints differ; refusing trimming, padding or time stretching')
        indices = np.searchsorted(value['times'], times)
        right = np.clip(indices, 0, len(value['times'])-1)
        left = np.clip(indices-1, 0, len(value['times'])-1)
        indices = np.where(abs(value['times'][left]-times) <= abs(value['times'][right]-times), left, right)
        error = abs(value['times'][indices]-times)
        if np.max(error) > tolerance or len(np.unique(indices)) != len(indices):
            raise ValueError('No one-to-one frames at matching real timestamps; refusing duplicated frames')
        aligned.append(indices)
        value['max_alignment_error_seconds'] = float(error.max())
    height = max(image['images'][0].height for image in loaded)
    panel_height = height+26
    if panel_height % 2:
        panel_height += 1
    canvas_width = width*len(videos)
    if canvas_width % 2:
        canvas_width += 1

    def compose(index):
        canvas = Image.new('RGB', (canvas_width, panel_height), '#161616')
        draw = ImageDraw.Draw(canvas)
        for column, ((label, _), value, indices) in enumerate(zip(videos, loaded, aligned)):
            image = value['images'][int(indices[index])]
            canvas.paste(image, (column*width, 26+(height-image.height)//2))
            draw.text((column*width+6, 6), f'{label[:35]} | t={times[index]:.3f}s', fill='white')
        return canvas

    output.mkdir(parents=True)
    temporary = output/'comparison.partial.mp4'
    time_base = Fraction(1, 90000)
    ticks = np.rint(times*90000).astype('int64')
    if np.any(np.diff(ticks) <= 0):
        raise ValueError('Timestamp resolution exceeds output time base')
    with av.open(str(temporary), 'w') as container:
        stream = container.add_stream('libx264', rate=reference['rate'] or Fraction(24))
        stream.width, stream.height = canvas_width, panel_height
        stream.pix_fmt = 'yuv420p'
        stream.time_base = time_base
        stream.codec_context.time_base = time_base
        stream.codec_context.thread_count = 2
        stream.options = {'crf': '20', 'preset': 'fast'}
        for index, tick in enumerate(ticks):
            frame = av.VideoFrame.from_image(compose(index))
            frame.pts, frame.time_base = int(tick), time_base
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    # Verify the encoded comparison, not just encoder completion.
    with av.open(str(temporary)) as container:
        actual = [float(f.pts*f.time_base) for f in container.decode(video=0)]
    if len(actual) != len(times) or not np.allclose(np.asarray(actual)-actual[0], times, atol=2/90000, rtol=0):
        raise ValueError('Encoded comparison did not preserve reference timestamps')
    temporary.replace(output/'comparison.mp4')
    selected = [0, int(np.argmin(abs(times-times[-1]/2))), len(times)-1]
    contact = Image.new('RGB', (canvas_width, panel_height*3))
    for row, index in enumerate(selected):
        contact.paste(compose(index), (0, row*panel_height))
    contact.save(output/'contact.jpg', quality=92)
    report = dict(scope='Timestamp-aligned visual comparison only; not proof of memory benefit',
                  alignment='elapsed presentation time relative to each first decoded PTS; no stretching/padding',
                  tolerance_seconds=tolerance, frames=len(times), sample_times_seconds=times.tolist(),
                  contact_indices=selected, contact_times_seconds=[float(times[i]) for i in selected],
                  sources=[dict(label=label, **{k:v for k,v in value.items() if k not in ('images', 'times', 'rate')},
                                average_rate=str(value['rate']), selected_frame_indices=indices.tolist())
                           for (label, _), value, indices in zip(videos, loaded, aligned)],
                  comparison=str((output/'comparison.mp4').resolve()), contact=str((output/'contact.jpg').resolve()))
    (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--video', action='append', default=[], metavar='LABEL=PATH')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width', type=int, default=480, help='Width per video panel')
    args = parser.parse_args()
    try:
        if args.width < 128:
            raise ValueError('width must be at least 128')
        if args.video:
            videos = []
            for value in args.video:
                label, separator, path = value.partition('=')
                if not separator or not label or not path:
                    raise ValueError('Expected --video label=path')
                videos.append((label, Path(path)))
        elif args.root and args.root.is_dir():
            videos = discover(args.root)
        else:
            raise ValueError('Supply --video arguments or an existing --root')
        report = compare(videos, args.output, args.width)
        print(json.dumps({k:report[k] for k in ('frames', 'contact_times_seconds', 'comparison', 'contact')}, indent=2))
    except (ValueError, OSError, av.FFmpegError) as exc:
        print(f'NO_COMPARISON: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
