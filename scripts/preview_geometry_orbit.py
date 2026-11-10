"""Inspect a candidate scene around its player; no world-state commits or H3."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
import av
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.pi.world_tools import read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    scene = read(a.scene)
    player = next(e for e in scene['entities'] if e.get('behavior', {}).get('type') == 'player')
    pivot = np.asarray(player['position'], float)
    a.output.mkdir(parents=True, exist_ok=False)
    camera = scene['camera']
    calibration = camera['intrinsics']
    width, height = int(calibration['width'])*4, int(calibration['height'])*4
    frames = []
    for index in range(1, 97):
        angle = 2*math.pi*index/96
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        view = deepcopy(camera)
        for key in ('position', 'target'):
            view[key] = (pivot+rotation@(np.asarray(camera[key])-pivot)).tolist()
        if 'up' in camera:
            view['up'] = (rotation@np.asarray(camera['up'])).tolist()
        frames.append(dict(time=index/24, camera=view, entities=deepcopy(scene['entities'])))
    timeline = dict(fps=24, frames=frames, scope='Candidate geometry inspection only; no accepted motion')
    for name, value in [('scene', scene), ('timeline', timeline)]:
        (a.output/(name+'.json')).write_text(json.dumps(value), encoding='utf-8')
    subprocess.run([sys.executable, str(ROOT/'whitebox/godot/render.py'), '--scene', str(a.output/'scene.json'),
                    '--timeline', str(a.output/'timeline.json'), '--output', str(a.output/'frames'),
                    '--frames', 'all', '--width', str(width), '--height', str(height)], check=True)
    files = [a.output/'frames/camera.png', *sorted((a.output/'frames').glob('frame_*.png'))]
    if len(files) != 97:
        raise ValueError('Incomplete orbit render')
    with av.open(str(a.output/'orbit.mp4'), 'w') as out:
        stream = out.add_stream('libx264', rate=24)
        stream.width, stream.height, stream.pix_fmt = width//2, height//2, 'yuv420p'
        for path in files:
            with Image.open(path) as im:
                im = im.convert('RGB').resize((width//2, height//2), Image.Resampling.LANCZOS)
                for packet in stream.encode(av.VideoFrame.from_image(im)):
                    out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    with av.open(str(a.output/'orbit.mp4')) as video:
        if sum(1 for _ in video.decode(video=0)) != 97:
            raise ValueError('Encoded orbit missing frames')
    print(a.output/'orbit.mp4')


if __name__ == '__main__':
    main()
