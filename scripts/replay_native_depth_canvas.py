"""Replay one recorded experiment changing only the depth render canvas."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_video_reconcile_trial import conditions, render_proxy


def main(a):
    original = json.loads(a.record.read_text())['command']
    def value(flag):
        return original[original.index(flag)+1]
    if (value('--width'), value('--height')) != ('1024', '768'):
        raise ValueError('This replay requires a recorded 1024x768 canvas')
    if '--prefix-video' in original:
        raise ValueError('This isolated replay is boundary-only')
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    conditions(a.plan, Path(value('--reference-image')), output/'conditions', {}, 'depth')
    render_proxy(a.plan, output/'preview', width=1024, height=768)
    command = list(original)
    command[command.index('--control-video')+1] = str(output/'conditions/depth.npz')
    command[command.index('--output')+1] = str(output/'render')
    changes = [{'index': i, 'before': old, 'after': new}
               for i, (old, new) in enumerate(zip(original, command)) if old != new]
    if len(changes) != 2:
        raise ValueError('Expected only output directory and depth input changes')
    (output/'experiment.json').write_text(json.dumps({
        'source_record': str(a.record.resolve()), 'plan': str(a.plan.resolve()),
        'command': command, 'changes': changes, 'world_commit': False,
        'changed': 'Render depth natively at 1024x768 instead of stretching 960x640',
        'unchanged': ['world geometry', 'camera parameters', 'prompt', 'RGB reference',
                      'seed', 'strength', 'sampling steps', 'cache'],
    }, indent=2), encoding='utf-8')
    started = time.monotonic()
    with (output/'generation.log').open('w') as log:
        result = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu)),
                                stdout=log, stderr=subprocess.STDOUT)
    (output/'exit.json').write_text(json.dumps({'returncode': result.returncode,
        'seconds': time.monotonic()-started}, indent=2))
    return result.returncode


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('record', 'plan', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    raise SystemExit(main(p.parse_args()))
