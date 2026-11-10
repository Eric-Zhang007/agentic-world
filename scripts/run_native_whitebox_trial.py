"""Validate native whitebox video + appearance image without ControlNet."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_comfy_control import save_video


def main():
    out = ROOT / 'outputs/native_whitebox_reference_20261013'
    out.mkdir(exist_ok=False)
    live = ROOT / 'outputs/identity_repair_20261013/live/segment_00'
    source = ROOT / 'outputs/identity_repair_20261013/prompt_regression_run1/baseline/render/metadata.json'
    request = json.loads(source.read_text())['request']
    preview = out / 'proxy'
    with (out / 'proxy.log').open('w') as log:
        subprocess.run([sys.executable, str(ROOT / 'whitebox/godot/render.py'),
                        '--scene', str(live / 'plan/scene.json'), '--timeline', str(live / 'plan/timeline.json'),
                        '--output', str(preview), '--frames', 'all', '--width', '1024', '--height', '768'],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    files = [preview / 'camera.png', *sorted(preview.glob('frame_*.png'))]
    assert len(files) == request['frames'] == 124
    pixels = np.stack([np.asarray(Image.open(p).convert('RGB')) for p in files])
    assert pixels.shape == (124, 768, 1024, 3)
    reference = out / 'whitebox_reference.mp4'
    save_video(pixels.astype(np.float32)/255, reference)
    prompt = request['prompt'] + (
        '\n<Video 1> is a coarse whitebox animation of this same scene and time interval. '
        'Use its camera path, relative object placement, world-facing directions, motion timing '
        'and occlusion changes. <Picture 1> supplies the actual initial appearance, materials '
        'and visual identities. Render the whitebox motion in that established visual world; '
        'do not copy the proxy colors, primitive geometry or mannequin appearance. '
        'The video is a motion/layout reference, not a second scene or another character.')
    (out / 'prompt.txt').write_text(prompt, encoding='utf-8')
    cases = out / 'cases.json'
    cases.write_text(json.dumps([{'label': 'native_video', 'controls': [], 'strength': 0, 'cache_threshold': 0}]))
    cmd = ['/home/star/agentic-world-comfy/.venv/bin/python', '-u', str(ROOT / 'scripts/run_comfy_control.py')]
    for key in ['comfy', 'steps', 'frames', 'width', 'height', 'seed', 'kernel_backend', 'reference_image']:
        cmd.extend(['--' + key.replace('_', '-'), str(request[key])])
    cmd.extend(['--prompt', prompt, '--anchor-first', '--reference-video', str(reference),
                '--output', str(out / 'render'), '--cases', str(cases)])
    (out / 'experiment.json').write_text(json.dumps({'command': cmd, 'world_commit': False,
        'source_baseline': str(source), 'source_plan': str(live / 'plan'),
        'changed': ['native video reference added', 'explicit modality-role prompt appended', 'ControlNet disabled'],
        'scope': 'Interface validation, not a prompt-identical depth ablation'}, indent=2))
    with (out / 'generation.log').open('w') as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_ENDPOINT='https://hf-mirror.com'))
    (out / 'exit.json').write_text(json.dumps({'returncode': result.returncode}))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
