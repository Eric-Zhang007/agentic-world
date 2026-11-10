"""Replay the frozen orbit with no control and half-strength depth."""
import json
import os
from pathlib import Path
import subprocess


def main():
    root = Path('/home/star/agentic_world')
    source = root / 'outputs/identity_repair_20261013/prompt_regression_run1/baseline/render/metadata.json'
    request = json.loads(source.read_text())['request']
    output = root / 'outputs/depth_strength_ablation_20261013'
    output.mkdir(exist_ok=False)
    cases = [
        {'label': 'no_depth', 'strength': 0.0, 'controls': [], 'cache_threshold': 0.0},
        {'label': 'depth_half', 'strength': 0.5, 'controls': [{'strength': 0.5}], 'cache_threshold': 0.0},
    ]
    case_path = output / 'cases.json'
    case_path.write_text(json.dumps(cases, indent=2))
    command = ['/home/star/agentic-world-comfy/.venv/bin/python', '-u',
               str(root / 'scripts/run_comfy_control.py')]
    for key in ['comfy', 'prompt', 'steps', 'frames', 'width', 'height', 'seed',
                'kernel_backend', 'reference_image', 'control_video', 'control_end', 'cache_threshold']:
        command.extend(['--' + key.replace('_', '-'), str(request[key])])
    assert request['anchor_first'] and not request['last_frame'] and not request['prefix_video']
    assert not request['appearance_image'] and not request['cases']
    command.extend(['--anchor-first', '--output', str(output / 'render'), '--cases', str(case_path)])
    (output / 'experiment.json').write_text(json.dumps({
        'source_metadata': str(source), 'command': command, 'cases': cases,
        'world_commit': False, 'gpu': 0,
        'note': 'No-depth retains input decoding for audit but applies zero control streams.'
    }, indent=2))
    with (output / 'generation.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_ENDPOINT='https://hf-mirror.com'))
    (output / 'exit.json').write_text(json.dumps({'returncode': result.returncode}))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
