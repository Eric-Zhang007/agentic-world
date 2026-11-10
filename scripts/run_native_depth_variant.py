"""Run a depth variant with the exact native-video trial references and prompt."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strength', type=float, choices=[0.5, 1.0], required=True)
    parser.add_argument('--gpu', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    trial = root / 'outputs/native_whitebox_reference_20261013'
    source = json.loads((trial / 'experiment.json').read_text())
    baseline = json.loads(Path(source['source_baseline']).read_text())['request']
    output = trial / ('depth_half' if args.strength == 0.5 else 'depth_full')
    output.mkdir(exist_ok=False)
    cases = output / 'cases.json'
    cases.write_text(json.dumps([{'label': 'native_video', 'strength': args.strength,
        'controls': [{'path': baseline['control_video'], 'strength': args.strength}],
        'cache_threshold': 0}], indent=2))
    command = list(source['command'])
    for flag, value in [('--output', str(output / 'render')), ('--cases', str(cases))]:
        command[command.index(flag)+1] = value
    command.extend(['--control-video', baseline['control_video']])
    (output / 'experiment.json').write_text(json.dumps({'command': command,
        'gpu': args.gpu, 'strength': args.strength, 'world_commit': False,
        'source': str(trial / 'experiment.json'),
        'note': 'Identical native video, first image, prompt, seed and sampling parameters; different GPU.'}, indent=2))
    with (output / 'generation.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, HF_ENDPOINT='https://hf-mirror.com'))
    (output / 'exit.json').write_text(json.dumps({'returncode': result.returncode}))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
