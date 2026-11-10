"""Execute prepared prompt-only regressions without editing world state."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

try:
    from scripts.prepare_prompt_regression import cases, command_for
except ModuleNotFoundError:
    from prepare_prompt_regression import cases, command_for


def validated_cases(baseline, prepared, output):
    source = json.loads((prepared / 'source_changed.json').read_text(encoding='utf-8'))['command']
    expected = cases(baseline['prompt'], source[source.index('--prompt') + 1])
    records = []
    for name in ['baseline', 'remove_player_detail', 'remove_staff_detail',
                 'remove_tiger_detail', 'append_architecture']:
        directory = prepared / name
        record = json.loads((directory / 'prepared.json').read_text())
        prompt = (directory / 'prompt.txt').read_text(encoding='utf-8')
        command = record['command']
        original = baseline['command']
        if command_for(command, baseline['prompt'], original[original.index('--output') + 1]) != original:
            raise ValueError(name + ': non-prompt renderer argument changed')
        if command[command.index('--prompt') + 1] != prompt:
            raise ValueError(name + ': prompt and prepared command disagree')
        if prompt != expected[name]:
            raise ValueError(name + ': prompt does not implement the named ablation')
        if name == 'baseline' and prompt != baseline['prompt']:
            raise ValueError('Baseline prompt changed')
        records.append((name, command_for(command, prompt, output / name / 'render')))
    return records


def main(a):
    output = a.output.resolve()
    baseline = json.loads(a.baseline.read_text(encoding='utf-8'))
    records = validated_cases(baseline, a.prepared, output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(a.baseline, output / 'baseline.json')
    renderer = Path(baseline['command'][2])
    shutil.copy2(renderer, output / renderer.name)
    root = Path(__file__).resolve().parents[1]
    shutil.copy2(root / 'scripts/evaluate_camera_revisit.py', output / 'evaluate_camera_revisit.py')
    gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version',
                                       '--format=csv'], text=True)
    (output / 'environment.txt').write_text(gpu_info, encoding='utf-8')
    active, results = {}, {}
    try:
        while records or active:
            for gpu in range(4):
                if gpu in active or not records:
                    continue
                # The append case reuses baseline GPU 0 for a same-device comparison.
                if records[0][0] == 'append_architecture' and gpu != 0:
                    continue
                name, command = records.pop(0)
                directory = output / name
                directory.mkdir()
                for filename in ['prompt.txt', 'prompt.diff']:
                    shutil.copy2(a.prepared / name / filename, directory / filename)
                (directory / 'executed.json').write_text(json.dumps({
                    'command': command, 'gpu': gpu, 'world_commit': False,
                    'source_baseline': str(a.baseline.resolve()), 'case': name,
                }, indent=2), encoding='utf-8')
                log = (directory / 'generation.log').open('w')
                process = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                           stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = (name, process, log, time.monotonic())
                print('START', name, 'GPU', gpu, 'PID', process.pid, flush=True)
            for gpu, (name, process, log, started) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                result = {'returncode': code, 'seconds': time.monotonic() - started, 'gpu': gpu}
                (output / name / 'exit.json').write_text(json.dumps(result, indent=2))
                results[name] = result
                del active[gpu]
                print('END', name, json.dumps(result), flush=True)
            if active:
                time.sleep(2)
    finally:
        for name, process, log, _ in active.values():
            process.terminate()
            process.wait()
            log.close()
    (output / 'execution_summary.json').write_text(json.dumps(results, indent=2))
    return int(any(row['returncode'] for row in results.values()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['baseline', 'prepared', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    raise SystemExit(main(p.parse_args()))
