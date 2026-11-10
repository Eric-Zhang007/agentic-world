"""Replay recorded H3 inputs with an explicit subset of appearance references."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time


def single_reference_prompt(prompt):
    marker = '<Picture 1> is the current accepted boundary.'
    body, separator, memory = prompt.partition(marker)
    if not separator:
        raise ValueError('Expected recorded appearance-memory prompt')
    body = re.sub(r'<Picture [2-9]> and <Picture [2-9]>', '<Picture 1>', body)
    body = re.sub(r'<Picture [2-9]>', '<Picture 1>', body)
    descriptions = []
    for line in memory.splitlines():
        match = re.fullmatch(r'<Picture [2-9]>: object (.*)', line)
        if match:
            descriptions.append(match.group(1))
    if not descriptions:
        raise ValueError('No recorded identity descriptions found')
    return (body.rstrip() + '\n' + marker +
            '\nIdentity descriptions from previously accepted video; retain current staging:\n' +
            '\n'.join(descriptions))


def main(a):
    original = json.loads(a.record.read_text())['command']
    prompt_index = original.index('--prompt') + 1
    prompt = (a.prompt_file.read_text(encoding='utf-8') if a.prompt_file
              else single_reference_prompt(original[prompt_index]))
    command = []
    i = 0
    reference_index = 0
    selected = set(a.appearance_indices)
    if any(index < 1 or index > original.count('--appearance-image') for index in selected):
        raise ValueError('Appearance index outside recorded reference list')
    while i < len(original):
        if original[i] == '--appearance-image':
            reference_index += 1
            if reference_index in selected:
                command.extend(original[i:i + 2])
            i += 2
            continue
        command.append(original[i])
        i += 1
    output = a.output.resolve()
    command[command.index('--output') + 1] = str(output / 'render')
    command[command.index('--prompt') + 1] = prompt
    control_change = None
    if a.strength is not None:
        if not math.isfinite(a.strength) or a.strength < 0:
            raise ValueError('Control strength must be finite and nonnegative')
        index = command.index('--strength') + 1
        control_change = {'before': float(command[index]), 'after': a.strength}
        command[index] = str(a.strength)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'prompt.txt').write_text(prompt, encoding='utf-8')
    (output / 'ablation.json').write_text(json.dumps({
        'source_record': str(a.record.resolve()), 'command': command,
        'appearance_indices': sorted(selected),
        'prompt_file': str(a.prompt_file.resolve()) if a.prompt_file else None,
        'changed': ['select recorded additional image references',
                    'use exact supplied prompt file' if a.prompt_file else
                    'retarget image tags to Picture 1; retain identity descriptions'] +
                   (['override recorded control strength'] if control_change else []),
        'world_commit': False, 'gpu': a.gpu,
        'control_strength_override': control_change,
    }, indent=2), encoding='utf-8')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu))
    started = time.time()
    with (output / 'generation.log').open('w') as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    (output / 'exit.json').write_text(json.dumps({
        'returncode': result.returncode, 'seconds': time.time() - started,
    }, indent=2))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--prompt-file', type=Path,
                        help='Exact experimental prompt; otherwise adapt recorded memory tags')
    parser.add_argument('--strength', type=float,
                        help='Explicit control-strength ablation; otherwise preserve recorded strength')
    parser.add_argument('--appearance-indices', type=int, nargs='*', default=[],
                        help='One-based additional-reference indices to retain; omitted selects none')
    main(parser.parse_args())
