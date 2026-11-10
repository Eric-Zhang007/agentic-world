"""Prepare literal single-edit prompt cases from the successful replay."""
import argparse
import difflib
import json
from pathlib import Path

HEADER = 'Stable appearance descriptions from accepted video memory.'


def cases(baseline, changed):
    if baseline.count(HEADER) != 1 or changed.count(HEADER) != 1:
        raise ValueError('Expected one appearance section per prompt')
    prefix, suffix = baseline.split(HEADER)
    lines = suffix.splitlines(keepends=True)
    # These cases describe this recorded experiment, not a general prompt parser.
    expected = ['Armored monkey', 'Long rigid dark staff', 'Stocky orange tiger',
                'Monkey-like warrior', 'Long straight staff', 'Broad, stocky tiger']
    if len(lines) != 7 or any(not line.startswith(start)
                             for line, start in zip(lines[1:], expected)):
        raise ValueError('Baseline layout changed; review experiment definitions')
    new_lines = changed.split(HEADER)[1].splitlines()
    architecture = new_lines[-3:]
    if len(architecture) != 3 or not all(line.startswith(start) for line, start in zip(architecture,
            ['Carved stone monument', 'Weathered stone retaining', 'Weathered temple terrace'])):
        raise ValueError('Unexpected architectural memory block')
    result = {'baseline': baseline}
    for name, index in [('remove_player_detail', 4), ('remove_staff_detail', 5),
                        ('remove_tiger_detail', 6)]:
        result[name] = prefix + HEADER + ''.join(lines[:index] + lines[index + 1:])
    result['append_architecture'] = baseline + ('' if baseline.endswith('\n') else '\n') + '\n'.join(architecture)
    return result


def command_for(command, prompt, output):
    result = list(command)
    for flag, value in [('--prompt', prompt), ('--output', str(output))]:
        if result.count(flag) != 1:
            raise ValueError('Expected one ' + flag)
        result[result.index(flag) + 1] = value
    return result


def main(a):
    baseline = json.loads(a.baseline.read_text(encoding='utf-8'))
    changed = json.loads(a.changed.read_text(encoding='utf-8'))['command']
    command = baseline['command']
    # Verify the historical comparison, including every renderer argument.
    if command_for(changed, baseline['prompt'], command[command.index('--output') + 1]) != command:
        raise ValueError('Historical records differ beyond prompt/output')
    variants = cases(baseline['prompt'], changed[changed.index('--prompt') + 1])
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'source_changed.json').write_text(json.dumps({'command': changed}, indent=2), encoding='utf-8')
    for name, prompt in variants.items():
        directory = a.output / name
        directory.mkdir()
        (directory / 'prompt.txt').write_text(prompt, encoding='utf-8')
        (directory / 'prompt.diff').write_text(''.join(difflib.unified_diff(
            baseline['prompt'].splitlines(keepends=True), prompt.splitlines(keepends=True),
            fromfile='baseline', tofile=name)), encoding='utf-8')
        (directory / 'prepared.json').write_text(json.dumps({
            'command': command_for(command, prompt, directory.resolve() / 'render'),
            'baseline': str(a.baseline.resolve()), 'case': name, 'world_commit': False,
            'status': 'PREPARED_NOT_EXECUTED',
        }, indent=2), encoding='utf-8')
    print('Prepared cases:', ', '.join(variants))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['baseline', 'changed', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    main(p.parse_args())
