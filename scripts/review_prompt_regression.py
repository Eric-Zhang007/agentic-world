"""Review completed prompt experiments with the unchanged camera evaluator."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np


def replay_pixels(left, right):
    """Decoded equality is a reproducibility diagnostic, not an identity score."""
    captures = [cv2.VideoCapture(str(path)) for path in (left, right)]
    try:
        if not all(cap.isOpened() for cap in captures):
            raise ValueError('Cannot read replay videos')
        count, error, identical = 0, 0., True
        while True:
            (ok_a, a), (ok_b, b) = [cap.read() for cap in captures]
            if ok_a != ok_b:
                raise ValueError('Replay video lengths differ')
            if not ok_a:
                break
            if a.shape != b.shape:
                raise ValueError('Replay dimensions differ')
            identical = identical and np.array_equal(a, b)
            error += float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())
            count += 1
        if not count:
            raise ValueError('Empty replay')
        return {'decoded_frames': count, 'all_decoded_pixels_equal': identical,
                'mean_absolute_channel_difference_0_255': error / count,
                'scope': 'Replay reproducibility only; not a semantic identity or spatial score'}
    finally:
        for cap in captures:
            cap.release()


def main(a):
    root = Path(__file__).resolve().parents[1]
    sources = [('exit.json', 'exit.json'), ('ablation.json', 'ablation.json'),
               ('review/review.json', 'review.json'), ('render/metadata.json', 'metadata.json')]
    for source, _ in sources:
        if not (a.historical / source).is_file():
            raise FileNotFoundError(a.historical / source)
    history = a.output / 'historical_evidence'
    history.mkdir(parents=True, exist_ok=True)
    for source, target in sources:
        if not (history / target).exists():
            shutil.copy2(a.historical / source, history / target)
        elif (history / target).read_bytes() != (a.historical / source).read_bytes():
            raise ValueError('Historical evidence changed: ' + source)
    names = ['baseline', 'remove_player_detail', 'remove_staff_detail',
             'remove_tiger_detail', 'append_architecture']
    summary = a.output / 'review_summary.json'
    results = json.loads(summary.read_text()) if summary.exists() else {}
    pending = {name for name in names if 'review' not in results.get(name, {})}
    deadline = time.monotonic() + 7200
    while pending:
        if time.monotonic() > deadline:
            raise TimeoutError('Missing generation exit records: ' + ', '.join(sorted(pending)))
        for name in names:
            directory = a.output / name
            if name not in pending or not (directory / 'exit.json').is_file():
                continue
            try:
                generation = json.loads((directory / 'exit.json').read_text())
            except json.JSONDecodeError:
                continue  # An older runner can still be writing its exit record.
            result = {'generation': generation}
            if generation['returncode'] == 0:
                video = directory / 'render/controlled.mp4'
                if name == 'baseline':
                    try:
                        result['historical_replay'] = replay_pixels(a.historical / 'render/controlled.mp4', video)
                    except ValueError as exc:
                        result['historical_replay_error'] = str(exc)
                for attempt in range(2):
                    index = attempt
                    while (directory / f'review_{index}').exists():
                        index += 1
                    review = directory / f'review_{index}'
                    command = [sys.executable, str(root / 'scripts/evaluate_camera_revisit.py'),
                               '--plan', str(a.segment / 'plan'), '--preview', str(a.segment / 'planned_preview'),
                               '--reference', str(a.segment / 'conditions/first.png'),
                               '--video', str(video), '--output', str(review)]
                    with (directory / f'review_{index}.log').open('w') as log:
                        try:
                            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=420)
                            code = completed.returncode
                        except subprocess.TimeoutExpired:
                            code = 124
                    result.setdefault('review_attempts', []).append({'directory': str(review), 'returncode': code})
                    if code == 0:
                        result['review'] = json.loads((review / 'review.json').read_text())
                        break
            results[name] = result
            pending.remove(name)
            (a.output / 'review_summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
            print('REVIEW', name, result.get('review', {}).get('overall', 'NO_VALID_REVIEW'), flush=True)
        if pending:
            time.sleep(5)
    return int(any('review' not in row for row in results.values()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['output', 'historical', 'segment']:
        p.add_argument('--' + name, type=Path, required=True)
    raise SystemExit(main(p.parse_args()))
